# model/inference.py
# Rebuttal Engine v2 - Inference (Letter Generation)
#
# Pipeline: Parse Denial → Find Similar Gold Letter → Generate Rebuttal
#
# Key Feature: Vector search finds most similar past denial from gold letters,
# then uses that successful rebuttal as context for the Writer.
#
# Run on Databricks Runtime 15.4 LTS ML

# =============================================================================
# CELL 1: Install Dependencies (run once per cluster)
# =============================================================================
# %pip install openai
# dbutils.library.restartPython()

# =============================================================================
# CELL 2: Imports and Configuration
# =============================================================================
import os
import json
import math
from datetime import datetime, date
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp

spark = SparkSession.builder.getOrCreate()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCOPE_FILTER = "sepsis"  # "sepsis" | "all"
SEPSIS_DRG_CODES = ["870", "871", "872"]
MATCH_SCORE_THRESHOLD = 0.7  # Minimum similarity to use gold letter

# =============================================================================
# CELL 3: Environment Setup
# =============================================================================
trgt_cat = os.environ.get('trgt_cat', 'dev')
if trgt_cat == 'dev':
    spark.sql('USE CATALOG prod;')
else:
    spark.sql(f'USE CATALOG {trgt_cat};')

# Table names
GOLD_LETTERS_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_gold_letters"
INFERENCE_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_inference"
INFERENCE_SCORE_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_inference_score"

print(f"Catalog: {trgt_cat}")

# =============================================================================
# CELL 4: Azure OpenAI Setup
# =============================================================================
from openai import AzureOpenAI

api_key = dbutils.secrets.get(scope='idp_etl', key='az-openai-key1')
azure_endpoint = dbutils.secrets.get(scope='idp_etl', key='az-openai-base')
api_version = '2024-10-21'
model = 'gpt-4.1'
embedding_model = 'text-embedding-ada-002'

client = AzureOpenAI(api_key=api_key, azure_endpoint=azure_endpoint, api_version=api_version)
print(f"Azure OpenAI client initialized (model: {model})")

# =============================================================================
# CELL 5: Create Score Table (run once)
# =============================================================================
create_score_table_sql = f"""
CREATE TABLE IF NOT EXISTS {INFERENCE_SCORE_TABLE} (
    hsp_account_id STRING,
    pat_mrn_id STRING,
    discharge_summary_note_id STRING,
    discharge_note_csn_id STRING,
    hp_note_id STRING,
    hp_note_csn_id STRING,
    letter_type STRING,
    letter_text STRING,
    letter_curated_date DATE,
    denial_info_json STRING,
    gold_letter_used STRING,
    gold_letter_score FLOAT,
    pipeline_version STRING,
    insert_tsp TIMESTAMP
)
USING DELTA
COMMENT 'Generated rebuttal letters'
"""

spark.sql(create_score_table_sql)
print(f"Table {INFERENCE_SCORE_TABLE} ready")

# =============================================================================
# CELL 6: Check for New Records
# =============================================================================
try:
    last_processed_ts = spark.sql(f"""
        SELECT COALESCE(MAX(insert_tsp), TIMESTAMP'2020-01-01 00:00:00') AS last_ts
        FROM {INFERENCE_SCORE_TABLE}
    """).collect()[0]["last_ts"]
except Exception:
    last_processed_ts = "2020-01-01 00:00:00"

n_new_rows = spark.sql(f"""
    WITH scored_accounts AS (
        SELECT DISTINCT hsp_account_id
        FROM {INFERENCE_SCORE_TABLE}
    )
    SELECT COUNT(*) AS cnt
    FROM {INFERENCE_TABLE} src
    LEFT JOIN scored_accounts sa ON src.hsp_account_id = sa.hsp_account_id
    WHERE src.insert_tsp > TIMESTAMP'{last_processed_ts}'
       OR sa.hsp_account_id IS NULL
""").collect()[0]["cnt"]

print(f"New rows to process: {n_new_rows}")

# =============================================================================
# CELL 7: Load Gold Letters (for vector search)
# =============================================================================
print("\nLoading gold standard letters...")

try:
    gold_letters_df = spark.sql(f"""
        SELECT letter_id, source_file, payor, denial_text, rebuttal_text, denial_embedding, metadata
        FROM {GOLD_LETTERS_TABLE}
    """)
    gold_letters = gold_letters_df.collect()
    gold_letters_cache = [
        {
            "letter_id": row["letter_id"],
            "source_file": row["source_file"],
            "payor": row["payor"],
            "denial_text": row["denial_text"],
            "rebuttal_text": row["rebuttal_text"],
            "denial_embedding": list(row["denial_embedding"]) if row["denial_embedding"] else None,
            "metadata": dict(row["metadata"]) if row["metadata"] else {},
        }
        for row in gold_letters
    ]
    print(f"Loaded {len(gold_letters_cache)} gold standard letters")
except Exception as e:
    print(f"Warning: Could not load gold letters: {e}")
    gold_letters_cache = []

# =============================================================================
# CELL 8: Parser Prompt
# =============================================================================
PARSER_PROMPT = '''You are a medical billing expert extracting information from denial letters.

# Task
Extract ALL relevant information from this denial letter into structured JSON.

# Denial Letter
{denial_letter_text}

# Output Format
Return ONLY valid JSON (no markdown):
{{
  "denial_date": "YYYY-MM-DD or null",
  "payer_name": "Insurance company name",
  "payer_address": "Full mailing address or null",
  "reviewer_name": "Name and credentials or null",
  "original_drg": "Billed DRG (e.g., '871')",
  "proposed_drg": "Payer's proposed DRG (e.g., '872')",
  "administrative_data": {{
    "claim_reference_number": "Primary claim/reference number",
    "member_id": "Patient member ID",
    "authorization_number": "Prior auth number or null",
    "date_of_service": "Admission date or date range",
    "other_identifiers": {{}}
  }},
  "denial_reasons": [{{
    "type": "clinical_validation | medical_necessity | level_of_care | coding | other",
    "summary": "Brief summary",
    "specific_arguments": ["Each specific argument made"],
    "payer_quote": "Direct quote if available"
  }}],
  "is_sepsis_related": true/false,
  "is_single_issue": true/false
}}'''

# =============================================================================
# CELL 9: Writer Prompt
# =============================================================================
WRITER_PROMPT = '''You are a clinical coding expert writing a DRG validation appeal letter for Mercy Hospital.

# Denial Information
{denial_info_json}

# Clinical Notes (PRIMARY EVIDENCE - use these first)
## Discharge Summary
{discharge_summary}

## H&P Note
{hp_note}

# Gold Standard Letter (WINNING REBUTTAL - learn from this)
{gold_letter_section}

# Patient Information
{patient_info_json}

# Instructions
{gold_letter_instructions}
1. ADDRESS EACH DENIAL ARGUMENT - quote the payer, then refute
2. CITE CLINICAL EVIDENCE from provider notes FIRST (best source)
3. Follow the Mercy Hospital template structure exactly
4. Include specific clinical values (lactate 2.4, MAP 62, etc.)
5. DELETE sections that don't apply to this patient

# Template Structure
Return the complete letter text following this structure:

Mercy Hospital
Payor Audits & Denials Dept
ATTN: Compliance Manager
2115 S Fremont Ave - Ste LL1
Springfield, MO 65804

{current_date}

[PAYOR ADDRESS]

First Level Appeal

Beneficiary Name: [NAME]
DOB: [DOB]
Claim reference #: [CLAIM_REF]
Hospital Account #: [HSP_ACCOUNT_ID]
Date of Service: [DOS]

Dear [REVIEWER]:

[Opening paragraph about receiving DRG review...]

Justification for Appeal:
[Why we disagree...]

Rationale:
[Quote payer's argument, then provide clinical evidence...]

Infection Source: [...]
Organ Dysfunction: [List each with values...]
SIRS Criteria Met: [List each with values...]
[Other relevant sections...]

Hospital Course:
[Narrative from clinical notes...]

[Summary paragraph...]

Conclusion:
We anticipate our original DRG of [X] will be approved.

[Contact info and signature...]

Return ONLY the letter text, no JSON wrapper.'''

# =============================================================================
# CELL 10: Process Each Row
# =============================================================================
if n_new_rows == 0:
    print("No new rows to process")
else:
    print(f"\nProcessing {n_new_rows} rows...")

    # Pull unprocessed rows
    df = spark.sql(f"""
        WITH scored_accounts AS (
            SELECT DISTINCT hsp_account_id
            FROM {INFERENCE_SCORE_TABLE}
        )
        SELECT src.*
        FROM {INFERENCE_TABLE} src
        LEFT JOIN scored_accounts sa ON src.hsp_account_id = sa.hsp_account_id
        WHERE src.insert_tsp > TIMESTAMP'{last_processed_ts}'
           OR sa.hsp_account_id IS NULL
    """).toPandas()

    print(f"Pulled {len(df)} rows for processing")

    results = []

    for idx, row in df.iterrows():
        hsp_account_id = row.get("hsp_account_id", "unknown")
        print(f"\n{'='*60}")
        print(f"Processing {idx+1}/{len(df)}: {hsp_account_id}")
        print(f"{'='*60}")

        result = {
            "hsp_account_id": hsp_account_id,
            "status": None,
            "letter_text": None,
            "denial_info": None,
            "gold_letter_used": None,
            "gold_letter_score": None,
        }

        # =====================================================================
        # STEP 1: Parse denial letter
        # =====================================================================
        denial_text = row.get("denial_letter_text", "")
        if not denial_text or str(denial_text).strip() == "" or denial_text is None:
            result["status"] = "no_denial_text"
            result["error"] = "No denial letter text"
            results.append(result)
            print("  SKIP: No denial letter text")
            continue

        print("  Step 1: Parsing denial letter...")
        parser_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Extract information accurately. Return only valid JSON."},
                {"role": "user", "content": PARSER_PROMPT.format(denial_letter_text=denial_text)}
            ],
            temperature=0,
            max_tokens=2000
        )

        raw_parser = parser_response.choices[0].message.content.strip()
        json_str = raw_parser
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        try:
            denial_info = json.loads(json_str)
        except json.JSONDecodeError as e:
            result["status"] = "parse_error"
            result["error"] = f"JSON parse failed: {e}"
            results.append(result)
            print(f"  ERROR: JSON parse failed")
            continue

        result["denial_info"] = denial_info
        print(f"  Parsed: DRG {denial_info.get('original_drg')} -> {denial_info.get('proposed_drg')}")

        # =====================================================================
        # STEP 2: Check scope
        # =====================================================================
        if SCOPE_FILTER == "sepsis":
            if not denial_info.get("is_sepsis_related"):
                result["status"] = "out_of_scope"
                result["reason"] = "Not sepsis-related"
                results.append(result)
                print("  SKIP: Not sepsis-related")
                continue

        # =====================================================================
        # STEP 3: Vector search for best matching gold letter
        # =====================================================================
        print("  Step 2: Finding similar gold standard letter...")

        gold_letter = None
        gold_letter_score = 0.0

        if gold_letters_cache:
            # Generate embedding for this denial
            embed_text = denial_text[:30000] if len(denial_text) > 30000 else denial_text
            embed_response = client.embeddings.create(
                model=embedding_model,
                input=embed_text
            )
            query_embedding = embed_response.data[0].embedding

            # Compute cosine similarity against all gold letters
            best_score = 0.0
            best_match = None

            for letter in gold_letters_cache:
                if letter["denial_embedding"]:
                    # Cosine similarity
                    vec1 = query_embedding
                    vec2 = letter["denial_embedding"]
                    dot_product = sum(a * b for a, b in zip(vec1, vec2))
                    norm1 = math.sqrt(sum(a * a for a in vec1))
                    norm2 = math.sqrt(sum(b * b for b in vec2))

                    if norm1 > 0 and norm2 > 0:
                        similarity = dot_product / (norm1 * norm2)
                        if similarity > best_score:
                            best_score = similarity
                            best_match = letter

            gold_letter_score = best_score

            if best_match and best_score >= MATCH_SCORE_THRESHOLD:
                gold_letter = best_match
                print(f"  Found match: {gold_letter['source_file']} | Payor: {gold_letter.get('payor', 'Unknown')} | Score: {best_score:.3f}")
            else:
                print(f"  No good match (best score: {best_score:.3f}, threshold: {MATCH_SCORE_THRESHOLD})")

        else:
            print("  No gold letters loaded - using template only")

        result["gold_letter_used"] = gold_letter["letter_id"] if gold_letter else None
        result["gold_letter_score"] = gold_letter_score

        # =====================================================================
        # STEP 4: Generate rebuttal letter
        # =====================================================================
        print("  Step 3: Generating rebuttal letter...")

        current_date_str = date.today().strftime("%m/%d/%Y")

        patient_info = {
            "formatted_name": row.get("formatted_name", ""),
            "formatted_birthdate": row.get("formatted_birthdate", ""),
            "hsp_account_id": row.get("hsp_account_id", ""),
            "claim_number": row.get("claim_number", ""),
            "formatted_date_of_service": row.get("formatted_date_of_service", ""),
            "facility_name": row.get("facility_name", ""),
            "number_of_midnights": row.get("number_of_midnights", ""),
            "code": row.get("code", ""),
            "dx_name": row.get("dx_name", ""),
        }

        # Build gold letter section
        if gold_letter:
            gold_letter_section = f"""## THIS LETTER WON A SIMILAR APPEAL - LEARN FROM IT
Source: {gold_letter.get('source_file', 'Unknown')}
Payor: {gold_letter.get('payor', 'Unknown')}
Match Score: {gold_letter_score:.3f}

### Winning Rebuttal Text:
{gold_letter['rebuttal_text']}
"""
            gold_letter_instructions = """**CRITICAL: A gold standard letter that won a similar appeal is provided above.**
- Study how it structures arguments and presents clinical evidence
- Emulate its persuasive techniques and medical reasoning
- Adapt its successful patterns to this patient's specific situation
- Do NOT copy verbatim - adapt the approach with this patient's actual clinical data

"""
        else:
            gold_letter_section = "No similar winning rebuttal available. Use the Mercy template structure."
            gold_letter_instructions = ""

        writer_prompt = WRITER_PROMPT.format(
            denial_info_json=json.dumps(denial_info, indent=2, default=str),
            discharge_summary=row.get("discharge_summary_text", "Not available"),
            hp_note=row.get("hp_note_text", "Not available"),
            gold_letter_section=gold_letter_section,
            gold_letter_instructions=gold_letter_instructions,
            patient_info_json=json.dumps(patient_info, indent=2),
            current_date=current_date_str,
        )

        writer_response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical coding expert writing DRG appeal letters. "
                        "Prioritize evidence from provider notes. Be thorough and specific."
                    )
                },
                {"role": "user", "content": writer_prompt}
            ],
            temperature=0.2,
            max_tokens=4000
        )

        letter_text = writer_response.choices[0].message.content.strip()
        result["status"] = "success"
        result["letter_text"] = letter_text
        results.append(result)

        print(f"  SUCCESS: Generated {len(letter_text)} character letter")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    success = sum(1 for r in results if r["status"] == "success")
    out_of_scope = sum(1 for r in results if r["status"] == "out_of_scope")
    errors = sum(1 for r in results if r["status"] not in ("success", "out_of_scope"))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Success: {success}")
    print(f"  Out of Scope: {out_of_scope}")
    print(f"  Errors/No Data: {errors}")

# =============================================================================
# CELL 11: Write Results to Table
# =============================================================================
WRITE_TO_TABLE = False  # Set to True to write to Delta table

if n_new_rows > 0 and 'results' in dir() and len(results) > 0:
    # Build output dataframe
    output_rows = []

    for r, orig_row in zip(results, df.itertuples()):
        if r["status"] == "success":
            output_rows.append({
                "hsp_account_id": r["hsp_account_id"],
                "pat_mrn_id": getattr(orig_row, 'pat_mrn_id', None),
                "formatted_name": getattr(orig_row, 'formatted_name', None),
                "discharge_summary_note_id": getattr(orig_row, 'discharge_summary_note_id', None),
                "discharge_note_csn_id": getattr(orig_row, 'discharge_note_csn_id', None),
                "hp_note_id": getattr(orig_row, 'hp_note_id', None),
                "hp_note_csn_id": getattr(orig_row, 'hp_note_csn_id', None),
                "letter_type": "Sepsis_v2",
                "letter_text": r["letter_text"],
                "letter_curated_date": datetime.now().date(),
                "denial_info_json": json.dumps(r.get("denial_info", {}), default=str),
                "gold_letter_used": r.get("gold_letter_used"),
                "gold_letter_score": r.get("gold_letter_score"),
                "pipeline_version": "rebuttal_engine_v2",
            })

    if output_rows:
        print(f"\n{len(output_rows)} letters ready to write")

        if WRITE_TO_TABLE:
            import pandas as pd
            output_df = pd.DataFrame(output_rows)
            spark_df = spark.createDataFrame(output_df)
            spark_df = spark_df.withColumn("insert_tsp", current_timestamp())
            spark_df.write.mode("append").saveAsTable(INFERENCE_SCORE_TABLE)
            print(f"Wrote {len(output_df)} letters to {INFERENCE_SCORE_TABLE}")
        else:
            print("To write to table, set WRITE_TO_TABLE = True")

        # Preview first letter
        print(f"\n{'='*60}")
        print("PREVIEW: First Generated Letter")
        print(f"{'='*60}")
        print(output_rows[0]["letter_text"][:2000])
        print("...")
    else:
        print("No successful letters to write")

# =============================================================================
# CELL 12: POC - Export to DOCX for User Feedback
# =============================================================================
EXPORT_TO_DOCX = True  # Set to True for POC output
DOCX_OUTPUT_PATH = "/Workspace/Repos/your_user/fudgsicle/output"  # UPDATE THIS

if EXPORT_TO_DOCX and 'output_rows' in dir() and len(output_rows) > 0:
    print(f"\n{'='*60}")
    print("EXPORTING TO DOCX")
    print(f"{'='*60}")

    from docx import Document
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    os.makedirs(DOCX_OUTPUT_PATH, exist_ok=True)

    for row in output_rows:
        hsp_account_id = row["hsp_account_id"]
        patient_name = row.get("formatted_name", "Unknown")
        letter_text = row["letter_text"]
        denial_info = json.loads(row["denial_info_json"]) if row["denial_info_json"] else {}

        doc = Document()

        # Title
        title = doc.add_heading(f'Rebuttal Letter', level=1)

        # Metadata section
        meta = doc.add_paragraph()
        meta.add_run("Generated: ").bold = True
        meta.add_run(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        meta.add_run("Account: ").bold = True
        meta.add_run(f"{hsp_account_id}\n")
        meta.add_run("Patient: ").bold = True
        meta.add_run(f"{patient_name}\n")
        meta.add_run("Payor: ").bold = True
        meta.add_run(f"{denial_info.get('payer_name', 'Unknown')}\n")
        meta.add_run("DRG: ").bold = True
        meta.add_run(f"{denial_info.get('original_drg', '?')} → {denial_info.get('proposed_drg', '?')}\n")
        meta.add_run("Gold Letter: ").bold = True
        if row.get("gold_letter_used"):
            meta.add_run(f"{row['gold_letter_used']} (score: {row.get('gold_letter_score', 0):.3f})\n")
        else:
            meta.add_run("None\n")

        doc.add_paragraph("─" * 60)

        # Letter content
        for paragraph in letter_text.split('\n\n'):
            if paragraph.strip():
                p = doc.add_paragraph(paragraph.strip())
                p.paragraph_format.space_after = Pt(12)

        # Save
        safe_name = "".join(c for c in patient_name if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"{hsp_account_id}_{safe_name}_rebuttal.docx"
        filepath = os.path.join(DOCX_OUTPUT_PATH, filename)
        doc.save(filepath)
        print(f"  Saved: {filename}")

    print(f"\nDOCX files saved to: {DOCX_OUTPUT_PATH}")

print("\nInference complete.")
