# data/featurization.py
# Rebuttal Engine v2 - Featurization (Data Gathering)
#
# This notebook handles TWO data pipelines:
# 1. Gold Standard Letters: Ingest winning rebuttals for vector search
# 2. New Denials: Pull clinical data for accounts needing rebuttals
#
# Run on Databricks Runtime 15.4 LTS ML

# =============================================================================
# CELL 1: Install Dependencies (run once per cluster)
# =============================================================================
# %pip install azure-ai-documentintelligence azure-core openai
# dbutils.library.restartPython()

# =============================================================================
# CELL 2: Imports and Configuration
# =============================================================================
import os
import json
import uuid
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit
from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType, FloatType,
    TimestampType, MapType
)

spark = SparkSession.builder.getOrCreate()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCOPE_FILTER = "sepsis"  # "sepsis" | "all"
SEPSIS_DRG_CODES = ["870", "871", "872"]

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

# Paths - UPDATE FOR YOUR ENVIRONMENT
GOLD_LETTERS_PATH = "/Workspace/Repos/your_user/fudgsicle/utils/gold_standard_rebuttals"
DENIAL_LETTERS_PATH = "/Workspace/Repos/your_user/fudgsicle/utils/Sample_Denial_Letters"

print(f"Catalog: {trgt_cat}")
print(f"Gold letters path: {GOLD_LETTERS_PATH}")

# =============================================================================
# CELL 4: Azure Credentials
# =============================================================================
AZURE_OPENAI_KEY = dbutils.secrets.get(scope='idp_etl', key='az-openai-key1')
AZURE_OPENAI_ENDPOINT = dbutils.secrets.get(scope='idp_etl', key='az-openai-base')
AZURE_DOC_INTEL_KEY = dbutils.secrets.get(scope='idp_etl', key='az-doc-intelligence-key')
AZURE_DOC_INTEL_ENDPOINT = dbutils.secrets.get(scope='idp_etl', key='az-doc-intelligence-endpoint')

print("Credentials loaded")

# =============================================================================
# CELL 5: Initialize Azure Clients
# =============================================================================
from openai import AzureOpenAI
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential

openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-10-21"
)

doc_intel_client = DocumentIntelligenceClient(
    endpoint=AZURE_DOC_INTEL_ENDPOINT,
    credential=AzureKeyCredential(AZURE_DOC_INTEL_KEY)
)

print("Azure clients initialized")

# =============================================================================
# CELL 6: Create Gold Letters Table (run once)
# =============================================================================
create_gold_table_sql = f"""
CREATE TABLE IF NOT EXISTS {GOLD_LETTERS_TABLE} (
    letter_id STRING NOT NULL,
    source_file STRING NOT NULL,
    payor STRING,
    denial_text STRING,
    rebuttal_text STRING,
    denial_embedding ARRAY<FLOAT>,
    created_at TIMESTAMP,
    metadata MAP<STRING, STRING>
)
USING DELTA
COMMENT 'Gold standard rebuttal letters with embeddings for vector search'
"""

spark.sql(create_gold_table_sql)
print(f"Table {GOLD_LETTERS_TABLE} ready")

# =============================================================================
# CELL 7: Create Inference Table (run once)
# =============================================================================
create_inference_table_sql = f"""
CREATE TABLE IF NOT EXISTS {INFERENCE_TABLE} (
    hsp_account_id STRING,
    pat_id STRING,
    pat_mrn_id STRING,
    formatted_name STRING,
    formatted_birthdate STRING,
    facility_name STRING,
    number_of_midnights INT,
    formatted_date_of_service STRING,
    claim_number STRING,
    tax_id STRING,
    npi STRING,
    code STRING,
    dx_name STRING,
    discharge_summary_note_id STRING,
    discharge_note_csn_id STRING,
    discharge_summary_text STRING,
    hp_note_id STRING,
    hp_note_csn_id STRING,
    hp_note_text STRING,
    denial_letter_text STRING,
    denial_letter_filename STRING,
    scope_filter STRING,
    featurization_timestamp STRING,
    insert_tsp TIMESTAMP
)
USING DELTA
COMMENT 'Input data for rebuttal letter generation'
"""

spark.sql(create_inference_table_sql)
print(f"Table {INFERENCE_TABLE} ready")

# #############################################################################
# PART 1: GOLD STANDARD LETTER INGESTION
# Run this section when you have new gold standard letters to ingest
# #############################################################################

# =============================================================================
# CELL 8: LLM Prompt for Splitting Rebuttal from Denial
# =============================================================================
SPLIT_PROMPT = """You are analyzing a document that contains TWO business letters combined into one file.

The document structure is:
1. FIRST: A successful rebuttal/appeal letter (from a hospital to an insurance company)
2. SECOND: The original denial letter that was being rebutted (tacked on at the end)

Each letter has typical business letter formatting: logo/letterhead, date, addresses, salutation, body, signature.

Your tasks:
1. Identify where the rebuttal letter ENDS and the original denial letter BEGINS
2. Extract the PAYOR (insurance company) name from the denial letter

Here is the document text:
---
{document_text}
---

Return a JSON object with exactly this structure:
{{
    "rebuttal_text": "The complete text of the rebuttal/appeal letter (the first letter)",
    "denial_text": "The complete text of the original denial letter (the second letter, tacked on at end)",
    "payor": "The insurance company/payor name (e.g., 'UnitedHealthcare', 'Aetna', 'Blue Cross Blue Shield', 'Humana', 'Cigna')",
    "split_confidence": 0.0 to 1.0,
    "split_reasoning": "Brief explanation of how you identified the split point"
}}

Look for indicators like:
- Change in letterhead/sender (hospital vs insurance company)
- Change in date
- Change in tone (appealing vs denying)
- Phrases like "Dear Medical Review Team" (rebuttal) vs "Dear Provider" (denial)
- References to "your denial" (rebuttal) vs "we have determined" (denial)
- The payor name is usually in the denial letter's letterhead, signature block, or referenced in the rebuttal

Return ONLY valid JSON, no markdown."""

# =============================================================================
# CELL 9: Process Gold Standard Letters
# =============================================================================
# Set to True to run gold letter ingestion
RUN_GOLD_INGESTION = False

if RUN_GOLD_INGESTION:
    print("="*60)
    print("GOLD STANDARD LETTER INGESTION")
    print("="*60)

    # List PDF files
    pdf_files = [f for f in os.listdir(GOLD_LETTERS_PATH) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files")

    gold_records = []

    for i, pdf_file in enumerate(pdf_files):
        print(f"\nProcessing {i+1}/{len(pdf_files)}: {pdf_file}")
        file_path = os.path.join(GOLD_LETTERS_PATH, pdf_file)

        # Step 1: Read PDF with Document Intelligence
        print("  Reading PDF...")
        with open(file_path, "rb") as f:
            document_bytes = f.read()

        poller = doc_intel_client.begin_analyze_document(
            model_id="prebuilt-read",
            body=AnalyzeDocumentRequest(bytes_source=document_bytes),
        )
        result = poller.result()

        text_parts = []
        for page in result.pages:
            for line in page.lines:
                text_parts.append(line.content)
        full_text = "\n".join(text_parts)
        print(f"  Extracted {len(full_text)} characters")

        # Step 2: Split rebuttal and denial using LLM
        print("  Splitting rebuttal and denial...")
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "Extract the two letters accurately. Return only valid JSON."},
                {"role": "user", "content": SPLIT_PROMPT.format(document_text=full_text)}
            ],
            temperature=0.1,
            max_tokens=8000
        )

        raw_response = response.choices[0].message.content.strip()
        json_str = raw_response
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        split_result = json.loads(json_str)
        rebuttal_text = split_result.get("rebuttal_text", "")
        denial_text = split_result.get("denial_text", "")
        payor = split_result.get("payor", "Unknown")
        split_confidence = str(split_result.get("split_confidence", 0.0))
        split_reasoning = split_result.get("split_reasoning", "")

        print(f"  Rebuttal: {len(rebuttal_text)} chars, Denial: {len(denial_text)} chars")
        print(f"  Payor: {payor}")
        print(f"  Split confidence: {split_confidence}")

        # Step 3: Generate embedding of denial (for matching new denials)
        print("  Generating embedding...")
        embed_text = denial_text[:30000] if len(denial_text) > 30000 else denial_text
        embed_response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=embed_text
        )
        denial_embedding = embed_response.data[0].embedding
        print(f"  Embedding dimension: {len(denial_embedding)}")

        # Step 4: Create record
        record = {
            "letter_id": str(uuid.uuid4()),
            "source_file": pdf_file,
            "payor": payor,
            "denial_text": denial_text,
            "rebuttal_text": rebuttal_text,
            "denial_embedding": denial_embedding,
            "created_at": datetime.now(),
            "metadata": {"split_confidence": split_confidence, "split_reasoning": split_reasoning},
        }
        gold_records.append(record)
        print(f"  Record created: {record['letter_id']}")

    print(f"\nProcessed {len(gold_records)} gold standard letters")

    # Write to Delta table
    if gold_records:
        schema = StructType([
            StructField("letter_id", StringType(), False),
            StructField("source_file", StringType(), False),
            StructField("payor", StringType(), True),
            StructField("denial_text", StringType(), True),
            StructField("rebuttal_text", StringType(), True),
            StructField("denial_embedding", ArrayType(FloatType()), True),
            StructField("created_at", TimestampType(), True),
            StructField("metadata", MapType(StringType(), StringType()), True),
        ])

        gold_df = spark.createDataFrame(gold_records, schema)
        gold_df.write.format("delta").mode("append").saveAsTable(GOLD_LETTERS_TABLE)
        print(f"Wrote {len(gold_records)} records to {GOLD_LETTERS_TABLE}")

    # Verify
    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {GOLD_LETTERS_TABLE}").collect()[0]["cnt"]
    print(f"Total records in gold letters table: {count}")

else:
    print("Gold ingestion skipped (set RUN_GOLD_INGESTION = True to run)")

# #############################################################################
# PART 2: NEW DENIAL FEATURIZATION
# Run this section to prepare new denial cases for inference
# #############################################################################

# =============================================================================
# CELL 10: Target Accounts Configuration
# =============================================================================
# Format: (HSP_ACCOUNT_ID, denial_letter_filename)
TARGET_ACCOUNTS = [
    # UPDATE THESE with your test cases:
    # ("123456789", "denial_letter_1.pdf"),
    # ("987654321", "denial_letter_2.pdf"),
]

print(f"\nTarget accounts configured: {len(TARGET_ACCOUNTS)}")

# =============================================================================
# CELL 11: Clinical Data Query
# =============================================================================
RUN_DENIAL_FEATURIZATION = False

if RUN_DENIAL_FEATURIZATION and len(TARGET_ACCOUNTS) > 0:
    print("="*60)
    print("NEW DENIAL FEATURIZATION")
    print("="*60)

    account_ids = [a[0] for a in TARGET_ACCOUNTS]
    account_list = ",".join(f"'{a}'" for a in account_ids)

    # Query clinical data from Clarity
    print("\nQuerying clinical data from Clarity...")

    clinical_query = f"""
    WITH target_accounts AS (
        SELECT explode(array({account_list})) AS hsp_account_id
    ),
    encounters AS (
        SELECT
            pe.pat_enc_csn_id,
            pe.pat_id,
            pe.hsp_account_id,
            pe.hosp_admsn_time,
            pe.hosp_disch_time,
            ROW_NUMBER() OVER (PARTITION BY pe.hsp_account_id ORDER BY pe.hosp_admsn_time DESC) AS rn
        FROM prod.clarity_cur.pat_enc_hsp_har_enh pe
        INNER JOIN target_accounts ta ON pe.hsp_account_id = ta.hsp_account_id
        WHERE pe.hosp_admsn_time IS NOT NULL
    ),
    latest_encounters AS (
        SELECT * FROM encounters WHERE rn = 1
    ),
    discharge_notes AS (
        SELECT
            e.hsp_account_id,
            hno.note_id AS discharge_note_id,
            hno.pat_enc_csn_id AS discharge_note_csn_id,
            CONCAT_WS(' ', COLLECT_LIST(hnt.note_text)) AS discharge_summary_text
        FROM prod.clarity_cur.hno_info_enh hno
        INNER JOIN latest_encounters e ON hno.pat_enc_csn_id = e.pat_enc_csn_id
        INNER JOIN prod.clarity_cur.hno_note_text_enh hnt ON hno.note_id = hnt.note_id
        WHERE hno.note_type_c = 2
        GROUP BY e.hsp_account_id, hno.note_id, hno.pat_enc_csn_id
    ),
    hp_notes AS (
        SELECT
            e.hsp_account_id,
            hno.note_id AS hp_note_id,
            hno.pat_enc_csn_id AS hp_note_csn_id,
            CONCAT_WS(' ', COLLECT_LIST(hnt.note_text)) AS hp_note_text
        FROM prod.clarity_cur.hno_info_enh hno
        INNER JOIN latest_encounters e ON hno.pat_enc_csn_id = e.pat_enc_csn_id
        INNER JOIN prod.clarity_cur.hno_note_text_enh hnt ON hno.note_id = hnt.note_id
        WHERE hno.note_type_c = 1
        GROUP BY e.hsp_account_id, hno.note_id, hno.pat_enc_csn_id
    ),
    patient_info AS (
        SELECT
            e.hsp_account_id,
            p.pat_id,
            p.pat_mrn_id,
            CONCAT(p.pat_last_name, ', ', p.pat_first_name) AS formatted_name,
            DATE_FORMAT(p.birth_date, 'MM/dd/yyyy') AS formatted_birthdate
        FROM prod.clarity_cur.patient_enh p
        INNER JOIN latest_encounters e ON p.pat_id = e.pat_id
    ),
    account_info AS (
        SELECT
            ha.hsp_account_id,
            ha.prim_enc_csn_id,
            f.facility_name,
            DATEDIFF(ha.disch_date_time, ha.adm_date_time) AS number_of_midnights,
            CONCAT(DATE_FORMAT(ha.adm_date_time, 'MM/dd/yyyy'), ' - ',
                   DATE_FORMAT(ha.disch_date_time, 'MM/dd/yyyy')) AS formatted_date_of_service
        FROM prod.clarity_cur.hsp_account_enh ha
        INNER JOIN target_accounts ta ON ha.hsp_account_id = ta.hsp_account_id
        LEFT JOIN prod.clarity.zc_loc_facility f ON ha.loc_id = f.facility_id
    ),
    claim_info AS (
        SELECT
            hcd.hsp_account_id,
            FIRST(hcd.claim_number) AS claim_number,
            FIRST(hcd.tax_id) AS tax_id,
            FIRST(hcd.npi) AS npi
        FROM prod.clarity.hsp_claim_detail hcd
        INNER JOIN target_accounts ta ON hcd.hsp_account_id = ta.hsp_account_id
        GROUP BY hcd.hsp_account_id
    ),
    diagnosis AS (
        SELECT
            hadl.hsp_account_id,
            FIRST(edg.code) AS code,
            FIRST(edg.dx_name) AS dx_name
        FROM prod.clarity_cur.hsp_acct_dx_list_enh hadl
        INNER JOIN target_accounts ta ON hadl.hsp_account_id = ta.hsp_account_id
        INNER JOIN prod.clarity_cur.edg_current_icd10_enh edg ON hadl.dx_id = edg.dx_id
        WHERE hadl.line = 1
        GROUP BY hadl.hsp_account_id
    )
    SELECT
        ta.hsp_account_id,
        pi.pat_id,
        pi.pat_mrn_id,
        pi.formatted_name,
        pi.formatted_birthdate,
        ai.facility_name,
        ai.number_of_midnights,
        ai.formatted_date_of_service,
        ci.claim_number,
        ci.tax_id,
        ci.npi,
        d.code,
        d.dx_name,
        CAST(dn.discharge_note_id AS STRING) AS discharge_summary_note_id,
        CAST(dn.discharge_note_csn_id AS STRING) AS discharge_note_csn_id,
        dn.discharge_summary_text,
        CAST(hp.hp_note_id AS STRING) AS hp_note_id,
        CAST(hp.hp_note_csn_id AS STRING) AS hp_note_csn_id,
        hp.hp_note_text
    FROM target_accounts ta
    LEFT JOIN patient_info pi ON ta.hsp_account_id = pi.hsp_account_id
    LEFT JOIN account_info ai ON ta.hsp_account_id = ai.hsp_account_id
    LEFT JOIN claim_info ci ON ta.hsp_account_id = ci.hsp_account_id
    LEFT JOIN diagnosis d ON ta.hsp_account_id = d.hsp_account_id
    LEFT JOIN discharge_notes dn ON ta.hsp_account_id = dn.hsp_account_id
    LEFT JOIN hp_notes hp ON ta.hsp_account_id = hp.hsp_account_id
    """

    clinical_df = spark.sql(clinical_query).toPandas()
    print(f"Retrieved clinical data for {len(clinical_df)} accounts")

    # Read denial letters
    print("\nReading denial letters...")
    denial_texts = {}
    filename_map = {}

    for hsp_account_id, filename in TARGET_ACCOUNTS:
        file_path = os.path.join(DENIAL_LETTERS_PATH, filename)
        filename_map[hsp_account_id] = filename

        try:
            # Check file extension
            _, ext = os.path.splitext(filename.lower())

            if ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    denial_texts[hsp_account_id] = f.read()
            else:
                # Use Document Intelligence
                with open(file_path, "rb") as f:
                    document_bytes = f.read()

                poller = doc_intel_client.begin_analyze_document(
                    model_id="prebuilt-read",
                    body=AnalyzeDocumentRequest(bytes_source=document_bytes),
                )
                result = poller.result()

                text_parts = []
                for page in result.pages:
                    for line in page.lines:
                        text_parts.append(line.content)
                denial_texts[hsp_account_id] = "\n".join(text_parts)

            print(f"  {filename}: {len(denial_texts[hsp_account_id])} chars")
        except Exception as e:
            print(f"  ERROR reading {filename}: {e}")
            denial_texts[hsp_account_id] = None

    # Add denial letter text and metadata
    clinical_df['denial_letter_text'] = clinical_df['hsp_account_id'].map(denial_texts)
    clinical_df['denial_letter_filename'] = clinical_df['hsp_account_id'].map(filename_map)
    clinical_df['scope_filter'] = SCOPE_FILTER
    clinical_df['featurization_timestamp'] = datetime.now().isoformat()

    print(f"\nFeaturized dataset: {len(clinical_df)} rows")
    print(f"Columns: {list(clinical_df.columns)}")

    # Preview
    print("\nPREVIEW:")
    print(clinical_df[['hsp_account_id', 'formatted_name', 'denial_letter_filename']].to_string())

    # Write to table
    WRITE_TO_TABLE = False  # Set to True to write

    if WRITE_TO_TABLE:
        spark_df = spark.createDataFrame(clinical_df)
        spark_df = spark_df.withColumn("insert_tsp", current_timestamp())
        spark_df.write.mode("append").saveAsTable(INFERENCE_TABLE)
        print(f"\nWrote {len(clinical_df)} rows to {INFERENCE_TABLE}")
    else:
        print("\nTo write to table, set WRITE_TO_TABLE = True")

else:
    if len(TARGET_ACCOUNTS) == 0:
        print("\nNo target accounts configured. Update TARGET_ACCOUNTS list.")
    else:
        print("\nDenial featurization skipped (set RUN_DENIAL_FEATURIZATION = True)")

# =============================================================================
# CELL 12: Verify Tables
# =============================================================================
print("\n" + "="*60)
print("TABLE STATUS")
print("="*60)

try:
    gold_count = spark.sql(f"SELECT COUNT(*) as cnt FROM {GOLD_LETTERS_TABLE}").collect()[0]["cnt"]
    print(f"Gold letters table: {gold_count} records")
except Exception as e:
    print(f"Gold letters table: Error - {e}")

try:
    inference_count = spark.sql(f"SELECT COUNT(*) as cnt FROM {INFERENCE_TABLE}").collect()[0]["cnt"]
    print(f"Inference table: {inference_count} records")
except Exception as e:
    print(f"Inference table: Error - {e}")

print("\nFeaturization complete.")
