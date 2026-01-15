# Sepsis Appeal Engine - Master Prompt

**Last Updated:** 2026-01-15
**Repo:** https://github.com/michaeljoyce217/SEPSIS

---

## Project Overview

**Goal:** Automated generation of DRG appeal letters for sepsis-related insurance denials (DRG 870/871/872).

**Architecture:** Single-letter processing pipeline using Azure OpenAI GPT-4.1

**Platform:** Databricks on Azure with Unity Catalog

---

## Repository Structure

```
SEPSIS/
├── data/
│   └── featurization.py        # ONE-TIME: Knowledge base ingestion (gold letters + propel)
├── model/
│   └── inference.py            # MAIN: Single-letter end-to-end processing
├── utils/
│   ├── gold_standard_appeals/  # Past winning appeal letters (PDFs) + default template
│   ├── sample_denial_letters/  # New denial letters to process (PDFs)
│   ├── propel_data/            # Clinical criteria definitions (PDFs)
│   └── outputs/                # Generated appeal letters (timestamped folders)
├── docs/
│   └── rebuttal-engine-overview.html  # Executive overview with Technical Architecture
├── compare_denials.py          # Utility: check for duplicate denials
├── test_queries.sql            # Validation queries for Unity Catalog
├── README.md                   # Project documentation
└── MASTER_PROMPT.md            # This file
```

---

## Pipeline Architecture

### One-Time Setup: featurization.py
Run once to populate knowledge base tables:

| Step | Technology | Function |
|------|------------|----------|
| Gold Letter Parsing | Azure AI Document Intelligence + GPT-4.1 | Extract appeal/denial from gold PDFs |
| Denial Embedding | text-embedding-ada-002 | Generate 1536-dim vectors for similarity search |
| Propel Extraction | GPT-4.1 | Extract key clinical criteria from Propel PDFs |

### Per-Letter Processing: inference.py
Run for each denial letter:

| Step | Technology | Function |
|------|------------|----------|
| 1. PDF Parse | Azure AI Document Intelligence | OCR extraction from denial PDF |
| 2. Vector Search | Cosine Similarity | Find best-matching gold letter (uses denial text only) |
| 3. Info Extract | GPT-4.1 | Extract: account_id, payor, DRGs, is_sepsis |
| 4. Clarity Query | Spark SQL | Get 14 clinical note types for this account |
| 5. Note Extraction | GPT-4.1 (parallel) | Extract clinical data with timestamps from long notes |
| 6. Letter Generation | GPT-4.1 | Generate appeal using gold letter + clinical evidence |
| 7. Export | python-docx | Output DOCX for CDI review |

---

## Unity Catalog Tables

| Table | Purpose |
|-------|---------|
| `dev.fin_ds.fudgesicle_gold_letters` | Past winning appeals with denial embeddings |
| `dev.fin_ds.fudgesicle_propel_data` | Official clinical criteria (definition_summary for prompts) |

Note: No intermediate inference tables - single-letter processing queries Clarity directly.

---

## Key Features

### 14 Clinical Note Types (from Epic Clarity)
Progress Notes, Consults, H&P, Discharge Summary, ED Notes, Initial Assessments, ED Triage Notes, ED Provider Notes, Addendum Note, Hospital Course, Subjective & Objective, Assessment & Plan Note, Nursing Note, Code Documentation

### Smart Note Extraction
Notes >8,000 chars are automatically extracted via LLM to pull relevant clinical data WITH timestamps (e.g., "03/15/2024 08:00: Lactate 4.2, MAP 63").

### Propel Definition Summary
Full Propel PDFs are processed at ingestion time - LLM extracts key clinical criteria into `definition_summary` field for efficient prompt inclusion.

### Default Template Fallback
When vector search doesn't find a good match (score < 0.7), the system falls back to `default_sepsis_appeal_template.docx` as a structural guide.

### Single-Letter Processing
Each denial is processed end-to-end in one run - no batch processing, no intermediate tables. This:
- Eliminates driver memory issues
- Matches production workflow (Epic workqueue feeds one case at a time)
- Simplifies debugging and testing

### Output Location
Appeal letters are saved to `utils/outputs/` with filename format: `{account_id}_{patient_name}_appeal.docx`

---

## Configuration

### featurization.py Flags
| Flag | Default | Description |
|------|---------|-------------|
| `RUN_GOLD_INGESTION` | False | Process gold standard letter PDFs |
| `RUN_PROPEL_INGESTION` | False | Process Propel definition PDFs |

### inference.py Settings
| Setting | Default | Description |
|---------|---------|-------------|
| `DENIAL_PDF_PATH` | (required) | Path to denial letter PDF to process |
| `KNOWN_ACCOUNT_ID` | None | Account ID (if known from Epic workqueue) |
| `SCOPE_FILTER` | "sepsis" | Which denial types to process |
| `MATCH_SCORE_THRESHOLD` | 0.7 | Minimum similarity for gold letter match |
| `NOTE_EXTRACTION_THRESHOLD` | 8000 | Char limit before LLM extraction |
| `EXPORT_TO_DOCX` | True | Export as Word documents |

---

### POC vs Production

**POC Mode:** Set `KNOWN_ACCOUNT_ID = None` - LLM extracts account ID from denial letter text. Some generic denials may lack identifiable information.

**Production Mode:** Set `KNOWN_ACCOUNT_ID = "12345678"` - Epic workqueue provides account ID directly, enabling 100% coverage.

---

## Quick Start (Databricks)

1. **Install dependencies** (run Cell 1 alone, then restart):
   ```python
   %pip install azure-ai-documentintelligence==1.0.2 openai python-docx
   dbutils.library.restartPython()
   ```

2. **One-time setup** - Run `featurization.py` with flags enabled:
   ```python
   RUN_GOLD_INGESTION = True   # First run
   RUN_PROPEL_INGESTION = True # First run
   ```

3. **Process a denial** - In `inference.py`, set the PDF path:
   ```python
   DENIAL_PDF_PATH = "/path/to/denial.pdf"
   ```

4. **Review output** in `utils/outputs/output_<timestamp>/`

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Platform | Databricks on Azure |
| LLM | Azure OpenAI GPT-4.1 |
| Embeddings | text-embedding-ada-002 (1536 dims) |
| Document OCR | Azure AI Document Intelligence |
| Storage | Delta Lake tables in Unity Catalog |
| Clinical Data | Epic Clarity |

---

## Team

**Financial Data Science** | Mercy Hospital
