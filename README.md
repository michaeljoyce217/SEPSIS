# Sepsis Rebuttal Engine

**Multi-Agent AI for DRG Appeal Letter Automation**

Automated generation of DRG appeal letters for sepsis-related insurance denials. Built by the Financial Data Science team at Mercy Hospital.

---

## Overview

When insurance payors deny or downgrade sepsis DRG claims (870/871/872), this system generates professional rebuttal letters by:

1. **Parsing denial letters** - LLM extracts payor, DRG codes, account ID, and determines if sepsis-related
2. **Vector search** - Finds the most similar past denial from our gold standard library using embeddings
3. **Learning from winners** - Uses the matched winning rebuttal as a template/guide
4. **Applying clinical criteria** - Includes official Propel sepsis definitions
5. **Generating rebuttals** - Creates patient-specific appeal letters using clinical notes from Clarity

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KNOWLEDGE LAYER                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │    Propel    │    │    Gold      │    │   Clinical   │                  │
│  │   Criteria   │    │   Letters    │    │    Notes     │                  │
│  │  (propel_    │    │  (gold_      │    │   (Clarity)  │                  │
│  │   data)      │    │   letters)   │    │              │                  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                  │
│         │                   │                   │                           │
│         └─────────┬─────────┴─────────┬─────────┘                           │
│                   ▼                   ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        PIPELINE                                     │   │
│  │                                                                     │   │
│  │  [Denial PDF] ──► FEATURIZATION ──► [Inference Table]              │   │
│  │                   • Extract text (Document Intelligence)            │   │
│  │                   • Parse info (LLM: account ID, payor, DRGs)      │   │
│  │                   • Generate embedding                              │   │
│  │                   • Join with Clarity clinical notes                │   │
│  │                                                                     │   │
│  │  [Inference Table] ──► INFERENCE ──► [DOCX Letter]                 │   │
│  │                        • Filter sepsis cases                        │   │
│  │                        • Vector search gold letters                 │   │
│  │                        • Include Propel definition                  │   │
│  │                        • Generate rebuttal (LLM)                    │   │
│  │                        • Export to DOCX for review                  │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  [DOCX Letter] ──► CDI Review ──► Approved Letter ──► Send to Payor        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Vector Search** | Embeddings-based similarity matching finds the most relevant past denial |
| **Gold Letter Learning** | Uses winning rebuttals as templates - proven arguments get reused |
| **Propel Integration** | Official clinical criteria injected into prompts |
| **Human-in-the-Loop** | All letters output as DOCX for CDI review before sending |
| **Scope Filtering** | Built for expansion - currently filtered to sepsis (870/871/872) |
| **No Vendor Lock-in** | Runs entirely on existing Databricks/Azure infrastructure |

## Repository Structure

```
SEPSIS/
├── data/
│   └── featurization.py      # Data ingestion and preparation
├── model/
│   └── inference.py          # Letter generation
├── utils/
│   ├── gold_standard_rebuttals/  # Past winning appeal letters (PDFs)
│   ├── Sample_Denial_Letters/    # New denial letters to process
│   └── propel_data/              # Clinical criteria definitions (DOCX)
├── docs/
│   └── rebuttal-assistant-guide.html  # Executive overview
├── output/                   # Generated DOCX letters
├── compare_denials.py        # Utility: check for duplicate denials
├── test_queries.sql          # Validation queries for Unity Catalog
└── README.md
```

## Unity Catalog Tables

| Table | Purpose |
|-------|---------|
| `dev.fin_ds.fudgesicle_gold_letters` | Past winning rebuttals with denial embeddings |
| `dev.fin_ds.fudgesicle_propel_data` | Official clinical criteria (sepsis definition) |
| `dev.fin_ds.fudgesicle_inference` | Denial cases ready for processing |
| `dev.fin_ds.fudgesicle_inference_score` | Generated rebuttal letters |

## Quick Start (Databricks)

### 1. Initial Setup

Copy files to Databricks notebooks and set the paths:
```python
GOLD_LETTERS_PATH = "/Workspace/Repos/your-user/fudgesicle/utils/gold_standard_rebuttals"
DENIAL_LETTERS_PATH = "/Workspace/Repos/your-user/fudgesicle/utils/Sample_Denial_Letters"
PROPEL_DATA_PATH = "/Workspace/Repos/your-user/fudgesicle/utils/propel_data"
```

### 2. Ingest Gold Standard Letters (one-time)

In `featurization.py`:
```python
RUN_GOLD_INGESTION = True
```
Run the notebook. This extracts rebuttals and denials from gold letter PDFs, generates embeddings, and stores in `fudgesicle_gold_letters`.

### 3. Ingest Propel Definitions (one-time)

In `featurization.py`:
```python
RUN_PROPEL_INGESTION = True
```
Run the notebook. This reads DOCX files from `propel_data/` and stores in `fudgesicle_propel_data`.

### 4. Process New Denial Letters

In `featurization.py`:
```python
RUN_DENIAL_PROCESSING = True
RUN_DENIAL_FEATURIZATION = True
WRITE_TO_TABLE = True
```
Run the notebook. This:
- Extracts text from denial PDFs
- Uses LLM to extract account ID, payor, DRGs, sepsis flag
- Generates embeddings
- Joins with Clarity clinical notes
- Writes to `fudgesicle_inference`

### 5. Generate Rebuttal Letters

In `inference.py`:
```python
WRITE_TO_TABLE = True   # Optional: persist to score table
EXPORT_TO_DOCX = True   # Generate Word documents
```
Run the notebook. For each sepsis case:
- Finds best matching gold letter via vector search
- Includes Propel sepsis definition
- Generates rebuttal using clinical notes
- Exports DOCX to `output/` folder

## Configuration Flags

### featurization.py

| Flag | Default | Description |
|------|---------|-------------|
| `RUN_GOLD_INGESTION` | False | Process gold standard letter PDFs |
| `RUN_PROPEL_INGESTION` | False | Process Propel definition DOCX files |
| `RUN_DENIAL_PROCESSING` | False | Process new denial letter PDFs |
| `RUN_DENIAL_FEATURIZATION` | False | Join with Clarity and write to table |
| `WRITE_TO_TABLE` | False | Write results to Unity Catalog |

### inference.py

| Flag | Default | Description |
|------|---------|-------------|
| `SCOPE_FILTER` | "sepsis" | Which denial types to process |
| `MATCH_SCORE_THRESHOLD` | 0.7 | Minimum similarity to use gold letter |
| `WRITE_TO_TABLE` | False | Persist generated letters to score table |
| `EXPORT_TO_DOCX` | True | Export letters as Word documents |

## Technology Stack

| Component | Technology |
|-----------|------------|
| Platform | Databricks on Azure |
| LLM | Azure OpenAI GPT-4.1 |
| Embeddings | Azure OpenAI text-embedding-ada-002 (1536 dims) |
| Document OCR | Azure AI Document Intelligence |
| Storage | Delta Lake tables in Unity Catalog |
| Clinical Data | Epic Clarity |
| Runtime | Databricks Runtime 15.4 LTS ML |

## Validation

Run queries from `test_queries.sql` to verify data:

```sql
-- Check inference table has all columns populated
SELECT
    COUNT(*) as total_rows,
    COUNT(denial_letter_text) as has_denial_text,
    COUNT(denial_embedding) as has_embedding,
    SUM(CASE WHEN is_sepsis THEN 1 ELSE 0 END) as sepsis_count
FROM dev.fin_ds.fudgesicle_inference;
```

## Extending to Other Conditions

The architecture supports any denial type. To add a new condition (e.g., pneumonia):

1. **Add clinical criteria**: Place `pneumonia.docx` in `utils/propel_data/`
2. **Add gold letters**: Add winning pneumonia rebuttals to `gold_standard_rebuttals/`
3. **Update scope filter**: Modify `SCOPE_FILTER` logic in inference.py
4. **Run ingestion**: Re-run with ingestion flags enabled

No architectural changes needed - the same pipeline handles any denial type.

## Team

**Financial Data Science** | Mercy Hospital

---

*Built with Azure OpenAI, Databricks, and Epic Clarity integration.*
