# data/structured_data_ingestion.py
# Structured Data Ingestion - Linear Script
#
# Run this notebook to gather structured data for a single account.
# Will be merged into featurization.py later.
#
# Run on Databricks Runtime 15.4 LTS ML

# =============================================================================
# CELL 1: Configuration
# =============================================================================
import os
from datetime import datetime
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# -----------------------------------------------------------------------------
# INPUT: Set the account ID to process
# -----------------------------------------------------------------------------
HSP_ACCOUNT_ID = "12345678"  # <-- CHANGE THIS

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
trgt_cat = os.environ.get('trgt_cat', 'dev')
if trgt_cat == 'dev':
    spark.sql('USE CATALOG prod;')
else:
    spark.sql(f'USE CATALOG {trgt_cat};')

# Table names
LABS_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_labs"
VITALS_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_vitals"
MEDS_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_meds"
PROCEDURES_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_procedures"
ICD10_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_icd10"
MERGED_TABLE = f"{trgt_cat}.fin_ds.fudgesicle_structured_timeline"

print(f"Account ID: {HSP_ACCOUNT_ID}")
print(f"Catalog: {trgt_cat}")

# =============================================================================
# CELL 2: Create Tables (run once)
# =============================================================================
CREATE_TABLES = False  # Set to True on first run

if CREATE_TABLES:
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {LABS_TABLE} (
            hsp_account_id STRING,
            event_timestamp TIMESTAMP,
            lab_name STRING,
            lab_value STRING,
            lab_units STRING,
            reference_range STRING,
            abnormal_flag STRING,
            ingested_at TIMESTAMP
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {VITALS_TABLE} (
            hsp_account_id STRING,
            event_timestamp TIMESTAMP,
            vital_name STRING,
            vital_value STRING,
            vital_units STRING,
            ingested_at TIMESTAMP
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {MEDS_TABLE} (
            hsp_account_id STRING,
            event_timestamp TIMESTAMP,
            med_name STRING,
            med_dose STRING,
            med_units STRING,
            med_route STRING,
            admin_action STRING,
            ingested_at TIMESTAMP
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {PROCEDURES_TABLE} (
            hsp_account_id STRING,
            event_timestamp TIMESTAMP,
            procedure_name STRING,
            procedure_code STRING,
            ingested_at TIMESTAMP
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {ICD10_TABLE} (
            hsp_account_id STRING,
            event_timestamp TIMESTAMP,
            icd10_code STRING,
            icd10_description STRING,
            diagnosis_type STRING,
            ingested_at TIMESTAMP
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {MERGED_TABLE} (
            hsp_account_id STRING,
            event_timestamp TIMESTAMP,
            event_type STRING,
            event_detail STRING,
            ingested_at TIMESTAMP
        ) USING DELTA
    """)

    print("Tables created")

# =============================================================================
# CELL 3: Query Labs
# =============================================================================
print(f"\n{'='*60}")
print(f"QUERYING LABS FOR {HSP_ACCOUNT_ID}")
print(f"{'='*60}")

# TODO: Replace with actual Clarity query
# Key labs: Lactate, WBC, Platelets, Creatinine, Bilirubin, Procalcitonin,
#           CRP, Blood gases, Cultures, BUN, Glucose, PT/INR, Troponin
labs_query = f"""
    INSERT INTO {LABS_TABLE}
    SELECT
        '{HSP_ACCOUNT_ID}' AS hsp_account_id,
        -- TODO: Replace with actual columns
        result_time AS event_timestamp,
        component_name AS lab_name,
        result_value AS lab_value,
        result_unit AS lab_units,
        reference_range AS reference_range,
        abnormal_flag AS abnormal_flag,
        current_timestamp() AS ingested_at
    FROM
        -- TODO: Replace with actual Clarity table
        -- Example: prod.clarity_cur.order_results ord
        --          JOIN prod.clarity_cur.order_proc op ON ord.order_id = op.order_id
        --          JOIN prod.clarity_cur.hsp_account ha ON op.pat_enc_csn_id = ha.pat_enc_csn_id
        (SELECT
            CAST(NULL AS TIMESTAMP) AS result_time,
            'TODO' AS component_name,
            'TODO' AS result_value,
            'TODO' AS result_unit,
            'TODO' AS reference_range,
            'TODO' AS abnormal_flag
        ) AS placeholder
    WHERE
        1 = 0  -- Remove this line when query is ready
"""

# Uncomment when ready:
# spark.sql(labs_query)
# print(f"Labs written to {LABS_TABLE}")

print("[PLACEHOLDER] Labs query not yet configured")

# =============================================================================
# CELL 4: Query Vitals
# =============================================================================
print(f"\n{'='*60}")
print(f"QUERYING VITALS FOR {HSP_ACCOUNT_ID}")
print(f"{'='*60}")

# TODO: Replace with actual Clarity query
# Key vitals: Temperature, Heart Rate, Respiratory Rate, BP (Systolic/Diastolic),
#             MAP, SpO2, GCS, Urine Output, FiO2
vitals_query = f"""
    INSERT INTO {VITALS_TABLE}
    SELECT
        '{HSP_ACCOUNT_ID}' AS hsp_account_id,
        -- TODO: Replace with actual columns
        recorded_time AS event_timestamp,
        flowsheet_name AS vital_name,
        meas_value AS vital_value,
        meas_unit AS vital_units,
        current_timestamp() AS ingested_at
    FROM
        -- TODO: Replace with actual Clarity table
        -- Example: prod.clarity_cur.ip_flwsht_rec flo
        --          JOIN prod.clarity_cur.ip_flwsht_meas meas ON flo.fsd_id = meas.fsd_id
        (SELECT
            CAST(NULL AS TIMESTAMP) AS recorded_time,
            'TODO' AS flowsheet_name,
            'TODO' AS meas_value,
            'TODO' AS meas_unit
        ) AS placeholder
    WHERE
        1 = 0  -- Remove this line when query is ready
"""

# Uncomment when ready:
# spark.sql(vitals_query)
# print(f"Vitals written to {VITALS_TABLE}")

print("[PLACEHOLDER] Vitals query not yet configured")

# =============================================================================
# CELL 5: Query Medications
# =============================================================================
print(f"\n{'='*60}")
print(f"QUERYING MEDICATIONS FOR {HSP_ACCOUNT_ID}")
print(f"{'='*60}")

# TODO: Replace with actual Clarity query
# Key meds: Antibiotics (all), Vasopressors (norepinephrine, vasopressin, dopamine,
#           dobutamine, epinephrine, phenylephrine), IV Fluids (NS, LR)
meds_query = f"""
    INSERT INTO {MEDS_TABLE}
    SELECT
        '{HSP_ACCOUNT_ID}' AS hsp_account_id,
        -- TODO: Replace with actual columns
        admin_time AS event_timestamp,
        medication_name AS med_name,
        dose AS med_dose,
        dose_unit AS med_units,
        route AS med_route,
        admin_action AS admin_action,
        current_timestamp() AS ingested_at
    FROM
        -- TODO: Replace with actual Clarity table
        -- Example: prod.clarity_cur.mar_admin_info mar
        --          JOIN prod.clarity_cur.order_med om ON mar.order_med_id = om.order_med_id
        (SELECT
            CAST(NULL AS TIMESTAMP) AS admin_time,
            'TODO' AS medication_name,
            'TODO' AS dose,
            'TODO' AS dose_unit,
            'TODO' AS route,
            'TODO' AS admin_action
        ) AS placeholder
    WHERE
        1 = 0  -- Remove this line when query is ready
"""

# Uncomment when ready:
# spark.sql(meds_query)
# print(f"Medications written to {MEDS_TABLE}")

print("[PLACEHOLDER] Medications query not yet configured")

# =============================================================================
# CELL 6: Query Procedures
# =============================================================================
print(f"\n{'='*60}")
print(f"QUERYING PROCEDURES FOR {HSP_ACCOUNT_ID}")
print(f"{'='*60}")

# TODO: Replace with actual Clarity query
# Key procedures: Central line, Arterial line, Intubation, Mechanical ventilation,
#                 Dialysis/CRRT, Blood culture collection, Source control procedures
procedures_query = f"""
    INSERT INTO {PROCEDURES_TABLE}
    SELECT
        '{HSP_ACCOUNT_ID}' AS hsp_account_id,
        -- TODO: Replace with actual columns
        proc_time AS event_timestamp,
        proc_name AS procedure_name,
        proc_code AS procedure_code,
        current_timestamp() AS ingested_at
    FROM
        -- TODO: Replace with actual Clarity table
        -- Example: prod.clarity_cur.or_log
        (SELECT
            CAST(NULL AS TIMESTAMP) AS proc_time,
            'TODO' AS proc_name,
            'TODO' AS proc_code
        ) AS placeholder
    WHERE
        1 = 0  -- Remove this line when query is ready
"""

# Uncomment when ready:
# spark.sql(procedures_query)
# print(f"Procedures written to {PROCEDURES_TABLE}")

print("[PLACEHOLDER] Procedures query not yet configured")

# =============================================================================
# CELL 7: Query ICD-10 Codes
# =============================================================================
print(f"\n{'='*60}")
print(f"QUERYING ICD-10 CODES FOR {HSP_ACCOUNT_ID}")
print(f"{'='*60}")

# TODO: Replace with actual Clarity query
# Key codes: A41.x (Sepsis), R65.20 (Severe sepsis), R65.21 (Septic shock),
#            A40.x (Streptococcal sepsis), infection source codes
icd10_query = f"""
    INSERT INTO {ICD10_TABLE}
    SELECT
        '{HSP_ACCOUNT_ID}' AS hsp_account_id,
        -- TODO: Replace with actual columns
        dx_time AS event_timestamp,
        icd10_code AS icd10_code,
        dx_name AS icd10_description,
        dx_type AS diagnosis_type,
        current_timestamp() AS ingested_at
    FROM
        -- TODO: Replace with actual Clarity table
        -- Example: prod.clarity_cur.hsp_acct_dx_list
        (SELECT
            CAST(NULL AS TIMESTAMP) AS dx_time,
            'TODO' AS icd10_code,
            'TODO' AS dx_name,
            'TODO' AS dx_type
        ) AS placeholder
    WHERE
        1 = 0  -- Remove this line when query is ready
"""

# Uncomment when ready:
# spark.sql(icd10_query)
# print(f"ICD-10 codes written to {ICD10_TABLE}")

print("[PLACEHOLDER] ICD-10 query not yet configured")

# =============================================================================
# CELL 8: Join All Into Merged Timeline
# =============================================================================
print(f"\n{'='*60}")
print(f"MERGING ALL DATA FOR {HSP_ACCOUNT_ID}")
print(f"{'='*60}")

merge_query = f"""
    INSERT OVERWRITE {MERGED_TABLE}

    SELECT hsp_account_id, event_timestamp, 'LAB' AS event_type,
           CONCAT(lab_name, ': ', lab_value,
                  COALESCE(CONCAT(' ', lab_units), ''),
                  COALESCE(CONCAT(' [', abnormal_flag, ']'), '')) AS event_detail,
           ingested_at
    FROM {LABS_TABLE}
    WHERE hsp_account_id = '{HSP_ACCOUNT_ID}'

    UNION ALL

    SELECT hsp_account_id, event_timestamp, 'VITAL' AS event_type,
           CONCAT(vital_name, ': ', vital_value,
                  COALESCE(CONCAT(' ', vital_units), '')) AS event_detail,
           ingested_at
    FROM {VITALS_TABLE}
    WHERE hsp_account_id = '{HSP_ACCOUNT_ID}'

    UNION ALL

    SELECT hsp_account_id, event_timestamp, 'MED' AS event_type,
           CONCAT(med_name, ' ', med_dose,
                  COALESCE(CONCAT(' ', med_units), ''),
                  COALESCE(CONCAT(' ', med_route), ''),
                  COALESCE(CONCAT(' [', admin_action, ']'), '')) AS event_detail,
           ingested_at
    FROM {MEDS_TABLE}
    WHERE hsp_account_id = '{HSP_ACCOUNT_ID}'

    UNION ALL

    SELECT hsp_account_id, event_timestamp, 'PROCEDURE' AS event_type,
           CONCAT(procedure_name,
                  COALESCE(CONCAT(' (', procedure_code, ')'), '')) AS event_detail,
           ingested_at
    FROM {PROCEDURES_TABLE}
    WHERE hsp_account_id = '{HSP_ACCOUNT_ID}'

    UNION ALL

    SELECT hsp_account_id, event_timestamp, 'DIAGNOSIS' AS event_type,
           CONCAT(icd10_code, ': ', icd10_description,
                  COALESCE(CONCAT(' [', diagnosis_type, ']'), '')) AS event_detail,
           ingested_at
    FROM {ICD10_TABLE}
    WHERE hsp_account_id = '{HSP_ACCOUNT_ID}'

    ORDER BY event_timestamp
"""

# Uncomment when intermediate tables have data:
# spark.sql(merge_query)
# print(f"Merged timeline written to {MERGED_TABLE}")

print("[PLACEHOLDER] Merge query ready - run after intermediate tables have data")

# =============================================================================
# CELL 9: View Results
# =============================================================================
print(f"\n{'='*60}")
print(f"RESULTS FOR {HSP_ACCOUNT_ID}")
print(f"{'='*60}")

# Uncomment to view results:
# display(spark.sql(f"SELECT * FROM {MERGED_TABLE} WHERE hsp_account_id = '{HSP_ACCOUNT_ID}' ORDER BY event_timestamp"))

print("[PLACEHOLDER] Uncomment display() after running queries")
print("\nDone.")
