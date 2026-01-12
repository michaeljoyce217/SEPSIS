# Clinical Risk Prediction Pipeline for Imbalanced Healthcare Data

A production-grade machine learning pipeline for colorectal cancer (CRC) risk prediction, demonstrating rigorous methodology for healthcare ML with extreme class imbalance (~250:1).

## The Challenge

Predicting rare medical events from electronic health records presents unique challenges:

- **Extreme class imbalance**: 0.4% positive rate (1 case per 250 observations)
- **EHR data complexity**: Multiple data sources, workflow artifacts, selective ordering patterns
- **Temporal dependencies**: Patient observations over time with variable lookback windows
- **Label uncertainty**: Negative labels require validation through follow-up
- **Clinical interpretability**: Model must surface actionable, clinically meaningful signals

This pipeline addresses each challenge with principled solutions validated against clinical domain knowledge.

## Pipeline Architecture

```
Book 0: Cohort Creation          Book 1-7: Domain Feature Engineering
         │                                    │
         ▼                                    ▼
┌─────────────────────┐         ┌──────────────────────────────────┐
│ Patient-Month       │         │ Vitals      │ ICD-10  │ Labs    │
│ Observations        │────────▶│ Social      │ Meds    │ Visits  │
│ + Tiered Labels     │         │ Procedures  │         │         │
│ + Train/Val/Test    │         └──────────────────────────────────┘
└─────────────────────┘                       │
                                              ▼
                              ┌─────────────────────────────┐
                              │ Book 8: Compilation         │
                              │ Anti-memorization transforms│
                              │ Quality validation          │
                              └─────────────────────────────┘
                                              │
                                              ▼
                              ┌─────────────────────────────┐
                              │ Book 9: Feature Selection   │
                              │ Hybrid clustering + SHAP    │
                              │ CV stability analysis       │
                              └─────────────────────────────┘
```

## Key Methodological Innovations

### 1. Tiered Label Quality System

Not all negative labels are equally reliable. Our three-tier approach maximizes training data while maintaining label integrity:

| Tier | Criteria | Confidence | Coverage |
|------|----------|------------|----------|
| **Tier 1** | Return visit 7-12 months | High | ~47% |
| **Tier 2** | Return visit 4-6 months + PCP | Medium | ~23% |
| **Tier 3** | No return but has PCP | Assumed | ~30% |

### 2. Patient-Level Stratification

StratifiedGroupKFold ensures no patient appears in multiple splits, preventing data leakage while maintaining class balance:

- **Groups**: Patient IDs (all observations from one patient stay together)
- **Stratification**: Patient-level outcome (positive if any observation is positive)
- **Temporal holdout**: Q6 test set provides true out-of-time validation

### 3. Multi-Source Feature Engineering

**2.7 billion** raw lab records transformed into 93 sophisticated features:

| Feature Type | Example | Clinical Rationale |
|--------------|---------|-------------------|
| **Trajectory** | Hemoglobin velocity | Rate of decline indicates bleeding |
| **Acceleration** | Second derivative | Captures disease progression speed |
| **Composite** | Anemia severity score | Combines weak signals into strong predictors |

### 4. EHR Workflow Artifact Detection

Real-world EHR data contains systematic biases from clinical workflows:

- **Epic default problem**: 78% marked "never smoker" includes unanswered defaults
- **Selective ordering**: Missing CEA indicates low cancer suspicion (informative missingness)
- **Solution**: Preserve missingness as signal rather than impute; document artifacts

### 5. Anti-Memorization Transformations

Prevent models from memorizing patient-specific patterns:

- Remove temporal identifiers (months since cohort entry)
- Bin continuous values (age groups, BMI categories)
- Transform patient characteristics to ordinal scales

### 6. Hybrid Feature Selection with Validation Gates

Two-phase approach balances dimensionality reduction with performance preservation:

**Phase 1: Cluster-Based Reduction**
- Spearman correlation distance matrix
- Dynamic threshold via silhouette score optimization
- Clinical must-keep features preserved regardless of ranking

**Phase 2: Iterative SHAP Winnowing**
- 2:1 positive case weighting for importance scores
- Automatic stop conditions (validation AUPRC drop, train-val gap)
- Test set evaluated only at final model (no peeking)

**Phase 3: Cross-Validation Stability**
- 3-fold CV confirms feature robustness
- Features must appear in ≥67% of folds
- Identifies overfitting artifacts

## Validated Risk Signals

The pipeline surfaces clinically meaningful predictors:

| Feature | Risk Ratio | Clinical Interpretation |
|---------|------------|------------------------|
| Hemoglobin acceleration | 10.9x | Rapid decline indicates occult bleeding |
| GI bleeding symptoms | 6.3x | Cardinal sign despite low prevalence |
| Hemorrhoid medications (recent) | 16.8x | May mask colorectal bleeding source |
| Iron supplementation | 4.9x | Treatment for chronic blood loss |
| Rapid weight loss (>5% in 60d) | 3.6x | Strongest vital sign predictor |
| Symptom combinations | 4.6x | Triad patterns more predictive than individual symptoms |

## Technical Stack

- **Platform**: Databricks (PySpark + Python)
- **ML Framework**: XGBoost with native missing value handling
- **Interpretability**: SHAP TreeExplainer
- **Checkpointing**: JSON/Parquet for version stability across environments

## Repository Structure

```
2nd_Dataset_Creation/
├── V2_Book0_Cohort_Creation.py      # Patient-month cohort with tiered labels
├── V2_Book1_Vitals.py               # Weight loss, BMI, blood pressure features
├── V2_Book2_ICD10.py                # Diagnoses, symptoms, comorbidity scores
├── V2_Book3_Social_Factors.py       # Smoking, alcohol (with artifact analysis)
├── V2_Book4_Labs_Combined.py        # Laboratory values with temporal dynamics
├── V2_Book5_1_Medications_Outpatient.py
├── V2_Book5_2_Medications_Inpatient.py
├── V2_Book6_Visit_History.py        # Healthcare utilization patterns
├── V2_Book7_Procedures.py           # Diagnostic and therapeutic procedures
├── V2_Book8_Compilation.py          # Assembly and anti-memorization transforms
└── V2_Book9_Feature_Selection.py    # Hybrid selection with CV stability
```

## Running the Pipeline

Execute notebooks in order (Book 0 → Book 9). Each notebook:

1. Documents clinical motivation and expected findings
2. Implements feature engineering with validation
3. Reduces features while preserving signal
4. Saves outputs to Unity Catalog tables

Feature selection (Book 9) includes checkpoint recovery—interrupt anytime and resume without losing progress.

## Design Principles

1. **Clinical grounding**: Every feature motivated by medical literature
2. **Transparency**: Document artifacts and limitations, don't hide them
3. **Robustness**: Cross-validation stability, not just single-split performance
4. **Reproducibility**: Deterministic splits, seeded randomness, version-stable checkpoints
5. **Test set discipline**: Evaluate once at final model, never during development

## Citation

If you use this methodology, please cite:

```
CRC Risk Prediction Pipeline: A Methodological Framework for
Imbalanced Healthcare ML. 2025.
```

## License

[Specify license]
