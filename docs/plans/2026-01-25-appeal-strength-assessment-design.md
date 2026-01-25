# Appeal Strength Assessment Design

**Date:** 2026-01-25
**Status:** Approved for implementation

---

## Overview

Add a strength assessment feature that evaluates generated appeal letters against Propel sepsis criteria, argument structure, and evidence quality. The assessment appears as a separate section in the DOCX output before the letter body, providing CDI reviewers with a quick signal about letter quality.

---

## Output Format

The assessment appears after the header metadata and before "Dear...":

```
Generated: 2026-01-15 21:12
Patient: John Doe
Payor: United Healthcare
DRG: 871 -> 872
Gold Letter: WON-Sepsis-UHC-2024-001

═══════════════════════════════════════════════════════
APPEAL STRENGTH ASSESSMENT (Internal Review Only)
═══════════════════════════════════════════════════════

Overall Strength: 7/10 - MODERATE

Summary: Strong infection documentation with clear antibiotic
timeline. Organ dysfunction evidence present but could cite
specific lactate values from Day 2 notes.

Detailed Breakdown:
┌─────────────────────────────────────────────────────────┐
│ PROPEL CRITERIA COVERAGE (from: Propel definitions)     │
│ Score: 8/10                                             │
│ ✓ Infection source identified (pneumonia)               │
│ ✓ SIRS criteria documented (temp, WBC, tachycardia)     │
│ ✓ Organ dysfunction present (respiratory, renal)        │
│ △ Could strengthen: Specific SOFA score calculation     │
├─────────────────────────────────────────────────────────┤
│ ARGUMENT STRUCTURE (from: denial letter, gold template) │
│ Score: 7/10                                             │
│ ✓ Directly addresses denial reason                      │
│ ✓ Follows proven gold letter structure                  │
│ △ Could strengthen: Explicit rebuttal of payor's claim  │
├─────────────────────────────────────────────────────────┤
│ EVIDENCE QUALITY                                        │
│                                                         │
│ From Clinical Notes (from: Epic Clarity notes): 6/10    │
│ ✓ Timestamps present for key events                     │
│ ✓ Antibiotic timing documented                          │
│ ✗ Missing: Lactate 4.2 from 03/15 08:00 notes          │
│ ✗ Missing: MAP 63 requiring vasopressors               │
│                                                         │
│ From Structured Data: (not yet available)               │
└─────────────────────────────────────────────────────────┘
═══════════════════════════════════════════════════════

Dear Medical Director,
...
```

All three formats (simple score, summary, detailed breakdown) are included for POC feedback gathering.

---

## Pipeline Integration

The assessment runs as **Step 6.5** in inference.py:

```
Step 1: PDF Parse (Document Intelligence)
Step 2: Vector Search for best gold letter
Step 3: Extract denial info (LLM)
Step 4: Query Clarity for clinical notes
Step 5: Extract clinical data from long notes (LLM)
Step 6: Generate appeal letter (LLM)
Step 6.5: ──► ASSESS APPEAL STRENGTH (LLM) ◄── NEW
Step 7: Export to DOCX (now includes assessment section)
```

### Data Flow In

| Input | Source | Purpose |
|-------|--------|---------|
| `generated_letter` | Step 6 output | The letter being evaluated |
| `propel_definition` | `fudgesicle_propel_data` table | Official sepsis criteria to check against |
| `denial_text` | Step 1 output | What payor claimed, so we can check if it's addressed |
| `extracted_clinical_data` | Step 5 output | Full clinical evidence available |
| `gold_letter_used` | Step 2 output | Template structure for argument comparison |

### Data Flow Out

| Output | Type | Used By |
|--------|------|---------|
| `overall_score` | int (1-10) | Header display |
| `overall_rating` | string (LOW/MODERATE/HIGH) | Header display |
| `summary` | string | Brief rationale |
| `propel_breakdown` | dict | Detailed section |
| `argument_breakdown` | dict | Detailed section |
| `evidence_breakdown` | dict | Detailed section with missed evidence |

---

## Assessment Prompt Structure

```
You are evaluating the strength of a sepsis DRG appeal letter.

═══ PROPEL SEPSIS CRITERIA (source: official definitions) ═══
{propel_definition_summary}

═══ DENIAL LETTER (source: payor's denial) ═══
{denial_text}

═══ GOLD LETTER TEMPLATE USED (source: past winning appeal) ═══
{gold_letter_appeal_text}

═══ AVAILABLE CLINICAL EVIDENCE ═══

── From Clinical Notes (source: Epic Clarity notes) ──
{extracted_clinical_data}

── From Structured Data (source: Epic Clarity tables) ──
{structured_data or "Not yet available - structured data integration pending"}

═══ GENERATED APPEAL LETTER (being evaluated) ═══
{generated_letter}

═══ EVALUATION INSTRUCTIONS ═══
Evaluate this appeal letter and provide:

1. OVERALL SCORE (1-10) and RATING (LOW/MODERATE/HIGH)

2. SUMMARY (2-3 sentences explaining the score)

3. DETAILED BREAKDOWN with scores and specific findings:

   a) PROPEL CRITERIA COVERAGE - Does the letter document:
      - Suspected or confirmed infection
      - Organ dysfunction (per SOFA criteria)
      - Clinical response to treatment
      Note what's present (✓), what could be stronger (△), what's missing (✗)

   b) ARGUMENT STRUCTURE - Does the letter:
      - Directly address the payor's stated denial reason
      - Follow the logical structure of the gold letter template
      - Provide clear clinical reasoning

   c) EVIDENCE QUALITY - Split into two parts:

      From Clinical Notes:
      - Are specific values cited (not just "elevated lactate")?
      - Are timestamps present for key events?
      - List any relevant evidence in the notes NOT cited in the letter

      From Structured Data:
      - [Evaluate when available, otherwise note "pending"]
      - Labs, vitals, medications with timestamps
      - List any relevant structured evidence NOT cited in the letter

Return as JSON: {output_schema}
```

---

## LLM Output Schema

```json
{
  "overall_score": 7,
  "overall_rating": "MODERATE",
  "summary": "Strong infection documentation with clear antibiotic timeline. Organ dysfunction evidence present but could cite specific lactate values from Day 2 notes.",

  "propel_criteria": {
    "score": 8,
    "source": "Propel definitions",
    "findings": [
      {"status": "present", "item": "Infection source identified (pneumonia)"},
      {"status": "present", "item": "SIRS criteria documented (temp, WBC, tachycardia)"},
      {"status": "present", "item": "Organ dysfunction present (respiratory, renal)"},
      {"status": "could_strengthen", "item": "Specific SOFA score calculation"}
    ]
  },

  "argument_structure": {
    "score": 7,
    "source": "denial letter, gold template",
    "findings": [
      {"status": "present", "item": "Directly addresses denial reason"},
      {"status": "present", "item": "Follows proven gold letter structure"},
      {"status": "could_strengthen", "item": "Explicit rebuttal of payor's claim"}
    ]
  },

  "evidence_quality": {
    "clinical_notes": {
      "score": 6,
      "source": "Epic Clarity notes",
      "findings": [
        {"status": "present", "item": "Timestamps present for key events"},
        {"status": "present", "item": "Antibiotic timing documented"},
        {"status": "missing", "item": "Lactate 4.2 from 03/15 08:00 notes"},
        {"status": "missing", "item": "MAP 63 requiring vasopressors"}
      ]
    },
    "structured_data": {
      "score": null,
      "source": "Epic Clarity tables",
      "findings": [],
      "status": "pending_integration"
    }
  }
}
```

**Status values:**
- `present` → ✓ in output
- `could_strengthen` → △ in output
- `missing` → ✗ in output

**Rating thresholds:**
- 1-4: LOW
- 5-7: MODERATE
- 8-10: HIGH

---

## Error Handling

### LLM Call Failures

| Scenario | Handling |
|----------|----------|
| Assessment LLM call times out | Proceed with letter export, add "Assessment unavailable - LLM timeout" |
| LLM returns malformed JSON | Retry once; if still fails, show "Assessment unavailable - parse error" |
| LLM returns scores outside 1-10 | Clamp to valid range, log warning |

### Data Availability Edge Cases

| Scenario | Handling |
|----------|----------|
| No Propel definition found | Skip Propel section, note "Propel criteria not available for this condition" |
| No gold letter matched | Note "Evaluated against default template (no gold letter match)" |
| Clinical notes extraction failed | Show "No extracted notes available" with score N/A |
| Structured data not integrated | Show "pending_integration" status |

### Fallback Display

If assessment fails entirely:

```
═══════════════════════════════════════════════════════
APPEAL STRENGTH ASSESSMENT (Internal Review Only)
═══════════════════════════════════════════════════════

Assessment unavailable: {error_reason}

Please review letter manually before sending.
═══════════════════════════════════════════════════════
```

The assessment never blocks letter generation.

---

## Cost & Performance

**Assessment call tokens:**
- Input: ~12,000 tokens
- Output: ~800 tokens
- Cost: ~$0.03 per assessment

**Updated per-letter cost:** ~$0.23 (was ~$0.20)

**Latency:** Adds ~3-5 seconds (not a concern given workflow)

**ROI context:** One successful appeal brings $5-30k; assessment cost is negligible.

---

## Open Items & To-Do List

### Data to Add (pending Clarity integration)

| Item | Source | Status |
|------|--------|--------|
| Labs with timestamps | Epic Clarity | To do |
| Medications with administration times | Epic Clarity | To do |
| Vitals with measurement times | Epic Clarity | To do |
| Procedures with timestamps | Epic Clarity | To do |
| ICD-10 codes (with diagnosis timing if available) | Epic Clarity | To do |
| Inpatient medication orders | Epic Clarity | To do |

### Information Needed from Writing Team

| Question | Why It Matters | Status |
|----------|----------------|--------|
| Which of the ~75 note types are most valuable for sepsis appeals? | Currently pulling 14 - may be missing key evidence or pulling noise | Pending |
| Are there specific phrases/arguments that win appeals? | Could weight these higher in argument structure scoring | Pending |
| What makes a "weak" vs "strong" appeal in their experience? | Calibrate assessment scoring to match expert judgment | Pending |

### Decisions to Make

| Decision | Options | Status |
|----------|---------|--------|
| SOFA score calculation | LLM estimates vs. explicit calculation from structured data | Pending structured data |
| Assessment threshold for flagging | What score triggers "needs extra review"? | Gather feedback first |
| Outcome tracking | How to capture won/lost status for correlation analysis | Pending |

---

## Future Enhancements

| When | Enhancement |
|------|-------------|
| Structured data added | Enable `evidence_quality.structured_data` scoring |
| Enough letters processed | Correlate scores with actual appeal outcomes (won/lost) |
| Patterns emerge | Add payor-specific scoring adjustments |
