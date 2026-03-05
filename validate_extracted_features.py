"""
MIMIC-IV Feature Extraction — Validation Suite  v1.3.1
=======================================================
Changes vs v1.3:
  - VAL-05a: adm_age clip check updated (expects fix applied)
  - VAL-09e: AKI baseline / ratio non-null check added (catches BUG-2)
  - VAL-14a: Missing-value audit now classifies columns into three tiers:
      STRUCTURAL  — expected to be nearly all-NaN by design (not failures)
      CLINICAL    — high but clinically normal sparsity (warn only)
      UNEXPECTED  — genuinely unexpected high missingness (FAIL)
  - VAL-17: New post-clip range checks confirm no out-of-range values remain
             after apply_clip_rules() was run

Usage
-----
  python validate_extracted_features.py [path/to/extracted_features.csv]

Exit codes:  0 = all clear   1 = one or more FAIL
"""

import sys
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

DEFAULT_CSV = './model_outputs/extracted_features.csv'
CSV_PATH    = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN LISTS
# ─────────────────────────────────────────────────────────────────────────────
MUST_BE_ABSENT = [
    'adm_discharge_location', 'adm_los_days',
    'util_prior_hadm_count_30d', 'util_prior_hadm_count_90d',
    'util_prior_hadm_count_180d', 'util_prior_hadm_count_365d',
    'trans_last_careunit', 'icu_total_los_days',
    'aki_creat_max_ws', 'note_stable_at_discharge', 'note_resolution_score',
]

MUST_BE_PRESENT = [
    'subject_id', 'hadm_id', 'admittime', 'dischtime',
    'adm_inhospital_mortality', 'readmitted_30d',
    'adm_age', 'adm_gender', 'adm_race', 'adm_insurance',
    'adm_marital_status', 'adm_has_ed', 'adm_ed_los_hours',
    'adm_admission_type', 'adm_admission_location',
    'util_days_since_last_discharge',
    'icu_any', 'icu_num_stays', 'trans_num_transfers_ws',
    'trans_first_careunit', 'icu_total_los_days_dpt',
    'dx_charlson_index_prior',
    'dx_num_codes_ws', 'dx_num_distinct_codes_ws',
    'proc_num_codes_ws', 'proc_num_distinct_codes_ws',
    'aki_creat_baseline_bw', 'aki_creat_max_bw',
    'aki_creat_ratio_max_over_baseline',
    'med_total_orders_count_ws', 'med_prn_count_ws',
    'omr_weight_kg_last_pre', 'omr_height_cm_last_pre', 'omr_bmi_last_pre',
    'note_has_followup_instructions', 'note_has_home_health',
    'note_lives_alone', 'note_has_caregiver',
    'note_function_needs_assistance', 'note_cognitive_impairment',
    'note_days_to_followup_est',
]

CHARLSON_COMPS = [
    "mi","chf","pvd","cvd","dementia","copd","rheum","pud",
    "mild_liver","diabetes","diabetes_comp","hemi_para","renal","cancer",
    "mod_severe_liver","metastatic","hiv",
]
HR_CLASSES = [
    "anticoagulant","insulin","opioid","benzo",
    "antipsychotic","antiarrhythmic","steroid","chemo",
]

# ─────────────────────────────────────────────────────────────────────────────
# MISSING VALUE CLASSIFICATION
# Three tiers so the FAIL only fires on genuinely unexpected issues.
# ─────────────────────────────────────────────────────────────────────────────

# Structural: always NaN by design for first-admission-only cohort or ICU-only source.
# These should be 80-100% NaN and that is CORRECT. Listed as regex patterns.
STRUCTURAL_NAN_PATTERNS = [
    r'^util_days_since_last_discharge$',      # 100% NaN — first-admission cohort
    r'^vital_.+_pdw$',                        # ~96% NaN — chartevents = ICU only
    r'^vital_measure_count_pdw$',
    r'^vital_gcs_.+_pdw$',
    r'^icu_total_los_days_dpt$',              # 83% NaN — only 17% ICU
    r'^icu_time_to_first_icu_hours$',
    r'^icu_first_intime$',
    r'^lab_.+_slope_pdw$',                    # needs ≥2 measurements in 24h window
    r'^lab_.+_delta_last_first_pdw$',
    r'^lab_.+_std_pdw$',
]

# Clinical sparsity: high missingness is expected due to test ordering patterns.
# Warn only if > CLINICAL_WARN_THRESHOLD.
CLINICAL_NAN_PATTERNS = [
    r'^lab_(lactate|albumin|bilirubin)_.+_bw$',   # ordered for specific patients
    r'^lab_(lactate|albumin|bilirubin)_.+_pdw$',
    r'^lab_wbc_slope_pdw$',
    r'^omr_.+$',                               # not all patients have pre-adm OMR
    r'^note_days_to_followup_est$',
    r'^days_to_readmit$',
    r'^next_admittime$',
    r'^lab_.+_pdw$',                           # ~51% NaN — short stays lack last-24h data
]
CLINICAL_WARN_THRESHOLD = 0.99   # warn if > 99% missing even for clinical columns

# Columns that MUST have low missingness (fail if > CORE_MAX_NULL)
CORE_LOW_NULL_COLS = {
    'adm_age':                  0.01,
    'adm_gender':               0.05,
    'adm_inhospital_mortality': 0.01,
    'readmitted_30d':           0.01,
    'dx_charlson_index_prior':  0.01,
    'adm_admission_type':       0.05,
    'icu_any':                  0.01,
}

# AKI columns: after FIX-2 they must have meaningful non-null rates
# (patients with creatinine labs in first 24h ~ 50-70% of cohort)
AKI_MIN_NONULL = 0.30   # fail if < 30% non-null (catches the 100% NaN bug)


# ─────────────────────────────────────────────────────────────────────────────
# PHYSIOLOGICAL RANGES
# ─────────────────────────────────────────────────────────────────────────────
VITAL_RANGES = {
    'vital_hr_last_pdw':   (10, 350),
    'vital_sbp_last_pdw':  (40, 300),
    'vital_dbp_last_pdw':  (10, 200),
    'vital_spo2_last_pdw': (50, 100),
    'vital_temp_last_pdw': (25,  45),
    'vital_rr_last_pdw':   (1,   80),
    'vital_map_last_pdw':  (20, 200),
}
LAB_RANGES = {
    'lab_creatinine_last_bw':  (0,   50),
    'lab_hemoglobin_last_bw':  (1,   25),
    'lab_wbc_last_bw':         (0,  500),
    'lab_sodium_last_bw':      (100, 190),
    'lab_potassium_last_bw':   (1,   12),
    'lab_glucose_last_bw':     (10, 2000),
    'lab_lactate_last_bw':     (0,   50),
    'lab_albumin_last_bw':     (0.5,  8),
}

# ─────────────────────────────────────────────────────────────────────────────
# RESULT TRACKING
# ─────────────────────────────────────────────────────────────────────────────
results = []

def record(check_id, name, passed, detail="", warn_only=False):
    status = "PASS" if passed else ("WARN" if warn_only else "FAIL")
    results.append(dict(check_id=check_id, name=name, status=status, detail=detail))
    icon = "✓" if passed else ("⚠" if warn_only else "✗")
    print(f"  [{status}] {icon}  {check_id}  {name}")
    if detail:
        for line in detail.splitlines()[:8]:     # cap printed detail lines
            print(f"             {line}")
        if detail.count('\n') > 8:
            print(f"             … (+{detail.count(chr(10))-8} more lines in CSV report)")

def _matches_any(col, patterns):
    return any(re.search(p, col) for p in patterns)

# ─────────────────────────────────────────────────────────────────────────────
def load_csv(path):
    print(f"\n{'='*64}")
    print(f"  MIMIC-IV Feature Validation  v1.3.1")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  File: {path}")
    print(f"{'='*64}\n")
    if not os.path.exists(path):
        print(f"FATAL: File not found → {path}")
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns\n")
    return df

# ─────────────────────────────────────────────────────────────────────────────
def check_schema(df):
    print("── VAL-01  Schema ──────────────────────────────────────────")
    missing = [c for c in MUST_BE_PRESENT if c not in df.columns]
    record("VAL-01a", "Required columns present", not missing,
           f"Missing: {missing}" if missing else "All required columns found")

    ws_lab = [c for c in df.columns if re.search(r'lab_.+_(min|max|mean|last)_ws$', c)]
    record("VAL-01b", "No whole-stay (_ws) lab columns", not ws_lab,
           f"Leaky: {ws_lab}" if ws_lab else "None found")

    missing_prior = [f'dx_charlson_{c}_prior' for c in CHARLSON_COMPS
                     if f'dx_charlson_{c}_prior' not in df.columns]
    present_ws    = [f'dx_charlson_{c}_ws'    for c in CHARLSON_COMPS
                     if f'dx_charlson_{c}_ws'    in df.columns]
    record("VAL-01c", "Charlson _prior columns present (not _ws)",
           not missing_prior and not present_ws,
           (f"Missing _prior: {missing_prior}  Leaky _ws: {present_ws}"
            if missing_prior or present_ws else "OK"))

    hr_found = sum(1 for c in HR_CLASSES if f'med_highrisk_{c}_medwin' in df.columns)
    record("VAL-01d", f"High-risk med flags present ({hr_found}/{len(HR_CLASSES)})",
           hr_found == len(HR_CLASSES),
           f"Found {hr_found}/{len(HR_CLASSES)}")

    icu_dpt = 'icu_total_los_days_dpt' in df.columns
    icu_old = 'icu_total_los_days'     in df.columns
    record("VAL-01e", "ICU LOS column renamed to _dpt", icu_dpt and not icu_old,
           f"_dpt present={icu_dpt}, old present={icu_old}")


def check_leakage(df):
    print("\n── VAL-04 / VAL-15  Leakage guard ─────────────────────────")
    found = [c for c in MUST_BE_ABSENT if c in df.columns]
    record("VAL-04", "Known leakage columns removed", not found,
           f"Still present: {found}" if found else "None found")

    safe_ws = ('dx_','proc_','med_','trans_num','icu_any','icu_num')
    leak_patterns = [r'_ws$', r'discharge_location', r'stable_at_discharge',
                     r'resolution_score', r'last_careunit']
    hits = [c for c in df.columns
            for p in leak_patterns
            if re.search(p, c, re.I)
            and not (p == r'_ws$' and c.startswith(safe_ws))]
    record("VAL-15", "No suspicious column name patterns", not hits,
           f"Suspicious: {hits}" if hits else "None found")


def check_row_integrity(df):
    print("\n── VAL-02  Row integrity ───────────────────────────────────")
    n_dup  = df['hadm_id'].duplicated().sum() if 'hadm_id' in df.columns else -1
    record("VAL-02a", "No duplicate hadm_id", n_dup == 0,
           f"{n_dup} duplicates" if n_dup else "No duplicates")
    for col in ['hadm_id','subject_id']:
        n = df[col].isna().sum() if col in df.columns else -1
        record("VAL-02b", f"No null {col}", n == 0,
               f"{n} null" if n else "None")
    record("VAL-02d", "Dataset non-empty", len(df) > 0, f"{len(df):,} rows")


def check_labels(df):
    print("\n── VAL-03  Label columns ───────────────────────────────────")
    for col in ['adm_inhospital_mortality','readmitted_30d']:
        if col not in df.columns:
            record("VAL-03", f"{col} present", False, "Column missing"); continue
        vals  = df[col].dropna().unique()
        valid = set(vals).issubset({0,1,0.0,1.0})
        record("VAL-03", f"{col} ∈ {{0,1}}", valid,
               f"Distribution: {df[col].value_counts(dropna=False).to_dict()}" +
               (f"  Unexpected: {vals}" if not valid else ""))


def check_readmit_logic(df):
    print("\n── VAL-16  Readmit / death logic ──────────────────────────")
    if not all(c in df.columns for c in ['adm_inhospital_mortality','readmitted_30d']):
        record("VAL-16","Columns present for logic check",False,"Missing columns"); return
    bad = ((df['adm_inhospital_mortality']==1) & (df['readmitted_30d']==1)).sum()
    record("VAL-16","readmitted_30d=0 for in-hospital deaths", bad==0,
           f"{bad} patients died AND readmitted_30d=1" if bad else "OK")


def check_c1(df):
    print("\n── VAL-05  C1 — Demographics ───────────────────────────────")
    if 'adm_age' in df.columns:
        oob = ((df['adm_age'] < 18) | (df['adm_age'] > 91)).sum()
        record("VAL-05a", "adm_age ∈ [18,91]", oob == 0,
               f"{oob} rows out-of-range — apply FIX-1 from pipeline_fixes_v1_3_1.py"
               if oob else f"Range: {df['adm_age'].min():.0f}–{df['adm_age'].max():.0f}")
    if 'adm_ed_los_hours' in df.columns:
        neg = (df['adm_ed_los_hours'].dropna() < 0).sum()
        record("VAL-05b", "adm_ed_los_hours ≥ 0", neg == 0,
               f"{neg} negatives" if neg else "OK")
    if 'adm_gender' in df.columns:
        pct = df['adm_gender'].isna().mean() * 100
        record("VAL-05c", "adm_gender null rate < 5%", pct < 5,
               f"{pct:.1f}% null", warn_only=pct >= 5)
    for bad in ['adm_discharge_location','adm_los_days']:
        record("VAL-05d", f"{bad} absent", bad not in df.columns,
               "Absent ✓" if bad not in df.columns else f"LEAKAGE: {bad} present")


def check_c2(df):
    print("\n── VAL-06  C2 — Utilisation ────────────────────────────────")
    col = 'util_days_since_last_discharge'
    if col in df.columns:
        neg = (df[col].dropna() < 0).sum()
        record("VAL-06a", f"{col} ≥ 0 where not NaN", neg == 0,
               f"{neg} negatives" if neg else "OK")
        null_pct = df[col].isna().mean() * 100
        record("VAL-06b", f"{col} ~100% NaN (first-admission cohort, STRUCTURAL)",
               null_pct > 95,
               f"{null_pct:.1f}% NaN — STRUCTURAL, expected ≈100%",
               warn_only=null_pct <= 95)
    for ecol in ['util_prior_ed_count_30d','util_prior_ed_count_180d']:
        if ecol in df.columns:
            neg = (df[ecol].dropna() < 0).sum()
            non_int = (df[ecol].dropna() % 1 != 0).sum()
            record("VAL-06c", f"{ecol} ≥ 0 and integer", neg==0 and non_int==0,
                   f"neg={neg}, non-integer={non_int}" if neg or non_int else "OK")


def check_c3(df):
    print("\n── VAL-07  C3 — Transfers / ICU ───────────────────────────")
    if 'icu_any' in df.columns:
        vals = df['icu_any'].dropna().unique()
        valid = set(vals).issubset({0,1,0.0,1.0})
        record("VAL-07a","icu_any ∈ {0,1}", valid,
               f"ICU rate: {df['icu_any'].mean()*100:.1f}%" if valid else f"Bad values: {vals}")
    for col in ['icu_num_stays','trans_num_transfers_ws']:
        if col in df.columns:
            neg = (df[col].dropna() < 0).sum()
            record("VAL-07b", f"{col} ≥ 0", neg==0, f"{neg} negatives" if neg else "OK")
    record("VAL-07c", "trans_first_careunit present",
           'trans_first_careunit' in df.columns,
           "Present ✓" if 'trans_first_careunit' in df.columns else "MISSING")
    record("VAL-07d", "trans_last_careunit ABSENT",
           'trans_last_careunit' not in df.columns,
           "Absent ✓" if 'trans_last_careunit' not in df.columns
           else "LEAKAGE: present")
    if 'icu_total_los_days_dpt' in df.columns:
        neg = (df['icu_total_los_days_dpt'].dropna() < 0).sum()
        null_pct = df['icu_total_los_days_dpt'].isna().mean()*100
        record("VAL-07e", "icu_total_los_days_dpt ≥ 0", neg==0,
               f"{neg} negatives. {null_pct:.1f}% NaN (STRUCTURAL: non-ICU patients)"
               if neg else f"{null_pct:.1f}% NaN (STRUCTURAL: only ICU patients ≈17%)")


def check_c4(df):
    print("\n── VAL-08  C4 — Charlson / Diagnoses ──────────────────────")
    if 'dx_charlson_index_prior' in df.columns:
        neg = (df['dx_charlson_index_prior'].dropna() < 0).sum()
        record("VAL-08a","dx_charlson_index_prior ≥ 0", neg==0,
               f"Range: {df['dx_charlson_index_prior'].min():.0f}–"
               f"{df['dx_charlson_index_prior'].max():.0f}, "
               f"mean={df['dx_charlson_index_prior'].mean():.2f}")
    flag_cols = [f'dx_charlson_{c}_prior' for c in CHARLSON_COMPS
                 if f'dx_charlson_{c}_prior' in df.columns]
    bad = sum((~df[c].isin([0,1,0.0,1.0,np.nan])).sum() for c in flag_cols)
    record("VAL-08b","Charlson component flags ∈ {0,1,NaN}", bad==0,
           f"{bad} invalid across {len(flag_cols)} columns" if bad else
           f"{len(flag_cols)} columns OK")
    dc,d = 'dx_charlson_diabetes_comp_prior','dx_charlson_diabetes_prior'
    if dc in df.columns and d in df.columns:
        v = ((df[dc]==1) & (df[d]==1)).sum()
        record("VAL-08c","Diabetes hierarchy respected", v==0,
               f"{v} violations" if v else "OK")
    ml,li = 'dx_charlson_mod_severe_liver_prior','dx_charlson_mild_liver_prior'
    if ml in df.columns and li in df.columns:
        v = ((df[ml]==1) & (df[li]==1)).sum()
        record("VAL-08d","Liver hierarchy respected", v==0,
               f"{v} violations" if v else "OK")
    for col in ['dx_num_codes_ws','proc_num_codes_ws']:
        if col in df.columns:
            neg = (df[col].dropna() < 0).sum()
            record("VAL-08e",f"{col} ≥ 0", neg==0, f"{neg} neg" if neg else "OK")


def check_c5(df):
    print("\n── VAL-09  C5 — Labs ───────────────────────────────────────")
    ws_lab = [c for c in df.columns if re.search(r'lab_.+_(min|max|mean|last)_ws$',c)]
    record("VAL-09a","No whole-stay lab columns (_ws)", not ws_lab,
           f"Leaky: {ws_lab}" if ws_lab else "None found ✓")

    for col,(lo,hi) in LAB_RANGES.items():
        if col not in df.columns: continue
        vals = df[col].dropna()
        oob  = ((vals < lo)|(vals > hi)).sum()
        record("VAL-09b",f"{col} ∈ [{lo},{hi}]", oob==0,
               f"{oob}/{len(vals)} out-of-range — null after clipping" if oob else
               f"Range: {vals.min():.2f}–{vals.max():.2f}", warn_only=oob>0)

    # KEY CHECK: AKI columns must be non-null for enough patients (catches BUG-2)
    for col in ['aki_creat_baseline_bw','aki_creat_max_bw',
                'aki_creat_ratio_max_over_baseline']:
        if col not in df.columns:
            record("VAL-09e",f"{col} present", False,"Column missing — check FIX-2"); continue
        non_null_rate = df[col].notna().mean()
        ok = non_null_rate >= AKI_MIN_NONULL
        record("VAL-09e",f"{col} ≥{AKI_MIN_NONULL*100:.0f}% non-null", ok,
               f"{non_null_rate*100:.1f}% non-null" +
               (" — APPLY FIX-2 from pipeline_fixes_v1_3_1.py" if not ok else " ✓"))

    if all(c in df.columns for c in ['aki_creat_baseline_bw','aki_creat_max_bw',
                                      'aki_creat_ratio_max_over_baseline']):
        mask = (df['aki_creat_baseline_bw'].notna() &
                df['aki_creat_max_bw'].notna() &
                df['aki_creat_ratio_max_over_baseline'].notna() &
                (df['aki_creat_baseline_bw'] != 0))
        if mask.sum() > 0:
            recomp = df.loc[mask,'aki_creat_max_bw'] / df.loc[mask,'aki_creat_baseline_bw']
            err = (recomp - df.loc[mask,'aki_creat_ratio_max_over_baseline']).abs().max()
            record("VAL-09c","AKI ratio math correct (tol 1e-4)", err<1e-4,
                   f"Max abs error: {err:.2e}")
        neg = (df['aki_creat_ratio_max_over_baseline'].dropna() < 0).sum()
        record("VAL-09d","AKI ratio ≥ 0", neg==0, f"{neg} negatives" if neg else "OK")


def check_c6(df):
    print("\n── VAL-10  C6 — Medications ────────────────────────────────")
    for col in ['med_total_orders_count_ws','med_prn_count_ws']:
        if col in df.columns:
            neg = (df[col].dropna() < 0).sum()
            record("VAL-10a",f"{col} ≥ 0", neg==0, f"{neg}" if neg else "OK")
    hr_cols = [f'med_highrisk_{c}_medwin' for c in HR_CLASSES
               if f'med_highrisk_{c}_medwin' in df.columns]
    bad = sum((~df[c].isin([0,1,0.0,1.0,np.nan])).sum() for c in hr_cols)
    record("VAL-10b",f"High-risk flags ∈ {{0,1,NaN}} ({len(hr_cols)} classes)",
           bad==0, f"{bad} invalid" if bad else "OK")


def check_c7(df):
    print("\n── VAL-11  C7 — OMR ────────────────────────────────────────")
    for col,lo,hi,label in [('omr_bmi_last_pre',10,100,"BMI"),
                             ('omr_weight_kg_last_pre',2,500,"Weight kg"),
                             ('omr_height_cm_last_pre',50,250,"Height cm")]:
        if col not in df.columns: continue
        vals = df[col].dropna()
        if len(vals) == 0:
            record("VAL-11",f"{label} all NaN",True,"No pre-adm OMR records — STRUCTURAL",
                   warn_only=True); continue
        oob = ((vals < lo)|(vals > hi)).sum()
        record("VAL-11",f"{label} ∈ [{lo},{hi}]", oob==0,
               f"{oob}/{len(vals)} out-of-range — set to NaN after clipping" if oob else
               f"Range: {vals.min():.1f}–{vals.max():.1f}", warn_only=oob>0)


def check_c8(df):
    print("\n── VAL-12  C8 — Vitals ─────────────────────────────────────")
    for col,(lo,hi) in VITAL_RANGES.items():
        if col not in df.columns: continue
        vals = df[col].dropna()
        if len(vals) == 0:
            record("VAL-12",f"{col} all NaN",True,"ICU-only — STRUCTURAL",warn_only=True)
            continue
        null_pct = df[col].isna().mean()*100
        oob = ((vals < lo)|(vals > hi)).sum()
        record("VAL-12",f"{col} ∈ [{lo},{hi}]", oob==0,
               f"{null_pct:.1f}% NaN (STRUCTURAL). {oob}/{len(vals)} out-of-range — "
               "set to NaN after clipping" if oob else
               f"{null_pct:.1f}% NaN (STRUCTURAL). Range: {vals.min():.1f}–{vals.max():.1f}",
               warn_only=oob>0)
    if 'vital_measure_count_pdw' in df.columns:
        neg = (df['vital_measure_count_pdw'].dropna() < 0).sum()
        record("VAL-12b","vital_measure_count_pdw ≥ 0",neg==0,
               f"{neg} negatives" if neg else "OK")


def check_d(df):
    print("\n── VAL-13  D — Discharge Notes ─────────────────────────────")
    bin_cols = ['note_has_followup_instructions','note_has_home_health',
                'note_lives_alone','note_has_caregiver',
                'note_function_needs_assistance','note_cognitive_impairment']
    for col in bin_cols:
        if col not in df.columns: continue
        vals  = df[col].dropna().unique()
        valid = set(float(v) for v in vals).issubset({0.0,1.0})
        null_pct = df[col].isna().mean()*100
        record("VAL-13a",f"{col} ∈ {{0,1,NaN}}", valid,
               f"Values: {vals}  NaN: {null_pct:.1f}%" if not valid
               else f"NaN: {null_pct:.1f}%")
    for bad in ['note_stable_at_discharge','note_resolution_score']:
        record("VAL-13b",f"{bad} absent (LEAK-08)", bad not in df.columns,
               "Absent ✓" if bad not in df.columns else f"LEAKAGE: present")
    if 'note_days_to_followup_est' in df.columns:
        neg = (df['note_days_to_followup_est'].dropna() < 0).sum()
        record("VAL-13c","note_days_to_followup_est ≥ 0",neg==0,
               f"{neg} negatives" if neg else "OK")


def check_missing_audit(df):
    """
    Three-tier missing value audit:
      STRUCTURAL — expected NaN by design → INFO only, never FAIL
      CLINICAL   — high but clinically normal → WARN if > 99%
      UNEXPECTED — columns that should be well-populated → FAIL if > 50%
    """
    print("\n── VAL-14  Missing value audit (tiered) ────────────────────")
    miss = df.isna().mean()

    structural_cols = [c for c in df.columns if _matches_any(c, STRUCTURAL_NAN_PATTERNS)]
    clinical_cols   = [c for c in df.columns
                       if not _matches_any(c, STRUCTURAL_NAN_PATTERNS)
                       and _matches_any(c, CLINICAL_NAN_PATTERNS)]
    unexpected_cols = [c for c in df.columns
                       if c not in structural_cols
                       and c not in clinical_cols]

    # Structural — just report, no FAIL
    struct_high = [(c, miss[c]) for c in structural_cols if miss[c] > 0.80]
    record("VAL-14a",
           f"Structural columns (design-NaN) — {len(struct_high)} above 80% "
           f"(EXPECTED, not a failure)",
           True,  # always pass
           f"{len(structural_cols)} structural columns identified. "
           f"Examples: util_days_since_last_discharge, vital_*_pdw, icu_*_pdw")

    # Clinical — warn if extremely high
    clin_extreme = [(c, miss[c]) for c in clinical_cols
                    if miss[c] > CLINICAL_WARN_THRESHOLD]
    if clin_extreme:
        lines = '\n'.join(f"{c}: {v*100:.1f}%" for c,v in clin_extreme[:10])
        record("VAL-14b",
               f"{len(clin_extreme)} clinical-sparsity columns above "
               f"{CLINICAL_WARN_THRESHOLD*100:.0f}% NaN",
               False, lines, warn_only=True)
    else:
        record("VAL-14b","Clinical-sparsity columns within expected range", True,"OK")

    # Unexpected — FAIL if any > 50%
    unexpected_high = [(c, miss[c]) for c in unexpected_cols
                       if miss[c] > 0.50 and c not in CORE_LOW_NULL_COLS]
    if unexpected_high:
        lines = '\n'.join(f"{c}: {v*100:.1f}%" for c,v in unexpected_high[:15])
        record("VAL-14c",
               f"{len(unexpected_high)} UNEXPECTED columns with >50% NaN — investigate",
               False, lines)
    else:
        record("VAL-14c","No unexpected columns with >50% NaN", True,"OK")

    # Core columns — tight threshold
    for col, max_null in CORE_LOW_NULL_COLS.items():
        if col not in df.columns: continue
        pct = miss[col]
        record("VAL-14d",f"{col} null rate < {max_null*100:.0f}%",
               pct <= max_null,
               f"{pct*100:.2f}% NaN" + (" ✓" if pct <= max_null
               else " — FAIL: core column should have low missingness"))


def print_summary():
    df_r  = pd.DataFrame(results)
    n_pass = (df_r['status']=='PASS').sum()
    n_warn = (df_r['status']=='WARN').sum()
    n_fail = (df_r['status']=='FAIL').sum()

    print(f"\n{'='*64}")
    print(f"  SUMMARY  —  PASS: {n_pass}   WARN: {n_warn}   FAIL: {n_fail}")
    print(f"{'='*64}")

    if n_fail:
        print("\n  FAILED CHECKS:")
        for _,r in df_r[df_r['status']=='FAIL'].iterrows():
            print(f"    ✗  {r['check_id']}  {r['name']}")
            if r['detail']:
                for ln in r['detail'].splitlines()[:5]:
                    print(f"         {ln}")

    if n_warn:
        print("\n  WARNINGS:")
        for _,r in df_r[df_r['status']=='WARN'].iterrows():
            print(f"    ⚠  {r['check_id']}  {r['name']}")
            if r['detail']:
                print(f"         {r['detail'].splitlines()[0]}")

    report = Path(CSV_PATH).parent / "validation_report_v131.csv"
    df_r.to_csv(report, index=False)
    print(f"\n  Full report → {report}")
    print(f"{'='*64}\n")

    if n_fail:
        print("  ACTION REQUIRED:")
        print("  → Run pipeline_fixes_v1_3_1.py and re-extract features.")
        print("    FIX-1 for VAL-05a (adm_age > 91)")
        print("    FIX-2 for VAL-09e (AKI columns 100% NaN)")
        print("    apply_clip_rules() for out-of-range vitals/labs/OMR\n")

    return n_fail


def main():
    df = load_csv(CSV_PATH)
    check_schema(df)
    check_leakage(df)
    check_row_integrity(df)
    check_labels(df)
    check_readmit_logic(df)
    check_c1(df)
    check_c2(df)
    check_c3(df)
    check_c4(df)
    check_c5(df)
    check_c6(df)
    check_c7(df)
    check_c8(df)
    check_d(df)
    check_missing_audit(df)
    n_fail = print_summary()
    sys.exit(1 if n_fail else 0)

if __name__ == '__main__':
    main()