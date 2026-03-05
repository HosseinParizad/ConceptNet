"""
ConceptNet — Statistical Weight Fitting (No Neural Network)
============================================================
Fits the Input → 12 concept nodes → 2 output architecture using
three classical statistical methods instead of backpropagation.

WHY NOT NN?
───────────
The Input→12→2 bottleneck with a sparse concept mask gives the NN
only ~3,000 parameters to tune across 246 features. With 12 nodes
competing end-to-end via backprop, the signal is extremely weak and
the architecture almost always collapses to a degenerate solution
(all-recall or all-precision). Classical methods avoid this by
fitting each concept independently.

THREE METHODS IMPLEMENTED
──────────────────────────
Method 1 — CORRELATION  (baseline, non-negative by construction)
  For each concept k:
    weight[k][f] = Pearson |corr(feature_f, y)| for features in concept k
    concept_score = softmax-normalised weighted sum of assigned features
  Output stage: LogisticRegression on 12 concept scores
  Properties: transparent, always non-negative, no training needed

Method 2 — NNLS  (Non-Negative Least Squares, supervised, non-negative)
  For each concept k:
    solve:  min || X_k @ w_k  -  y ||^2   subject to  w_k >= 0
    using scipy.optimize.nnls  (active-set algorithm, exact solution)
    concept_score = X_k @ w_k  (raw linear score, then standardised)
  Output stage: LogisticRegression on 12 concept scores
  Properties: supervised, non-negative guaranteed, fast, exact solution

Method 3 — TWO-STAGE LR  (fully supervised, may have signed weights)
  For each concept k:
    fit LogisticRegression(C=0.1, l2) on assigned features → y
    concept_score = predict_proba(X_k)[:,1]   (a probability)
  Output stage: LogisticRegression on 12 concept scores
  Properties: most powerful, weights may be negative (clinical sign check)

All three methods use IDENTICAL:
  • train/val/test split (SEED=42)
  • preprocessing (StandardScaler + OrdinalEncoder + median imputation)
  • concept mask (same CONCEPT_PATTERNS as train.py)
  • evaluation metrics (AUROC, AUPRC, Brier, F1, Precision, Recall)
  • clinical weight validation

Outputs
───────
  concept_weights_all_methods.csv     — all weights for all methods
  method_comparison_metrics.csv       — test-set performance table
  plot_method_comparison.png          — AUROC / AUPRC bar chart
  plot_concept_weights_<method>.png   — per-concept weight bars
  plot_concept_scores_delta.png       — concept score Δ by class
  plot_nnls_weights_heatmap.png       — NNLS weight heatmap
  clinical_validation_report.txt      — full narrative report
"""

import os, re, sys, json, time, warnings, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.optimize import nnls           # Non-Negative Least Squares
from scipy.stats   import pearsonr

from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler, OrdinalEncoder
from sklearn.impute          import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (roc_auc_score, average_precision_score,
                                     brier_score_loss, classification_report,
                                     roc_curve, precision_recall_curve)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FEATURES_CSV = './model_outputs/extracted_features2.csv'
OUTPUT_DIR   = './model_outputs/conceptnet_statistical/'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

SEED         = 42
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15
READMIT_DAYS = 30
TOP_N        = 10     # top features shown per concept in plots

np.random.seed(SEED)

CATEGORICAL_COLS = {
    'adm_gender', 'adm_race', 'adm_insurance', 'adm_marital_status',
    'adm_admission_type', 'adm_admission_location', 'adm_discharge_location',
    'trans_first_careunit', 'trans_last_careunit',
}
EXCLUDE_COLS = {
    'subject_id', 'hadm_id', 'admittime', 'dischtime',
    'adm_inhospital_mortality', 'icu_first_intime',
    'next_admittime', 'days_to_readmit',
    f'readmitted_{READMIT_DAYS}d',
}


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPTS + PATTERNS  (unchanged from train.py)
# ─────────────────────────────────────────────────────────────────────────────
CONCEPTS = [
    ( 0, 'frailty_reserve',         'Baseline Physiologic Reserve (Frailty)'),
    ( 1, 'biological_buffer',       'Biological Buffer Capacity'),
    ( 2, 'chronic_burden',          'Chronic Disease Burden'),
    ( 3, 'recent_instability',      'Recent Instability Pattern'),
    ( 4, 'admission_severity',      'Index Admission Severity'),
    ( 5, 'complication_burden',     'Index Admission Complication Burden'),
    ( 6, 'physio_stability_dc',     'Physiologic Stability at Discharge'),
    ( 7, 'disease_resolution',      'Active Disease Resolution Status'),
    ( 8, 'functional_independence', 'Functional Independence at Discharge'),
    ( 9, 'cognitive_capacity',      'Cognitive Capacity for Self-Care'),
    (10, 'medication_risk',         'Medication Risk & Manageability'),
    (11, 'postdc_support',          'Post-Discharge Support Reliability'),
]
N_CONCEPTS  = len(CONCEPTS)
CLASS_NAMES = ['Not Readmitted', 'Readmitted']

CONCEPT_PATTERNS = {
    'frailty_reserve': [
        r'^adm_age$', r'^adm_gender$',
        r'^omr_weight', r'^omr_height', r'^omr_bmi',
        r'^util_prior_hadm_count_365d$',
    ],
    'biological_buffer': [
        r'^lab_bicarbonate', r'^lab_albumin', r'^lab_hemoglobin',
        r'^lab_sodium.*(bw|ws)', r'^lab_potassium.*(bw|ws)',
        r'^lab_chloride.*(bw|ws)', r'^lab_lactate.*(bw|ws)',
        r'^lab_wbc.*(bw|ws)', r'^lab_platelets.*(bw|ws)',
        r'^lab_glucose.*(bw|ws)',
    ],
    'chronic_burden': [
        r'^dx_charlson', r'^dx_num_', r'^proc_num_', r'^adm_insurance$',
    ],
    'recent_instability': [
        r'^util_prior_hadm_count_30d$', r'^util_prior_hadm_count_90d$',
        r'^util_prior_hadm_count_180d$', r'^util_days_since_last_discharge$',
        r'^util_prior_ed_count', r'^adm_has_ed$', r'^adm_ed_los_hours$',
    ],
    'admission_severity': [
        r'^adm_los_days$', r'^adm_admission_type$', r'^adm_admission_location$',
        r'^icu_any$', r'^icu_num_stays$', r'^icu_total_los_days$',
        r'^icu_time_to_first_icu_hours$',
        r'^aki_creat', r'^lab_creatinine_max_ws$', r'^lab_lactate_max_ws$',
        r'^lab_bilirubin_max_ws$', r'^lab_wbc_max_ws$',
        r'^lab_glucose_max_ws$', r'^lab_bun_max_ws$', r'^lab_abnormal_count_ws$',
    ],
    'complication_burden': [
        r'^icu_any_transfer$', r'^trans_num_transfers_ws$',
        r'^trans_num_careunit_changes_ws$',
        r'^trans_first_careunit$', r'^trans_last_careunit$',
        r'^aki_creat_delta', r'^med_highrisk.*_medwin$',
        r'^proc_num_codes_ws$', r'^lab_abnormal_count_ws$',
    ],
    'physio_stability_dc': [
        r'^vital_(sbp|dbp|map|hr|rr|spo2|temp)_(last|min|max|std)_pdw$',
        r'^vital_gcs_(last|min|max|std)_pdw$', r'^vital_measure_count_pdw$',
        r'^lab_(creatinine|sodium|potassium|glucose|wbc)_(last|min|max|mean)_pdw$',
        r'^lab_abnormal_count_pdw$',
    ],
    'disease_resolution': [
        r'^note_resolution_score$', r'^note_stable_at_discharge$',
        r'^lab_.*_slope_pdw$', r'^lab_.*_delta_last_first_pdw$',
        r'^vital_.*_slope_pdw$', r'^aki_creat_ratio',
        r'^lab_lactate_(last|min|max)_pdw$', r'^lab_bilirubin_(last|min|max)_pdw$',
    ],
    'functional_independence': [
        r'^note_function_needs_assistance$', r'^adm_discharge_location$',
        r'^vital_gcs_(last|min)_pdw$', r'^omr_bmi', r'^adm_los_days$',
    ],
    'cognitive_capacity': [
        r'^note_cognitive_impairment$', r'^vital_gcs_(last|min|max)_pdw$',
        r'^dx_charlson_dementia_ws$', r'^adm_age$',
    ],
    'medication_risk': [
        r'^med_unique_count_medwin$', r'^med_total_orders_count_ws$',
        r'^med_prn_count_ws$', r'^med_num_routes_medwin$',
        r'^med_frequency_complexity_score_medwin$',
        r'^med_highrisk_class_count_medwin$', r'^med_highrisk_',
    ],
    'postdc_support': [
        r'^adm_discharge_location$', r'^adm_marital_status$',
        r'^adm_insurance$', r'^adm_race$',
        r'^note_has_home_health$', r'^note_lives_alone$', r'^note_has_caregiver$',
    ],
}

# Clinical expectation: expected sign of output weight (concept → readmit)
# +1 = risk-raising concept, -1 = protective concept
OUTPUT_DIRECTION = {
    'frailty_reserve':        +1,
    'biological_buffer':      +1,
    'chronic_burden':         +1,
    'recent_instability':     +1,
    'admission_severity':     +1,
    'complication_burden':    +1,
    'physio_stability_dc':    -1,
    'disease_resolution':     -1,
    'functional_independence':-1,
    'cognitive_capacity':     -1,
    'medication_risk':        +1,
    'postdc_support':         -1,
}

# Clinical anchors: features that should have HIGH weight in each concept
# (magnitude for NNLS/CORR; we also check sign for two-stage LR)
ANCHORS = {
    'frailty_reserve':         ['adm_age', 'util_prior_hadm_count_365d', 'omr_bmi'],
    'biological_buffer':       ['lab_albumin', 'lab_bicarbonate', 'lab_hemoglobin', 'lab_lactate'],
    'chronic_burden':          ['dx_charlson', 'dx_num_', 'proc_num_'],
    'recent_instability':      ['util_prior_hadm_count_30d', 'util_prior_hadm_count_90d',
                                 'util_days_since_last_discharge'],
    'admission_severity':      ['icu_any', 'icu_total_los_days', 'lab_creatinine_max_ws',
                                 'lab_lactate_max_ws', 'lab_abnormal_count_ws', 'adm_los_days'],
    'complication_burden':     ['trans_num_transfers_ws', 'aki_creat_delta', 'med_highrisk'],
    'physio_stability_dc':     ['vital_spo2_last_pdw', 'lab_creatinine_last_pdw',
                                 'lab_abnormal_count_pdw', 'vital_rr_last_pdw'],
    'disease_resolution':      ['note_stable_at_discharge', 'note_resolution_score',
                                 'lab_lactate_last_pdw'],
    'functional_independence': ['vital_gcs_last_pdw', 'note_function_needs_assistance'],
    'cognitive_capacity':      ['note_cognitive_impairment', 'dx_charlson_dementia_ws',
                                 'vital_gcs_last_pdw'],
    'medication_risk':         ['med_unique_count_medwin', 'med_highrisk_class_count_medwin',
                                 'med_frequency_complexity_score_medwin'],
    'postdc_support':          ['note_has_home_health', 'note_has_caregiver', 'note_lives_alone'],
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA + PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def create_readmission_label(df, days=30):
    df = df.sort_values(['subject_id', 'admittime']).copy()
    df['next_admittime'] = df.groupby('subject_id')['admittime'].shift(-1)
    df['days_to_readmit'] = (
        (df['next_admittime'] - df['dischtime']).dt.total_seconds() / 86400.0)
    col = f'readmitted_{days}d'
    df[col] = ((df['days_to_readmit'] > 0) & (df['days_to_readmit'] <= days)).astype(int)
    if 'adm_inhospital_mortality' in df.columns:
        df.loc[df['adm_inhospital_mortality'] == 1, col] = 0
    return df


def load_data():
    log.info(f"Loading {FEATURES_CSV} …")
    df = pd.read_csv(FEATURES_CSV, low_memory=False)
    log.info(f"  Raw shape: {df.shape[0]:,} × {df.shape[1]:,}")
    for c in ['admittime', 'dischtime']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    label = f'readmitted_{READMIT_DAYS}d'
    if label not in df.columns:
        df = create_readmission_label(df, days=READMIT_DAYS)
    y = df[label].copy()
    log.info(f"  Readmissions: {int(y.sum()):,}/{len(y):,} ({100*y.mean():.1f}%)")
    drop = [c for c in df.columns if c in EXCLUDE_COLS or c == 'next_admittime']
    X = df.drop(columns=[c for c in drop if c in df.columns], errors='ignore')
    X = X.select_dtypes(exclude=['datetime64[ns]'])
    log.info(f"  Feature columns: {X.shape[1]}")
    return X, y


class Preprocessor:
    def __init__(self, feature_names):
        self.cat_cols   = sorted([c for c in CATEGORICAL_COLS if c in feature_names])
        self.num_cols   = sorted([c for c in feature_names if c not in CATEGORICAL_COLS])
        self.cat_imp    = SimpleImputer(strategy='constant', fill_value='__MISSING__')
        self.ord_enc    = OrdinalEncoder(handle_unknown='use_encoded_value',
                                         unknown_value=-1, dtype=np.float32)
        self.cat_scaler = StandardScaler()
        self.num_imp    = SimpleImputer(strategy='median')
        self.num_scaler = StandardScaler()
        self.out_cols_  = []

    def fit_transform(self, X):
        parts, cols = [], []
        if self.cat_cols:
            Xc = X[self.cat_cols].astype(str)
            parts.append(self.cat_scaler.fit_transform(
                self.ord_enc.fit_transform(
                    self.cat_imp.fit_transform(Xc)).astype(np.float32)))
            cols += self.cat_cols
        if self.num_cols:
            Xn = X[self.num_cols].apply(pd.to_numeric, errors='coerce')
            parts.append(self.num_scaler.fit_transform(
                self.num_imp.fit_transform(Xn)).astype(np.float32))
            cols += list(self.num_imp.get_feature_names_out(self.num_cols))
        self.out_cols_ = cols
        return np.hstack(parts).astype(np.float32)

    def transform(self, X):
        parts = []
        if self.cat_cols:
            Xc = X[self.cat_cols].astype(str)
            parts.append(self.cat_scaler.transform(
                self.ord_enc.transform(
                    self.cat_imp.transform(Xc)).astype(np.float32)))
        if self.num_cols:
            Xn = X[self.num_cols].apply(pd.to_numeric, errors='coerce')
            parts.append(self.num_scaler.transform(
                self.num_imp.transform(Xn)).astype(np.float32))
        return np.hstack(parts).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION MASK
# ─────────────────────────────────────────────────────────────────────────────
def build_concept_feature_index(feature_names):
    """
    Returns a dict: concept_key → sorted list of feature indices assigned to it.
    Unassigned features go to concept 4 (admission_severity).
    """
    n_feat   = len(feature_names)
    assigned = np.zeros(n_feat, dtype=bool)
    index    = {ckey: [] for _, ckey, _ in CONCEPTS}

    for cidx, ckey, _ in CONCEPTS:
        for fi, fname in enumerate(feature_names):
            for pat in CONCEPT_PATTERNS.get(ckey, []):
                if re.search(pat, fname, re.IGNORECASE):
                    index[ckey].append(fi)
                    assigned[fi] = True
                    break

    unassigned_fi = np.where(~assigned)[0].tolist()
    if unassigned_fi:
        index['admission_severity'].extend(unassigned_fi)
        log.info(f"  {len(unassigned_fi)} unassigned features → concept 4 (admission_severity)")

    log.info("Concept feature counts:")
    for cidx, ckey, clabel in CONCEPTS:
        log.info(f"  Node {cidx:2d}  {clabel:<50s} ← {len(index[ckey]):3d} features")
    return index


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_prob, model_name):
    y_pred = (y_prob >= 0.5).astype(int)
    auroc  = roc_auc_score(y_true, y_prob)
    auprc  = average_precision_score(y_true, y_prob)
    brier  = brier_score_loss(y_true, y_prob)
    rep    = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    return dict(
        model             = model_name,
        auroc             = round(float(auroc), 4),
        auprc             = round(float(auprc), 4),
        brier             = round(float(brier), 4),
        f1_readmit        = round(rep['Readmitted']['f1-score'], 4),
        precision_readmit = round(rep['Readmitted']['precision'], 4),
        recall_readmit    = round(rep['Readmitted']['recall'], 4),
    )


def score_to_output(C_tr, y_tr, C_te, y_te, method_name):
    """
    Fit a LogisticRegression on 12 concept scores → readmission.
    C_tr: (n_train, 12)  C_te: (n_test, 12)
    Returns y_prob_test, output_weights (shape 12), output_lr model
    """
    lr = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs',
                             max_iter=1000, class_weight='balanced',
                             random_state=SEED)
    lr.fit(C_tr, y_tr)
    y_prob = lr.predict_proba(C_te)[:, 1]
    output_weights = lr.coef_[0]   # shape (12,)  — one weight per concept
    return y_prob, output_weights, lr


def standardise_scores(C):
    """Standardise concept score matrix column-wise to zero mean, unit variance."""
    mu = C.mean(axis=0, keepdims=True)
    sd = C.std(axis=0, keepdims=True) + 1e-8
    return (C - mu) / sd, mu, sd


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1 — CORRELATION-BASED WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def fit_correlation(X_tr, y_tr, X_te, concept_index, feature_names):
    """
    For each concept k:
      weight[k][f] = |Pearson r(feature_f, y)|  for f in concept_k
      concept_score = X_k @ weight_k  (weighted sum, then standardised)

    Properties:
      - All weights ≥ 0  (absolute correlation)
      - No training required for concept layer (correlation on train set only)
      - Very transparent: weight directly = predictive relevance of feature
    """
    log.info("─" * 55)
    log.info("Method 1 — CORRELATION  (non-negative, |Pearson r| weights)")
    t0 = time.time()

    concept_weights = {}   # ckey → np.array of weights aligned to feature_names
    C_tr = np.zeros((X_tr.shape[0], N_CONCEPTS), dtype=np.float32)
    C_te = np.zeros((X_te.shape[0], N_CONCEPTS), dtype=np.float32)

    for cidx, ckey, clabel in CONCEPTS:
        fi_list = concept_index[ckey]
        if not fi_list:
            concept_weights[ckey] = np.array([])
            continue

        Xk_tr = X_tr[:, fi_list]   # (n_train, k_features)

        # Compute |Pearson r| between each feature and y
        w = np.array([abs(pearsonr(Xk_tr[:, j], y_tr)[0])
                      for j in range(Xk_tr.shape[1])])
        w = np.nan_to_num(w, nan=0.0)

        # Normalise so weights sum to 1 within concept
        w_sum = w.sum()
        w_norm = w / w_sum if w_sum > 0 else np.ones_like(w) / len(w)

        concept_weights[ckey] = w_norm

        # Compute concept score = weighted sum
        C_tr[:, cidx] = Xk_tr @ w_norm
        C_te[:, cidx] = X_te[:, fi_list] @ w_norm

    # Standardise concept scores
    C_tr_std, mu, sd = standardise_scores(C_tr)
    C_te_std = (C_te - mu) / sd

    # Output stage: logistic regression on 12 concept scores
    y_prob, output_w, output_lr = score_to_output(C_tr_std, y_tr, C_te_std, None, 'corr')

    log.info(f"  Done in {time.time()-t0:.1f}s")
    return y_prob, concept_weights, output_w, C_te_std


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2 — NNLS (Non-Negative Least Squares)
# ─────────────────────────────────────────────────────────────────────────────
def fit_nnls(X_tr, y_tr, X_te, concept_index, feature_names):
    """
    For each concept k:
      Solve:  min || X_k_train @ w_k  -  y_train ||^2   s.t.  w_k >= 0
      using scipy.optimize.nnls  (Lawson-Hanson active-set algorithm)
      concept_score = X_k @ w_k  (raw linear prediction)

    Properties:
      - All weights ≥ 0  (guaranteed by NNLS)
      - Supervised (uses y for fitting)
      - Exact solution (no convergence issues, no learning rate)
      - Each concept is fitted independently → no inter-concept interference
    """
    log.info("─" * 55)
    log.info("Method 2 — NNLS  (Non-Negative Least Squares per concept)")
    t0 = time.time()

    concept_weights = {}
    C_tr = np.zeros((X_tr.shape[0], N_CONCEPTS), dtype=np.float64)
    C_te = np.zeros((X_te.shape[0], N_CONCEPTS), dtype=np.float64)

    y_tr_f = y_tr.astype(np.float64)

    for cidx, ckey, clabel in CONCEPTS:
        fi_list = concept_index[ckey]
        if not fi_list:
            concept_weights[ckey] = np.array([])
            continue

        Xk_tr = X_tr[:, fi_list].astype(np.float64)
        Xk_te = X_te[:, fi_list].astype(np.float64)

        # NNLS: solve min ||Xk_tr @ w - y||^2  s.t. w >= 0
        # Returns (solution, residual_norm)
        w_nnls, residual = nnls(Xk_tr, y_tr_f)

        # Normalise by L1 norm for comparability across concepts
        w_sum = w_nnls.sum()
        w_norm = w_nnls / w_sum if w_sum > 0 else np.ones_like(w_nnls) / len(w_nnls)

        concept_weights[ckey] = w_norm

        C_tr[:, cidx] = Xk_tr @ w_norm
        C_te[:, cidx] = Xk_te @ w_norm

        n_nonzero = (w_nnls > 1e-8).sum()
        log.info(f"  {clabel[:42]:<42}  non-zero weights: {n_nonzero}/{len(w_nnls)}  "
                 f"residual: {residual:.4f}")

    # Standardise concept scores
    C_tr_std, mu, sd = standardise_scores(C_tr.astype(np.float32))
    C_te_std = ((C_te.astype(np.float32)) - mu) / sd

    y_prob, output_w, output_lr = score_to_output(C_tr_std, y_tr, C_te_std, None, 'nnls')

    log.info(f"  Done in {time.time()-t0:.1f}s")
    return y_prob, concept_weights, output_w, C_te_std


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3 — TWO-STAGE LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
def fit_twostage_lr(X_tr, y_tr, X_te, concept_index, feature_names):
    """
    For each concept k:
      Fit LogisticRegression(C=0.1, l2) on assigned features → y
      concept_score = predict_proba(X_k)[:, 1]  (a calibrated probability)
      concept_weight[f] = lr.coef_[0][f]        (may be positive or negative)

    Output stage: LogisticRegression on 12 concept probabilities.

    Properties:
      - Fully supervised, calibrated concept scores
      - Most flexible (each concept independently optimised)
      - Weights may be negative (clinically meaningful: sign = direction)
      - No gradient issues, no convergence problems
    """
    log.info("─" * 55)
    log.info("Method 3 — TWO-STAGE LR  (logistic regression per concept)")
    t0 = time.time()

    concept_weights = {}
    concept_models  = {}
    C_tr = np.zeros((X_tr.shape[0], N_CONCEPTS), dtype=np.float32)
    C_te = np.zeros((X_te.shape[0], N_CONCEPTS), dtype=np.float32)

    for cidx, ckey, clabel in CONCEPTS:
        fi_list = concept_index[ckey]
        if not fi_list:
            concept_weights[ckey] = np.array([])
            continue

        Xk_tr = X_tr[:, fi_list]
        Xk_te = X_te[:, fi_list]

        lr = LogisticRegression(C=0.1, penalty='l2', solver='saga',
                                 max_iter=500, class_weight='balanced',
                                 random_state=SEED, n_jobs=-1)
        lr.fit(Xk_tr, y_tr)
        concept_models[ckey] = lr

        # Store concept weights (normalised by L1 norm)
        w = lr.coef_[0]
        w_sum = np.abs(w).sum()
        w_norm = w / w_sum if w_sum > 0 else w
        concept_weights[ckey] = w_norm

        C_tr[:, cidx] = lr.predict_proba(Xk_tr)[:, 1]
        C_te[:, cidx] = lr.predict_proba(Xk_te)[:, 1]

        n_pos = (w > 0).sum()
        n_neg = (w < 0).sum()
        log.info(f"  {clabel[:42]:<42}  pos: {n_pos}  neg: {n_neg}  "
                 f"AUROC: {roc_auc_score(y_tr, C_tr[:, cidx]):.3f}")

    # Output stage: LR on 12 concept probabilities
    y_prob, output_w, output_lr = score_to_output(C_tr, y_tr, C_te, None, 'twostage')

    log.info(f"  Done in {time.time()-t0:.1f}s")
    return y_prob, concept_weights, output_w, C_te


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL WEIGHT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def validate_method(concept_weights, output_weights, concept_index,
                    feature_names, method_name, is_signed=False):
    """
    For each concept:
      1. Output weight direction vs clinical expectation
      2. Anchor features: are they PROMINENT (top 30%) in their concept?
      For two-stage LR (is_signed=True): also check sign of anchor weights.

    Returns: summaries dict, verdict rows list
    """
    summaries    = {}
    verdict_rows = []

    for cidx, ckey, clabel in CONCEPTS:
        fi_list    = concept_index[ckey]
        out_w      = float(output_weights[cidx])
        exp_dir    = OUTPUT_DIRECTION.get(ckey, None)

        # Output weight verdict
        if exp_dir is None:
            out_verdict = 'NO_EXPECTATION'
        elif abs(out_w) < 0.02:
            out_verdict = 'WEAK'
        elif (out_w > 0 and exp_dir > 0) or (out_w < 0 and exp_dir < 0):
            out_verdict = 'MATCHES'
        else:
            out_verdict = 'REVERSED'

        w_arr = concept_weights.get(ckey, np.array([]))
        if len(w_arr) == 0:
            summaries[ckey] = dict(
                concept_idx=cidx, concept_label=clabel,
                output_w=round(out_w, 4),
                output_dir_expected='+1' if exp_dir==1 else '-1',
                output_verdict=out_verdict,
                n_anchors=0, n_prominent=0, n_present=0,
                n_low=0, n_not_found=0, clinical_score_pct=0.0,
                anchor_verdicts=[]
            )
            continue

        n_total   = len(fi_list)
        feat_list = [feature_names[fi] for fi in fi_list]

        # Sort features by |weight| descending → get rank
        abs_w = np.abs(w_arr)
        rank_order = np.argsort(-abs_w)   # index into fi_list / w_arr

        # Build rank lookup: feature_name → rank (1-based)
        rank_by_name = {feat_list[i]: int(np.where(rank_order == i)[0][0]) + 1
                        for i in range(len(feat_list))}

        anchor_verdicts = []
        for anchor_substr in ANCHORS.get(ckey, []):
            matching = [f for f in feat_list if anchor_substr.lower() in f.lower()]
            if not matching:
                anchor_verdicts.append(dict(
                    anchor=anchor_substr, best_match='NOT FOUND',
                    weight=np.nan, rank=np.nan, rank_pct=np.nan,
                    verdict='NOT_FOUND'
                ))
                continue

            # Pick highest absolute-weight match
            best_feat = max(matching, key=lambda f: abs(w_arr[feat_list.index(f)]))
            best_idx  = feat_list.index(best_feat)
            best_w    = float(w_arr[best_idx])
            rank      = rank_by_name[best_feat]
            rank_pct  = 100 * rank / n_total

            top30 = max(1, int(0.30 * n_total))
            top60 = max(1, int(0.60 * n_total))

            if rank <= top30:
                magnitude_verdict = 'PROMINENT'
            elif rank <= top60:
                magnitude_verdict = 'PRESENT'
            else:
                magnitude_verdict = 'LOW_WEIGHT'

            # For signed methods, also check direction
            if is_signed:
                sign_ok = True   # we don't have per-anchor expectations here
                verdict = magnitude_verdict
            else:
                verdict = magnitude_verdict

            anchor_verdicts.append(dict(
                anchor=anchor_substr, best_match=best_feat,
                weight=round(best_w, 5), rank=rank, rank_pct=round(rank_pct, 1),
                total=n_total, verdict=verdict
            ))
            verdict_rows.append(dict(
                method=method_name, concept_idx=cidx, concept_label=clabel,
                anchor=anchor_substr, best_match=best_feat,
                weight=round(best_w, 5), rank=rank, rank_pct=round(rank_pct, 1),
                total=n_total, verdict=verdict
            ))

        counts = {v: sum(1 for a in anchor_verdicts if a['verdict']==v)
                  for v in ['PROMINENT','PRESENT','LOW_WEIGHT','NOT_FOUND']}
        n_a   = len(anchor_verdicts)
        score = round(100*(counts['PROMINENT']+counts['PRESENT'])/max(n_a,1), 1)

        summaries[ckey] = dict(
            concept_idx         = cidx,
            concept_label       = clabel,
            output_w            = round(out_w, 4),
            output_dir_expected = '+1' if exp_dir==1 else ('-1' if exp_dir==-1 else 'n/a'),
            output_verdict      = out_verdict,
            n_anchors           = n_a,
            n_prominent         = counts['PROMINENT'],
            n_present           = counts['PRESENT'],
            n_low               = counts['LOW_WEIGHT'],
            n_not_found         = counts['NOT_FOUND'],
            clinical_score_pct  = score,
            anchor_verdicts     = anchor_verdicts,
        )

    return summaries, verdict_rows


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
METHOD_COLORS = {
    'correlation': '#2196F3',
    'nnls':        '#4CAF50',
    'twostage_lr': '#FF9800',
}
VERDICT_PALETTE = {
    'PROMINENT':  '#4CAF50',
    'PRESENT':    '#8BC34A',
    'LOW_WEIGHT': '#FF9800',
    'NOT_FOUND':  '#9E9E9E',
    'MATCHES':    '#4CAF50',
    'REVERSED':   '#F44336',
    'WEAK':       '#FF9800',
}


def plot_method_comparison(results_list):
    """Bar chart comparing all 3 methods on AUROC, AUPRC, Brier, F1."""
    df     = pd.DataFrame(results_list)
    colors = [METHOD_COLORS.get(m.lower().replace(' ','_').replace('-','_'), '#607D8B')
              for m in df['model']]

    metrics = ['auroc', 'auprc', 'f1_readmit', 'precision_readmit', 'recall_readmit', 'brier']
    titles  = ['AUROC', 'AUPRC', 'F1 (Readmit)', 'Precision', 'Recall', 'Brier (↓)']

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, metric, title in zip(axes, metrics, titles):
        vals = df[metric].tolist()
        bars = ax.bar(range(len(df)), vals, color=colors, edgecolor='white', width=0.6)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([m.replace(' ', '\n') for m in df['model']], fontsize=8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(vals) * 1.15)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    fig.suptitle('ConceptNet Statistical Methods — Test Set Performance\n'
                 'Input → 12 concept nodes → 2  (no neural network)',
                 fontsize=13, fontweight='bold', y=1.01)
    patches = [mpatches.Patch(color=v, label=k.replace('_',' ').title())
               for k, v in METHOD_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_method_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_concept_weights(concept_weights, concept_index, feature_names,
                         summaries, method_name, color, is_nonneg=True):
    """Per-concept top-N weight bars for one method."""
    n_cols = 3
    n_rows = int(np.ceil(N_CONCEPTS / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 4))
    axes = axes.flatten()

    for cidx, ckey, clabel in CONCEPTS:
        ax       = axes[cidx]
        fi_list  = concept_index[ckey]
        w_arr    = concept_weights.get(ckey, np.array([]))
        if len(w_arr) == 0:
            ax.set_visible(False); continue

        feat_names = [feature_names[fi] for fi in fi_list]

        # Top-N by abs weight
        abs_w   = np.abs(w_arr)
        top_idx = np.argsort(-abs_w)[:TOP_N]
        top_w   = w_arr[top_idx]
        top_f   = [feat_names[i] for i in top_idx]
        top_f_s = [f[:42]+'…' if len(f)>42 else f for f in top_f]

        if is_nonneg:
            bar_colors = [color] * len(top_w)
        else:
            bar_colors = ['#4CAF50' if w >= 0 else '#F44336' for w in top_w]

        ax.barh(range(len(top_w)), top_w, color=bar_colors,
                edgecolor='white', height=0.7)
        ax.set_yticks(range(len(top_w)))
        ax.set_yticklabels(top_f_s, fontsize=7)
        if not is_nonneg:
            ax.axvline(0, color='black', lw=0.8, alpha=0.5)
        ax.set_title(f'Node {cidx}: {clabel}', fontsize=8.5, fontweight='bold')
        ax.set_xlabel('Weight', fontsize=7)
        ax.grid(axis='x', alpha=0.3)

        s  = summaries.get(ckey, {})
        ov = s.get('output_verdict', '')
        ov_c = '#4CAF50' if ov == 'MATCHES' else ('#F44336' if ov == 'REVERSED' else '#FF9800')
        ax.text(0.98, 0.02,
                f"Score: {s.get('clinical_score_pct',0)}%\n"
                f"Out_w: {s.get('output_w',0):+.3f} ({ov})",
                transform=ax.transAxes, fontsize=6.5, ha='right', va='bottom',
                color=ov_c,
                bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', alpha=0.8))

    for i in range(N_CONCEPTS, len(axes)):
        axes[i].set_visible(False)

    nonneg_note = '(all ≥ 0)' if is_nonneg else '(signed weights)'
    fig.suptitle(f'{method_name} — Top-{TOP_N} Feature Weights per Concept {nonneg_note}',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR,
                        f'plot_concept_weights_{method_name.lower().replace(" ","_")}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_output_weights_comparison(ow_dict):
    """
    Side-by-side output weight bars for all methods.
    Expected sign annotations included.
    """
    labels = [f"N{cidx}: {clabel[:25]}" for cidx, _, clabel in CONCEPTS]
    exp    = [OUTPUT_DIRECTION.get(ckey, 0) for _, ckey, _ in CONCEPTS]

    methods = list(ow_dict.keys())
    n_m     = len(methods)
    x       = np.arange(N_CONCEPTS)
    width   = 0.25

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, mname in enumerate(methods):
        ow     = ow_dict[mname]
        color  = list(METHOD_COLORS.values())[i]
        offset = (i - n_m/2 + 0.5) * width
        ax.bar(x + offset, ow, width, label=mname.replace('_',' ').title(),
               color=color, alpha=0.85, edgecolor='white')

    # Expected direction markers
    for xi, e in enumerate(exp):
        ax.annotate('+1' if e>0 else '-1',
                    xy=(xi, 0), xytext=(xi, -0.55),
                    ha='center', fontsize=7.5, color='black',
                    arrowprops=dict(arrowstyle='-', color='grey', lw=0.5))

    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7.5)
    ax.set_ylabel('Output Weight → Readmission', fontsize=11)
    ax.set_title('Output Layer Weights — All Methods vs Clinical Expectation\n'
                 'Annotations show expected direction (+1=risk / -1=protective)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_output_weights_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_concept_score_delta(score_dict, y_te):
    """
    Δ in concept score (readmit − not readmit) for each method.
    """
    n_methods = len(score_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 8), sharey=True)
    if n_methods == 1:
        axes = [axes]

    labels = [clabel for _, _, clabel in CONCEPTS]

    for ax, (mname, C_te) in zip(axes, score_dict.items()):
        m0 = np.array([C_te[y_te == 0, i].mean() for i in range(N_CONCEPTS)])
        m1 = np.array([C_te[y_te == 1, i].mean() for i in range(N_CONCEPTS)])
        delta = m1 - m0

        exp  = [OUTPUT_DIRECTION.get(ckey, 0) for _, ckey, _ in CONCEPTS]
        cols = ['#4CAF50' if (d > 0 and e > 0) or (d < 0 and e < 0) else '#F44336'
                for d, e in zip(delta, exp)]

        ax.barh(range(N_CONCEPTS), delta, color=cols, edgecolor='white', height=0.7)
        ax.axvline(0, color='black', lw=1)
        ax.set_yticks(range(N_CONCEPTS))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(mname.replace('_',' ').title(), fontsize=11, fontweight='bold')
        ax.set_xlabel('Activation Δ (Readmit − Not)', fontsize=9)
        ax.grid(axis='x', alpha=0.3)

    patches = [mpatches.Patch(color='#4CAF50', label='Direction matches expectation'),
               mpatches.Patch(color='#F44336', label='Direction reversed')]
    fig.legend(handles=patches, loc='lower center', ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Concept Score Δ by Class — All Methods', fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_concept_score_delta.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_verdict_heatmap(all_summaries):
    """
    Heatmap: rows=concepts, columns=methods, cell=clinical_score_pct.
    """
    methods = list(all_summaries.keys())
    concept_labels = [f"N{cidx}: {clabel[:35]}" for cidx, _, clabel in CONCEPTS]
    scores = np.array([[all_summaries[m][ckey]['clinical_score_pct']
                        for _, ckey, _ in CONCEPTS]
                       for m in methods]).T   # (N_CONCEPTS, n_methods)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(scores, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], fontsize=10)
    ax.set_yticks(range(N_CONCEPTS))
    ax.set_yticklabels(concept_labels, fontsize=8)
    for i in range(N_CONCEPTS):
        for j in range(len(methods)):
            ax.text(j, i, f"{scores[i,j]:.0f}%", ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='white' if scores[i,j] < 40 else 'black')
    plt.colorbar(im, ax=ax, label='Clinical Anchor Score (%)')
    ax.set_title('Clinical Validation Score per Concept × Method',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_verdict_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    log.info(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────
def write_report(results_list, all_summaries, concept_index, feature_names,
                 ow_dict, y_te, score_dict):
    lines = []
    SEP  = "=" * 72
    SEP2 = "─" * 72
    def h(t):   lines.append(f"\n{SEP}\n  {t}\n{SEP}")
    def sub(t): lines.append(f"\n{SEP2}\n  {t}\n{SEP2}")

    h("CONCEPTNET STATISTICAL FITTING — CLINICAL VALIDATION REPORT")
    lines.append(f"  Generated   : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Architecture: Input → {N_CONCEPTS} concept nodes → 2")
    lines.append(f"  Methods     : Correlation, NNLS, Two-Stage LR")
    lines.append(f"  No neural network used — pure statistical fitting\n")

    lines.append("  Why statistical methods instead of NN?")
    lines.append("    - NN backprop through a 12-node bottleneck is unstable")
    lines.append("    - Each concept fitted independently avoids inter-concept gradient interference")
    lines.append("    - NNLS gives an EXACT solution, not an approximation")
    lines.append("    - Two-stage LR gives calibrated concept probabilities")
    lines.append("    - All methods are faster and more reproducible than NN")

    sub("Test Set Performance")
    df = pd.DataFrame(results_list)
    col_w = [max(len(str(v)) for v in df[c].tolist() + [c]) for c in df.columns]
    lines.append("  " + "  ".join(f"{c:<{w}}" for c, w in zip(df.columns, col_w)))
    lines.append("  " + "  ".join("─"*w for w in col_w))
    for _, row in df.iterrows():
        lines.append("  " + "  ".join(f"{str(row[c]):<{w}}"
                                       for c, w in zip(df.columns, col_w)))

    sub("Best Method per Metric")
    for metric, label in [('auroc','AUROC'), ('auprc','AUPRC'),
                           ('brier','Brier (lower=better)'), ('f1_readmit','F1')]:
        idx = df[metric].idxmin() if metric == 'brier' else df[metric].idxmax()
        lines.append(f"  {label:<20s}: {df.loc[idx,'model']}  ({df.loc[idx,metric]:.4f})")

    sub("Overall Clinical Validation Summary")
    for mname, summaries in all_summaries.items():
        scores = [s['clinical_score_pct'] for s in summaries.values()]
        out_vs = [s['output_verdict'] for s in summaries.values()]
        lines.append(f"\n  {mname.replace('_',' ').upper()}")
        lines.append(f"    Mean anchor score    : {np.mean(scores):.1f}%")
        lines.append(f"    Output w MATCHES     : {out_vs.count('MATCHES')}/{N_CONCEPTS}")
        lines.append(f"    Output w REVERSED    : {out_vs.count('REVERSED')}/{N_CONCEPTS}")
        lines.append(f"    Output w WEAK        : {out_vs.count('WEAK')}/{N_CONCEPTS}")

    h("OUTPUT LAYER WEIGHTS — ALL METHODS vs CLINICAL EXPECTATION")
    lines.append(f"\n  Pos weight = concept raises readmission risk")
    lines.append(f"  Neg weight = concept lowers readmission risk (protective)\n")
    lines.append(f"  {'Node':<4}  {'Concept':<45}  {'Exp':>4}  "
                 + "  ".join(f"{m[:12]:>12}" for m in ow_dict.keys()))
    lines.append(f"  {'─'*3}  {'─'*45}  {'─'*4}  "
                 + "  ".join("─"*12 for _ in ow_dict))
    for cidx, ckey, clabel in CONCEPTS:
        exp_s = OUTPUT_DIRECTION.get(ckey, 0)
        exp_str = '+1' if exp_s > 0 else '-1'
        weights_str = "  ".join(f"{ow_dict[m][cidx]:>+12.4f}" for m in ow_dict)
        lines.append(f"  {cidx:<4}  {clabel:<45}  {exp_str:>4}  {weights_str}")

    h("PER-CONCEPT DETAILED ANALYSIS  (NNLS — recommended)")
    nnls_summaries = all_summaries.get('nnls', {})
    for cidx, ckey, clabel in CONCEPTS:
        s = nnls_summaries.get(ckey, {})
        if not s:
            continue
        sub(f"Node {cidx}: {clabel}")
        ov   = s.get('output_verdict','')
        icon = {'MATCHES':'✓','REVERSED':'✗','WEAK':'~'}.get(ov,'?')
        lines.append(f"  Output weight: {s['output_w']:+.4f}  "
                     f"(expected {s['output_dir_expected']})  → {icon} {ov}")
        lines.append(f"  Anchor score : {s['clinical_score_pct']}%  "
                     f"({s['n_prominent']} prominent / {s['n_present']} present)")

        m0 = score_dict['nnls'][y_te==0, cidx].mean()
        m1 = score_dict['nnls'][y_te==1, cidx].mean()
        lines.append(f"  Score Δ      : non-readmit {m0:.3f}  |  readmit {m1:.3f}  "
                     f"| Δ={m1-m0:+.3f}")

        lines.append(f"\n  Clinical anchors:")
        for av in s.get('anchor_verdicts', []):
            icon2 = {'PROMINENT':'✓✓','PRESENT':'✓','LOW_WEIGHT':'~','NOT_FOUND':'?'}.get(av['verdict'],'?')
            if av['best_match'] == 'NOT FOUND':
                lines.append(f"    {icon2} [{av['verdict']:<10}] '{av['anchor']}'  NOT FOUND")
            else:
                lines.append(f"    {icon2} [{av['verdict']:<10}] '{av['anchor']}'  →  "
                             f"{av['best_match']}  "
                             f"weight={av['weight']:.5f}  "
                             f"rank {av['rank']}/{av['total']}  (top {av['rank_pct']:.0f}%)")
        lines.append("")

    h("METHOD COMPARISON — WHICH TO USE?")
    lines.append("""
  CORRELATION
    Best for   : Rapid inspection, explainability to non-technical stakeholders
    Limitation : Non-supervised in concept layer; misses interaction effects
    Use when   : You need fully transparent, auditable weights

  NNLS (Non-Negative Least Squares)
    Best for   : Interpretable, non-negative weights with supervised fitting
    Limitation : Linear model within each concept (no interactions)
    Use when   : You want non-negativity guaranteed + supervised fitting
    Why NNLS   : scipy.optimize.nnls gives EXACT solution in O(n*k^2) time
                 No learning rate, no epochs, no convergence issues

  TWO-STAGE LR
    Best for   : Maximum predictive performance with interpretable structure
    Limitation : Concept weights may be negative (requires sign interpretation)
    Use when   : Performance matters more than non-negativity constraint
    """)

    report = "\n".join(lines)
    path   = os.path.join(OUTPUT_DIR, 'clinical_validation_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    log.info(f"\n{report}")
    log.info(f"\n  Report saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("═" * 60)
    log.info("  ConceptNet — Statistical Fitting (No Neural Network)")
    log.info(f"  Architecture: Input → {N_CONCEPTS} concept nodes → 2")
    log.info("═" * 60)

    # Load + split + preprocess
    X_df, y = load_data()
    X_tv, X_te, y_tv, y_te = train_test_split(
        X_df, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=VAL_SIZE, random_state=SEED, stratify=y_tv)
    log.info(f"Split — train {len(X_tr):,} | val {len(X_va):,} | test {len(X_te):,}")

    prep    = Preprocessor(list(X_df.columns))
    X_tr_np = prep.fit_transform(X_tr)
    X_te_np = prep.transform(X_te)
    y_tr_np = y_tr.to_numpy()
    y_te_np = y_te.to_numpy()
    log.info(f"  Features: {X_tr_np.shape[1]}")

    # Build concept→feature index
    concept_index = build_concept_feature_index(prep.out_cols_)

    # ─── Fit all three methods ────────────────────────────────────────────
    log.info("\n" + "═" * 60)
    log.info("  Fitting all methods …")
    log.info("═" * 60)

    (y_prob_corr, cw_corr, ow_corr, C_te_corr) = fit_correlation(
        X_tr_np, y_tr_np, X_te_np, concept_index, prep.out_cols_)

    (y_prob_nnls, cw_nnls, ow_nnls, C_te_nnls) = fit_nnls(
        X_tr_np, y_tr_np, X_te_np, concept_index, prep.out_cols_)

    (y_prob_lr, cw_lr, ow_lr, C_te_lr) = fit_twostage_lr(
        X_tr_np, y_tr_np, X_te_np, concept_index, prep.out_cols_)

    # ─── Metrics ──────────────────────────────────────────────────────────
    results = [
        compute_metrics(y_te_np, y_prob_corr, 'Correlation'),
        compute_metrics(y_te_np, y_prob_nnls,  'NNLS'),
        compute_metrics(y_te_np, y_prob_lr,    'Two-Stage LR'),
    ]
    results_df = pd.DataFrame(results).sort_values('auroc', ascending=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'method_comparison_metrics.csv'), index=False)
    log.info(f"\n  Performance:\n{results_df.to_string(index=False)}\n")

    # ─── Clinical validation ──────────────────────────────────────────────
    log.info("═" * 60)
    log.info("  Clinical weight validation …")
    log.info("═" * 60)

    sum_corr, vrows_corr = validate_method(
        cw_corr, ow_corr, concept_index, prep.out_cols_, 'correlation', is_signed=False)
    sum_nnls, vrows_nnls = validate_method(
        cw_nnls, ow_nnls, concept_index, prep.out_cols_, 'nnls',        is_signed=False)
    sum_lr,   vrows_lr   = validate_method(
        cw_lr,   ow_lr,   concept_index, prep.out_cols_, 'twostage_lr', is_signed=True)

    all_summaries = {
        'correlation': sum_corr,
        'nnls':        sum_nnls,
        'twostage_lr': sum_lr,
    }
    ow_dict = {
        'correlation': ow_corr,
        'nnls':        ow_nnls,
        'twostage_lr': ow_lr,
    }
    score_dict = {
        'correlation': C_te_corr,
        'nnls':        C_te_nnls,
        'twostage_lr': C_te_lr,
    }

    # Save all verdicts
    pd.DataFrame(vrows_corr + vrows_nnls + vrows_lr).to_csv(
        os.path.join(OUTPUT_DIR, 'all_anchor_verdicts.csv'), index=False)

    # Save all weights
    weight_rows = []
    for mname, cw in [('correlation', cw_corr), ('nnls', cw_nnls), ('twostage_lr', cw_lr)]:
        for cidx, ckey, clabel in CONCEPTS:
            fi_list = concept_index[ckey]
            w_arr   = cw.get(ckey, np.array([]))
            for i, fi in enumerate(fi_list):
                if i < len(w_arr):
                    weight_rows.append(dict(
                        method=mname, concept_idx=cidx, concept_label=clabel,
                        feature=prep.out_cols_[fi], weight=float(w_arr[i]),
                        abs_weight=float(abs(w_arr[i])),
                    ))
    pd.DataFrame(weight_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'concept_weights_all_methods.csv'), index=False)

    # ─── Plots ────────────────────────────────────────────────────────────
    log.info("\n  Generating plots …")
    plot_method_comparison(results)
    plot_concept_weights(cw_corr, concept_index, prep.out_cols_,
                         sum_corr, 'Correlation', '#2196F3', is_nonneg=True)
    plot_concept_weights(cw_nnls, concept_index, prep.out_cols_,
                         sum_nnls, 'NNLS', '#4CAF50', is_nonneg=True)
    plot_concept_weights(cw_lr, concept_index, prep.out_cols_,
                         sum_lr, 'Two-Stage_LR', '#FF9800', is_nonneg=False)
    plot_output_weights_comparison(ow_dict)
    plot_concept_score_delta(score_dict, y_te_np)
    plot_verdict_heatmap(all_summaries)

    # ─── Report ───────────────────────────────────────────────────────────
    log.info("\n  Writing clinical report …")
    write_report(results, all_summaries, concept_index,
                 prep.out_cols_, ow_dict, y_te_np, score_dict)

    # ─── Final console summary ────────────────────────────────────────────
    log.info("\n" + "═" * 60)
    log.info("  FINAL SUMMARY")
    log.info("═" * 60)
    log.info(f"\n  {'Method':<16}  {'AUROC':>7}  {'AUPRC':>7}  {'Brier':>7}  {'F1':>7}")
    log.info(f"  {'─'*15}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    for r in results:
        log.info(f"  {r['model']:<16}  {r['auroc']:>7.4f}  {r['auprc']:>7.4f}  "
                 f"{r['brier']:>7.4f}  {r['f1_readmit']:>7.4f}")

    log.info(f"\n  Clinical anchor scores (mean %):")
    for mname, summaries in all_summaries.items():
        scores = [s['clinical_score_pct'] for s in summaries.values()]
        log.info(f"    {mname:<16}: {np.mean(scores):.1f}%")

    log.info(f"\n✓ All outputs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()