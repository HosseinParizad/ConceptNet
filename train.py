"""
Full Comparison — 30-Day Readmission Prediction
================================================
Runs ALL models on identical data splits and produces:
  • Unified results_summary.csv / .json
  • ROC curve comparison plot
  • PR curve comparison plot
  • Calibration curve plot
  • Metric bar-chart plot
  • Per-model prediction CSVs
  • ConceptNet concept-activation analysis
  • Full text report (report.txt)

Models
──────
  1. Logistic Regression
  2. Random Forest
  3. XGBoost
  4. LightGBM
  5. Standard MLP          (fully-connected NN)
  6. ConceptNet            (sparse concept-driven NN  Input→12→2)

All models use the SAME train / val / test split (SEED=42).
No model architecture, hyper-parameter, or input feature is changed.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0) IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, re, sys, json, time, warnings, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from textwrap import indent

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler, OrdinalEncoder
from sklearn.impute          import SimpleImputer
from sklearn.calibration     import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (roc_auc_score, average_precision_score,
                                     brier_score_loss, classification_report,
                                     roc_curve, precision_recall_curve)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1) CONFIG  (unchanged from both originals)
# ─────────────────────────────────────────────────────────────────────────────
FEATURES_CSV = './model_outputs/extracted_features2.csv'
OUTPUT_DIR   = './model_outputs/full_comparison/'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

SEED         = 42
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15
READMIT_DAYS = 30
BATCH_SIZE   = 2048
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MLP / ConceptNet training
MLP_MAX_EPOCHS  = 200
MLP_PATIENCE    = 20
CN_MAX_EPOCHS   = 300
CN_PATIENCE     = 25
LR              = 1e-3
WEIGHT_DECAY    = 1e-4

np.random.seed(SEED)
torch.manual_seed(SEED)

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
# 2) CONCEPTNET ARCHITECTURE DEFINITIONS
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
        r'^lab_bicarbonate', r'^lab_albumin',
        r'^lab_hemoglobin', r'^lab_sodium.*(bw|ws)',
        r'^lab_potassium.*(bw|ws)', r'^lab_chloride.*(bw|ws)',
        r'^lab_lactate.*(bw|ws)', r'^lab_wbc.*(bw|ws)',
        r'^lab_platelets.*(bw|ws)', r'^lab_glucose.*(bw|ws)',
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
        r'^vital_gcs_(last|min|max|std)_pdw$',
        r'^vital_measure_count_pdw$',
        r'^lab_(creatinine|sodium|potassium|glucose|wbc)_(last|min|max|mean)_pdw$',
        r'^lab_abnormal_count_pdw$',
    ],
    'disease_resolution': [
        r'^note_resolution_score$', r'^note_stable_at_discharge$',
        r'^lab_.*_slope_pdw$', r'^lab_.*_delta_last_first_pdw$',
        r'^vital_.*_slope_pdw$',
        r'^aki_creat_ratio', r'^lab_lactate_(last|min|max)_pdw$',
        r'^lab_bilirubin_(last|min|max)_pdw$',
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


# ─────────────────────────────────────────────────────────────────────────────
# 3) DATA LOADING & LABEL
# ─────────────────────────────────────────────────────────────────────────────
def create_readmission_label(df, days=30):
    df = df.sort_values(['subject_id', 'admittime']).copy()
    df['next_admittime'] = df.groupby('subject_id')['admittime'].shift(-1)
    df['days_to_readmit'] = (
        (df['next_admittime'] - df['dischtime']).dt.total_seconds() / 86400.0
    )
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
    log.info(f"  {READMIT_DAYS}-day readmissions: {int(y.sum()):,}/{len(y):,} "
             f"({100*y.mean():.1f}%)")

    drop = [c for c in df.columns if c in EXCLUDE_COLS or c == 'next_admittime']
    X = df.drop(columns=[c for c in drop if c in df.columns], errors='ignore')
    X = X.select_dtypes(exclude=['datetime64[ns]'])
    log.info(f"  Feature columns: {X.shape[1]}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 4) PREPROCESSOR  (shared by all models)
# ─────────────────────────────────────────────────────────────────────────────
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
            # Categorical imputer with constant strategy keeps all columns
            cols += self.cat_cols
        if self.num_cols:
            Xn = X[self.num_cols].apply(pd.to_numeric, errors='coerce')
            parts.append(self.num_scaler.fit_transform(
                self.num_imp.fit_transform(Xn)).astype(np.float32))
            # Numeric imputer may drop all-NaN columns - use get_feature_names_out to track
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
# 5) SHARED METRICS & CURVE SAVING
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_prob, model_name, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auroc  = roc_auc_score(y_true, y_prob)
    auprc  = average_precision_score(y_true, y_prob)
    brier  = brier_score_loss(y_true, y_prob)
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0)
    return dict(
        model             = model_name,
        auroc             = round(float(auroc), 4),
        auprc             = round(float(auprc), 4),
        brier             = round(float(brier), 4),
        f1_readmit        = round(report['Readmitted']['f1-score'], 4),
        precision_readmit = round(report['Readmitted']['precision'], 4),
        recall_readmit    = round(report['Readmitted']['recall'], 4),
    )


def save_curves(model_key, y_true, y_prob):
    """Save ROC and PR curve data, and return arrays for combined plots."""
    fpr, tpr, _   = roc_curve(y_true, y_prob)
    prec, rec, _  = precision_recall_curve(y_true, y_prob)
    frac_pos, mps = calibration_curve(y_true, y_prob, n_bins=10)

    pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(
        os.path.join(OUTPUT_DIR, f'{model_key}_roc.csv'), index=False)
    pd.DataFrame({'precision': prec, 'recall': rec}).to_csv(
        os.path.join(OUTPUT_DIR, f'{model_key}_pr.csv'), index=False)
    pd.DataFrame({'mean_pred': mps, 'frac_pos': frac_pos}).to_csv(
        os.path.join(OUTPUT_DIR, f'{model_key}_cal.csv'), index=False)

    return fpr, tpr, prec, rec, mps, frac_pos


def save_preds(model_key, y_true, y_prob):
    pd.DataFrame({'y_true': y_true, 'y_prob': y_prob}).to_csv(
        os.path.join(OUTPUT_DIR, f'{model_key}_preds.csv'), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 6) BASELINE MODELS  (unchanged from baselines script)
# ─────────────────────────────────────────────────────────────────────────────

# ── 6a. Logistic Regression ──────────────────────────────────────────────────
def run_logistic_regression(X_tr, y_tr, X_te, y_te):
    log.info("─" * 55)
    log.info("Model 1 — Logistic Regression")
    t0 = time.time()

    model = LogisticRegression(C=0.1, penalty='l2', solver='saga',
                               max_iter=1000, class_weight='balanced',
                               random_state=SEED, n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1]

    metrics = compute_metrics(y_te, y_prob, 'Logistic Regression')
    metrics['train_time_s'] = round(time.time() - t0, 1)
    log.info(f"  AUROC {metrics['auroc']:.4f} | AUPRC {metrics['auprc']:.4f} "
             f"| Brier {metrics['brier']:.4f}")
    curves = save_curves('logistic_regression', y_te, y_prob)
    save_preds('logistic_regression', y_te, y_prob)
    return metrics, y_prob, curves


# ── 6b. Random Forest ────────────────────────────────────────────────────────
def run_random_forest(X_tr, y_tr, X_te, y_te):
    log.info("─" * 55)
    log.info("Model 2 — Random Forest")
    t0    = time.time()
    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos

    model = RandomForestClassifier(
        n_estimators=500, max_depth=12, min_samples_leaf=50,
        max_features='sqrt', class_weight={0: 1, 1: n_neg / max(n_pos, 1)},
        random_state=SEED, n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1]

    metrics = compute_metrics(y_te, y_prob, 'Random Forest')
    metrics['train_time_s'] = round(time.time() - t0, 1)
    log.info(f"  AUROC {metrics['auroc']:.4f} | AUPRC {metrics['auprc']:.4f} "
             f"| Brier {metrics['brier']:.4f}")
    curves = save_curves('random_forest', y_te, y_prob)
    save_preds('random_forest', y_te, y_prob)
    return metrics, y_prob, curves


# ── 6c. XGBoost ──────────────────────────────────────────────────────────────
def run_xgboost(X_tr, y_tr, X_va, y_va, X_te, y_te):
    log.info("─" * 55)
    log.info("Model 3 — XGBoost")
    if not HAS_XGB:
        log.warning("  xgboost not installed — skipped")
        return None, None, None

    t0    = time.time()
    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos

    model = xgb.XGBClassifier(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=30,
        scale_pos_weight=n_neg / max(n_pos, 1),
        eval_metric='aucpr', early_stopping_rounds=30,
        use_label_encoder=False, random_state=SEED, n_jobs=-1, verbosity=0)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    y_prob = model.predict_proba(X_te)[:, 1]

    metrics = compute_metrics(y_te, y_prob, 'XGBoost')
    metrics['train_time_s'] = round(time.time() - t0, 1)
    log.info(f"  AUROC {metrics['auroc']:.4f} | AUPRC {metrics['auprc']:.4f} "
             f"| Brier {metrics['brier']:.4f}")
    curves = save_curves('xgboost', y_te, y_prob)
    save_preds('xgboost', y_te, y_prob)
    return metrics, y_prob, curves


# ── 6d. LightGBM ─────────────────────────────────────────────────────────────
def run_lightgbm(X_tr, y_tr, X_va, y_va, X_te, y_te):
    log.info("─" * 55)
    log.info("Model 4 — LightGBM")
    if not HAS_LGB:
        log.warning("  lightgbm not installed — skipped")
        return None, None, None

    t0    = time.time()
    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos

    model = lgb.LGBMClassifier(
        n_estimators=1000, max_depth=7, num_leaves=63,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=50, scale_pos_weight=n_neg / max(n_pos, 1),
        metric='average_precision', random_state=SEED, n_jobs=-1, verbose=-1)
    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False),
                 lgb.log_evaluation(period=-1)]
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=callbacks)
    y_prob = model.predict_proba(X_te)[:, 1]

    metrics = compute_metrics(y_te, y_prob, 'LightGBM')
    metrics['train_time_s'] = round(time.time() - t0, 1)
    log.info(f"  AUROC {metrics['auroc']:.4f} | AUPRC {metrics['auprc']:.4f} "
             f"| Brier {metrics['brier']:.4f}")
    curves = save_curves('lightgbm', y_te, y_prob)
    save_preds('lightgbm', y_te, y_prob)
    return metrics, y_prob, curves


# ── 6e. Standard MLP ─────────────────────────────────────────────────────────
class StandardMLP(nn.Module):
    def __init__(self, n_features, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),                              nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 2),
        )
    def forward(self, x): return self.net(x)


def _build_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te):
    def tt(X, y):
        return torch.from_numpy(X).float(), torch.from_numpy(y.astype(np.int64))
    Xt, yt = tt(X_tr, y_tr); Xv, yv = tt(X_va, y_va); Xe, ye = tt(X_te, y_te)
    n_pos  = int((yt == 1).sum()); n_neg = len(yt) - n_pos
    w      = torch.where(yt == 1,
                         torch.full_like(yt, float(n_neg)/max(n_pos,1), dtype=torch.float),
                         torch.ones(len(yt), dtype=torch.float))
    sampler  = WeightedRandomSampler(w, num_samples=len(yt), replacement=True)
    train_dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_dl   = DataLoader(TensorDataset(Xv, yv), batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)
    test_dl  = DataLoader(TensorDataset(Xe, ye), batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)
    return train_dl, val_dl, test_dl


@torch.no_grad()
def _eval_nn(model, loader, crit):
    model.eval()
    probs, labels, total_loss = [], [], 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        logits  = model(Xb)
        total_loss += crit(logits, yb).item() * len(yb)
        p = torch.softmax(logits, dim=1)[:, 1]
        probs.extend(p.cpu().tolist()); labels.extend(yb.cpu().tolist())
    return total_loss/len(labels), np.array(probs), np.array(labels)


def run_mlp(X_tr, y_tr, X_va, y_va, X_te, y_te):
    log.info("─" * 55)
    log.info("Model 5 — Standard MLP  (Input→256→128→64→2)")
    t0    = time.time()
    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
    pw    = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=DEVICE)
    crit  = nn.CrossEntropyLoss(weight=pw)

    model = StandardMLP(X_tr.shape[1]).to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=8)
    train_dl, val_dl, test_dl = _build_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te)

    best_auroc, best_state, no_imp = 0.0, None, 0
    history = []
    for ep in range(1, MLP_MAX_EPOCHS + 1):
        model.train(); ep_loss = 0.0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(Xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ep_loss += loss.item() * len(yb)
        ep_loss /= len(train_dl.dataset)
        _, vp, vl = _eval_nn(model, val_dl, crit)
        val_auroc  = roc_auc_score(vl, vp)
        sched.step(val_auroc)
        history.append(dict(epoch=ep, train_loss=ep_loss, val_auroc=val_auroc))
        if ep == 1 or ep % 20 == 0:
            log.info(f"  Ep {ep:3d} | loss {ep_loss:.4f} | val AUROC {val_auroc:.4f}")
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= MLP_PATIENCE:
                log.info(f"  Early stop at epoch {ep}"); break

    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, 'mlp_training_history.csv'), index=False)

    _, y_prob, y_true = _eval_nn(model, test_dl, crit)
    metrics = compute_metrics(y_true, y_prob, 'Standard MLP')
    metrics['train_time_s'] = round(time.time() - t0, 1)
    log.info(f"  AUROC {metrics['auroc']:.4f} | AUPRC {metrics['auprc']:.4f} "
             f"| Brier {metrics['brier']:.4f}")
    curves = save_curves('mlp', y_true, y_prob)
    save_preds('mlp', y_true, y_prob)
    return metrics, y_prob, curves


# ─────────────────────────────────────────────────────────────────────────────
# 7) CONCEPTNET MODEL  (unchanged from conceptnet_readmission.py)
# ─────────────────────────────────────────────────────────────────────────────
def build_connection_mask(feature_names):
    n_feat = len(feature_names)
    mask   = torch.zeros(N_CONCEPTS, n_feat, dtype=torch.bool)
    for cidx, ckey, _ in CONCEPTS:
        for fi, fname in enumerate(feature_names):
            for pat in CONCEPT_PATTERNS.get(ckey, []):
                if re.search(pat, fname, re.IGNORECASE):
                    mask[cidx, fi] = True; break
    unassigned = (~mask.any(dim=0)).nonzero(as_tuple=True)[0]
    if len(unassigned):
        mask[4, unassigned] = True
        log.info(f"  {len(unassigned)} unassigned features → concept 4 (admission severity)")
    log.info("Connection mask [concept nodes × input features]:")
    for cidx, _, label in CONCEPTS:
        log.info(f"  Node {cidx:2d}  {label:<50s} ← {int(mask[cidx].sum()):3d} inputs")
    return mask


class SparseConceptLayer(nn.Module):
    def __init__(self, n_features, mask):
        super().__init__()
        self.register_buffer('mask', mask)
        self.weight = nn.Parameter(torch.empty(N_CONCEPTS, n_features))
        self.bias   = nn.Parameter(torch.zeros(N_CONCEPTS))
        nn.init.kaiming_uniform_(self.weight, a=0, nonlinearity='sigmoid')
        with torch.no_grad():
            self.weight.data[~mask] = 0.0
        self.weight.register_hook(self._zero_masked_grad)

    def _zero_masked_grad(self, grad):
        return grad * self.mask.float()

    def forward(self, x):
        w = self.weight * self.mask.float()
        return torch.sigmoid(x @ w.t() + self.bias)


class ConceptNet(nn.Module):
    def __init__(self, n_features, mask):
        super().__init__()
        self.concept_layer = SparseConceptLayer(n_features, mask)
        self.output_layer  = nn.Linear(N_CONCEPTS, 2)

    def forward(self, x, return_concepts=False):
        concepts = self.concept_layer(x)
        logits   = self.output_layer(concepts)
        if return_concepts:
            return logits, concepts
        return logits


def run_conceptnet(X_tr, y_tr, X_va, y_va, X_te, y_te, feature_names):
    log.info("─" * 55)
    log.info("Model 7 — ConceptNet  (Input→12 sparse concepts→2)")
    t0 = time.time()

    mask  = build_connection_mask(feature_names)
    model = ConceptNet(X_tr.shape[1], mask).to(DEVICE)

    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
    pw    = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=DEVICE)
    crit  = nn.CrossEntropyLoss(weight=pw)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10)

    train_dl, val_dl, test_dl = _build_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te)

    best_auroc, best_state, no_imp = 0.0, None, 0
    history = []
    log.info("  Training …")
    for ep in range(1, CN_MAX_EPOCHS + 1):
        model.train(); ep_loss = 0.0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(Xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ep_loss += loss.item() * len(yb)
        ep_loss /= len(train_dl.dataset)
        _, vp, vl = _eval_nn(model, val_dl, crit)
        val_auroc  = roc_auc_score(vl, vp)
        val_auprc  = average_precision_score(vl, vp)
        sched.step(val_auroc)
        history.append(dict(epoch=ep, train_loss=ep_loss, val_auroc=val_auroc, val_auprc=val_auprc))
        if ep == 1 or ep % 20 == 0:
            log.info(f"  Ep {ep:3d} | loss {ep_loss:.4f} | val AUROC {val_auroc:.4f} | AUPRC {val_auprc:.4f}")
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= CN_PATIENCE:
                log.info(f"  Early stop at epoch {ep}  (best AUROC {best_auroc:.4f})"); break

    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, 'conceptnet_training_history.csv'), index=False)

    _, y_prob, y_true = _eval_nn(model, test_dl, crit)
    metrics = compute_metrics(y_true, y_prob, 'ConceptNet')
    metrics['train_time_s'] = round(time.time() - t0, 1)
    log.info(f"  AUROC {metrics['auroc']:.4f} | AUPRC {metrics['auprc']:.4f} "
             f"| Brier {metrics['brier']:.4f}")
    curves = save_curves('conceptnet', y_true, y_prob)
    save_preds('conceptnet', y_true, y_prob)

    # ── Concept activations ────────────────────────────────────────────────
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(0, len(X_te), 4096):
            Xb = torch.from_numpy(X_te[i:i+4096]).float().to(DEVICE)
            _, c = model(Xb, return_concepts=True)
            rows.append(c.cpu().numpy())
    acts   = np.vstack(rows)
    act_df = pd.DataFrame(acts, columns=[lbl for _, _, lbl in CONCEPTS])
    act_df.insert(0, 'y_true', y_true.astype(int))
    act_df.insert(1, 'p_readmit_30d', y_prob)
    act_df.to_csv(os.path.join(OUTPUT_DIR, 'conceptnet_concept_activations_test.csv'), index=False)

    log.info("\n  Concept activation — mean by class:")
    log.info(f"  {'Concept':<50s}  {'Non-Readmit':>11}  {'Readmit':>9}  {'Δ':>7}")
    for _, _, label in CONCEPTS:
        m0 = act_df.loc[act_df['y_true'] == 0, label].mean()
        m1 = act_df.loc[act_df['y_true'] == 1, label].mean()
        log.info(f"  {label:<50s}  {m0:11.3f}  {m1:9.3f}  {m1-m0:+7.3f}")

    # Output layer weights
    ow = model.output_layer.weight.detach().cpu().numpy()
    log.info("\n  Output layer weights (concept → readmit):")
    for cidx, _, label in CONCEPTS:
        log.info(f"  {label:<50s}  w_readmit={ow[1, cidx]:+.4f}")

    # Save mapping
    mapping = {label: [feature_names[i]
                        for i in mask[cidx].nonzero(as_tuple=True)[0].tolist()]
               for cidx, _, label in CONCEPTS}
    with open(os.path.join(OUTPUT_DIR, 'conceptnet_feature_mapping.json'), 'w') as f:
        json.dump(mapping, f, indent=2)

    torch.save(dict(model_state_dict=model.state_dict(), mask=mask,
                    feature_names=feature_names, n_features=X_tr.shape[1],
                    concepts=CONCEPTS,
                    metrics=dict(auroc=metrics['auroc'], auprc=metrics['auprc'],
                                 brier=metrics['brier'])),
               os.path.join(OUTPUT_DIR, 'conceptnet_readmission.pt'))

    return metrics, y_prob, curves, model, mask, act_df


# ─────────────────────────────────────────────────────────────────────────────
# 8) COMPARISON PLOTS
# ─────────────────────────────────────────────────────────────────────────────
# Colour palette — one colour per model
PALETTE = {
    'Logistic Regression': '#2196F3',
    'Random Forest':       '#4CAF50',
    'XGBoost':             '#FF9800',
    'LightGBM':            '#9C27B0',
    'Standard MLP':        '#F44336',
    'ConceptNet':          '#E91E63',
}

def _model_color(name):
    for k, v in PALETTE.items():
        if k.lower() in name.lower():
            return v
    return '#607D8B'


def plot_roc(curve_data):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Chance')
    for name, (fpr, tpr, _, _, _, _), auroc in curve_data:
        ax.plot(fpr, tpr, lw=2, color=_model_color(name),
                label=f'{name}  (AUC={auroc:.3f})')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — 30-Day Readmission', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_roc_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_pr(curve_data, base_rate):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axhline(base_rate, color='k', lw=1, ls='--', alpha=0.4,
               label=f'Chance (prevalence={base_rate:.3f})')
    for name, (_, _, prec, rec, _, _), auprc in curve_data:
        ax.plot(rec, prec, lw=2, color=_model_color(name),
                label=f'{name}  (AP={auprc:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision–Recall Curves — 30-Day Readmission', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_pr_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_calibration(curve_data):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Perfect calibration')
    for name, (_, _, _, _, mps, frac_pos), _ in curve_data:
        ax.plot(mps, frac_pos, 'o-', lw=2, ms=5, color=_model_color(name), label=name)
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curves — 30-Day Readmission', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_calibration_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_metric_bars(results_df):
    metrics   = ['auroc', 'auprc', 'f1_readmit', 'precision_readmit', 'recall_readmit']
    labels    = ['AUROC', 'AUPRC (Avg Prec)', 'F1 (Readmit)', 'Precision (Readmit)', 'Recall (Readmit)']
    models    = results_df['model'].tolist()
    colors    = [_model_color(m) for m in models]

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    for ax, metric, label in zip(axes, metrics, labels):
        vals = results_df[metric].tolist()
        bars = ax.bar(range(len(models)), vals, color=colors, edgecolor='white', width=0.6)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=7.5)
        ax.set_ylim(0, 1)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    # Highlight ConceptNet bars
    fig.suptitle('Model Comparison — 30-Day Readmission Prediction',
                 fontsize=14, fontweight='bold', y=1.02)
    legend_patches = [mpatches.Patch(color=_model_color(m), label=m) for m in models]
    fig.legend(handles=legend_patches, loc='lower center', ncol=len(models),
               fontsize=9, bbox_to_anchor=(0.5, -0.12), framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_metric_bars.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_brier_train_time(results_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    models = results_df['model'].tolist()
    colors = [_model_color(m) for m in models]

    # Brier scores (lower is better)
    vals = results_df['brier'].tolist()
    bars = ax1.bar(range(len(models)), vals, color=colors, edgecolor='white', width=0.6)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=8)
    ax1.set_title('Brier Score  (lower = better)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Training time
    if 'train_time_s' in results_df.columns:
        vals2 = results_df['train_time_s'].tolist()
        bars2 = ax2.bar(range(len(models)), vals2, color=colors, edgecolor='white', width=0.6)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=8)
        ax2.set_title('Training Time (seconds)', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars2, vals2):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals2)*0.01,
                     f'{val:.0f}s', ha='center', va='bottom', fontsize=8, fontweight='bold')

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_brier_traintime.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_conceptnet_concepts(act_df):
    """Concept activation by class — horizontal grouped bars."""
    labels = [lbl for _, _, lbl in CONCEPTS]
    m0 = [act_df.loc[act_df['y_true'] == 0, l].mean() for l in labels]
    m1 = [act_df.loc[act_df['y_true'] == 1, l].mean() for l in labels]

    y    = np.arange(len(labels))
    h    = 0.35
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y + h/2, m0, h, label='Not Readmitted', color='#2196F3', alpha=0.85)
    ax.barh(y - h/2, m1, h, label='Readmitted',     color='#E91E63', alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Mean Concept Node Activation (sigmoid, 0–1)', fontsize=11)
    ax.set_title('ConceptNet — Concept Activations by Class\n(Test Set)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_conceptnet_concept_activations.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_nn_training_histories():
    """Overlay MLP and ConceptNet val AUROC training curves."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for fname, label, color in [
        ('mlp_training_history.csv',        'Standard MLP', '#F44336'),
        ('conceptnet_training_history.csv',  'ConceptNet',  '#E91E63'),
    ]:
        p = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(p):
            h = pd.read_csv(p)
            ax.plot(h['epoch'], h['val_auroc'], lw=2, color=color, label=label)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation AUROC', fontsize=12)
    ax.set_title('Training Curves — MLP vs ConceptNet', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_nn_training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9) TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────
def write_report(results_df, dataset_info, feature_names, act_df, cn_model):
    lines = []
    SEP   = "=" * 70

    def h(title): lines.append(f"\n{SEP}\n  {title}\n{SEP}")
    def sub(t):   lines.append(f"\n{'─' * 50}\n  {t}\n{'─' * 50}")

    h("30-DAY HOSPITAL READMISSION — FULL MODEL COMPARISON REPORT")
    lines.append(f"\n  Generated : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Data file : {FEATURES_CSV}")
    lines.append(f"  Seed      : {SEED}   Test size: {TEST_SIZE}   Val size: {VAL_SIZE}")

    sub("Dataset Summary")
    for k, v in dataset_info.items():
        lines.append(f"  {k:<35s}: {v}")

    sub("Model Performance (sorted by AUROC)")
    col_w = [max(len(str(v)) for v in results_df[c].tolist() + [c]) for c in results_df.columns]
    header = "  " + "  ".join(f"{c:<{w}}" for c, w in zip(results_df.columns, col_w))
    lines.append(header)
    lines.append("  " + "  ".join("─" * w for w in col_w))
    for _, row in results_df.iterrows():
        lines.append("  " + "  ".join(f"{str(row[c]):<{w}}" for c, w in zip(results_df.columns, col_w)))

    # Best model per metric
    sub("Best Model per Metric")
    for metric, label in [('auroc','AUROC'), ('auprc','AUPRC'),
                           ('brier','Brier (lower=better)'),
                           ('f1_readmit','F1 Readmit')]:
        if metric in results_df.columns:
            if metric == 'brier':
                idx = results_df[metric].idxmin()
            else:
                idx = results_df[metric].idxmax()
            row = results_df.loc[idx]
            lines.append(f"  {label:<25s}: {row['model']}  ({row[metric]:.4f})")

    # ConceptNet vs best baseline
    sub("ConceptNet vs Best Baseline (AUROC)")
    cn_row  = results_df[results_df['model'] == 'ConceptNet']
    bl_rows = results_df[results_df['model'] != 'ConceptNet']
    if not cn_row.empty and not bl_rows.empty:
        cn_auroc   = cn_row['auroc'].values[0]
        best_bl    = bl_rows.loc[bl_rows['auroc'].idxmax()]
        delta      = cn_auroc - best_bl['auroc']
        lines.append(f"  ConceptNet AUROC : {cn_auroc:.4f}")
        lines.append(f"  Best Baseline    : {best_bl['model']} ({best_bl['auroc']:.4f})")
        lines.append(f"  Δ AUROC          : {delta:+.4f}  "
                     f"({'ConceptNet better' if delta > 0 else 'Baseline better'})")
        cn_params  = N_CONCEPTS * len(feature_names) + N_CONCEPTS + N_CONCEPTS * 2 + 2
        lines.append(f"\n  ConceptNet parameters : ~{cn_params:,}  (sparse concept layer + output)")
        lines.append(f"  ConceptNet hidden     : {N_CONCEPTS} interpretable concept nodes")
        lines.append(f"  Standard MLP params   : large fully-connected (256→128→64)")

    sub("ConceptNet — Concept Node Analysis")
    lines.append(f"\n  {'Concept':<50s}  {'Non-Readmit':>11}  {'Readmit':>9}  {'Δ':>7}  {'w_readmit':>9}")
    lines.append(f"  {'─'*50}  {'─'*11}  {'─'*9}  {'─'*7}  {'─'*9}")
    ow = cn_model.output_layer.weight.detach().cpu().numpy()
    for cidx, _, label in CONCEPTS:
        m0 = act_df.loc[act_df['y_true'] == 0, label].mean()
        m1 = act_df.loc[act_df['y_true'] == 1, label].mean()
        lines.append(f"  {label:<50s}  {m0:11.3f}  {m1:9.3f}  {m1-m0:+7.3f}  {ow[1,cidx]:+9.4f}")

    sub("Output Files")
    files = [
        ('results_summary.csv',                  'All model metrics'),
        ('results_summary.json',                  'Same in JSON'),
        ('plot_roc_comparison.png',               'ROC curves'),
        ('plot_pr_comparison.png',                'Precision-Recall curves'),
        ('plot_calibration_comparison.png',       'Calibration curves'),
        ('plot_metric_bars.png',                  'Metric bar charts'),
        ('plot_brier_traintime.png',              'Brier & training time'),
        ('plot_conceptnet_concept_activations.png','ConceptNet concept activations'),
        ('plot_nn_training_curves.png',           'MLP vs ConceptNet training'),
        ('conceptnet_concept_activations_test.csv','Concept scores per patient'),
        ('conceptnet_feature_mapping.json',       'Feature → concept assignment'),
        ('conceptnet_readmission.pt',             'ConceptNet checkpoint'),
        ('<model>_preds.csv',                     'Predicted probabilities per model'),
        ('<model>_roc.csv / _pr.csv / _cal.csv',  'Curve data per model'),
    ]
    for fname, desc in files:
        lines.append(f"  {fname:<48s} {desc}")

    report = "\n".join(lines)
    path   = os.path.join(OUTPUT_DIR, 'report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    log.info(f"\n{report}\n")
    log.info(f"  Report saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10) MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("═" * 60)
    log.info("  Full Comparison — 30-Day Readmission Prediction")
    log.info(f"  Device: {DEVICE}")
    log.info("═" * 60)

    # ── Load data ─────────────────────────────────────────────────────────
    X_df, y = load_data()

    # ── Shared train / val / test split ───────────────────────────────────
    X_tv, X_te, y_tv, y_te = train_test_split(
        X_df, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=VAL_SIZE, random_state=SEED, stratify=y_tv)

    log.info(f"Split — train {len(X_tr):,} | val {len(X_va):,} | test {len(X_te):,}")
    log.info(f"  Readmission rate — train {100*y_tr.mean():.1f}% | "
             f"val {100*y_va.mean():.1f}% | test {100*y_te.mean():.1f}%")

    dataset_info = {
        'Total admissions'            : len(y),
        'Train / Val / Test'          : f'{len(X_tr):,} / {len(X_va):,} / {len(X_te):,}',
        '30-day readmission rate'     : f'{100*y.mean():.1f}%',
        'Train positives'             : f'{int(y_tr.sum()):,}  ({100*y_tr.mean():.1f}%)',
        'Test positives'              : f'{int(y_te.sum()):,}  ({100*y_te.mean():.1f}%)',
    }

    # ── Preprocess (fit once on train, transform all splits) ──────────────
    log.info("Preprocessing …")
    prep    = Preprocessor(list(X_df.columns))
    X_tr_np = prep.fit_transform(X_tr)
    X_va_np = prep.transform(X_va)
    X_te_np = prep.transform(X_te)
    n_feat  = X_tr_np.shape[1]
    log.info(f"  Preprocessed feature count: {n_feat}")
    dataset_info['Preprocessed features'] = str(n_feat)

    y_tr_np = y_tr.to_numpy()
    y_va_np = y_va.to_numpy()
    y_te_np = y_te.to_numpy()
    base_rate = float(y_te_np.mean())

    # ── Run all models ─────────────────────────────────────────────────────
    all_metrics  = []
    all_curves   = []   # (name, curves_tuple, auroc, auprc)

    def register(metrics, y_prob, curves):
        if metrics is None: return
        all_metrics.append(metrics)
        if curves is not None:
            all_curves.append((metrics['model'], curves,
                                metrics['auroc'], metrics['auprc']))

    m1, p1, c1 = run_logistic_regression(X_tr_np, y_tr_np, X_te_np, y_te_np)
    register(m1, p1, c1)

    m2, p2, c2 = run_random_forest(X_tr_np, y_tr_np, X_te_np, y_te_np)
    register(m2, p2, c2)

    m3, p3, c3 = run_xgboost(X_tr_np, y_tr_np, X_va_np, y_va_np, X_te_np, y_te_np)
    register(m3, p3, c3)

    m4, p4, c4 = run_lightgbm(X_tr_np, y_tr_np, X_va_np, y_va_np, X_te_np, y_te_np)
    register(m4, p4, c4)

    m5, p5, c5 = run_mlp(X_tr_np, y_tr_np, X_va_np, y_va_np, X_te_np, y_te_np)
    register(m5, p5, c5)

    m6, p6, c6, cn_model, cn_mask, act_df = run_conceptnet(
        X_tr_np, y_tr_np, X_va_np, y_va_np, X_te_np, y_te_np, prep.out_cols_)
    register(m6, p6, c6)

    # ── Results table ──────────────────────────────────────────────────────
    col_order = ['model', 'auroc', 'auprc', 'brier',
                 'f1_readmit', 'precision_readmit', 'recall_readmit', 'train_time_s']
    results = (pd.DataFrame(all_metrics)
               [[c for c in col_order if c in pd.DataFrame(all_metrics).columns]]
               .sort_values('auroc', ascending=False)
               .reset_index(drop=True))

    results.to_csv(os.path.join(OUTPUT_DIR, 'results_summary.csv'), index=False)
    with open(os.path.join(OUTPUT_DIR, 'results_summary.json'), 'w') as f:
        json.dump(results.to_dict(orient='records'), f, indent=2)

    # ── Plots ──────────────────────────────────────────────────────────────
    log.info("\n" + "═" * 60)
    log.info("  Generating comparison plots …")
    log.info("═" * 60)

    curve_data_for_plot = [(name, curves, auroc)
                           for name, curves, auroc, _ in all_curves]
    pr_data_for_plot    = [(name, curves, auprc)
                           for name, curves, _, auprc in all_curves]

    plot_roc(curve_data_for_plot)
    plot_pr(pr_data_for_plot, base_rate)
    plot_calibration(curve_data_for_plot)
    plot_metric_bars(results)
    plot_brier_train_time(results)
    plot_conceptnet_concepts(act_df)
    plot_nn_training_histories()

    # ── Text report ────────────────────────────────────────────────────────
    log.info("\n" + "═" * 60)
    log.info("  Writing text report …")
    log.info("═" * 60)
    write_report(results, dataset_info, prep.out_cols_, act_df, cn_model)

    # ── Final summary ──────────────────────────────────────────────────────
    log.info("\n" + "═" * 60)
    log.info("  FINAL RESULTS  (sorted by AUROC)")
    log.info("═" * 60)
    log.info(f"\n{results.to_string(index=False)}\n")
    log.info(f"✓ All outputs saved to: {OUTPUT_DIR}")

    return results


if __name__ == '__main__':
    results = main()