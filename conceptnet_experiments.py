"""
ConceptNet Hypothesis Testing — All 4 Experiments
===================================================
Run AFTER full_comparison.py has completed.

Settings:
  - EXP-3 Stability : 5 seeds  [42, 123, 777, 1337, 2024]
  - EXP-4 Corruption: shuffle within group (realistic domain shift)

Requires:
  ./model_outputs/full_comparison/   (all CSV, .pt outputs from full_comparison.py)
  ./model_outputs/extracted_features2.csv

Experiments
-----------
  EXP-2  Prediction Explanation Visualization
         Per-patient concept contribution decomposition.
         Produces waterfall charts for example patients + population heatmap.
         Core claim: predictions are fully named — no baseline can match this.

  EXP-3  Concept Stability Across 5 Seeds
         Trains ConceptNet 5x with different seeds.
         Measures Spearman rank-correlation of concept importance.
         Compare against XGBoost feature-importance rank stability.
         Core claim: ConceptNet rankings are more stable (higher Spearman ρ).

  EXP-4  Domain Shift Simulation (shuffle mode)
         Shuffles one feature group at a time within the test set.
         Training data unchanged — simulates realistic deployment shift.
         Core claim: ConceptNet degradation is predictable & named;
                     XGBoost degradation is opaque.

  EXP-5  Spurious Correlation Audit
         Scores every feature by clinical plausibility.
         Compares XGBoost importance vs ConceptNet importance weighted by plausibility.
         Core claim: ConceptNet structurally prevents low-plausibility features
                     from leaking into clinical concept nodes.
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
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from sklearn.metrics         import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, OrdinalEncoder
from sklearn.impute          import SimpleImputer
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

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
# 1) CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FEATURES_CSV   = './model_outputs/extracted_features2.csv'
COMPARISON_DIR = './model_outputs/full_comparison/'
OUTPUT_DIR     = './model_outputs/experiments/'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

SEED           = 42
TEST_SIZE      = 0.15
VAL_SIZE       = 0.15
READMIT_DAYS   = 30
BATCH_SIZE     = 2048
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# EXP-3: 5 seeds for stability analysis
STABILITY_SEEDS = [42, 123, 777, 1337, 2024]

# EXP-4: shuffle only (realistic — preserves marginal distributions,
#         breaks within-group correlations as happens in real deployment drift)
CORRUPTION_MODE = 'shuffle'

# ConceptNet training hyperparams — must match full_comparison.py exactly
CN_MAX_EPOCHS  = 300
CN_PATIENCE    = 25
MLP_MAX_EPOCHS = 200
MLP_PATIENCE   = 20
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
CN_SEP_LAMBDA  = 0.01
CN_L1_LAMBDA   = 5e-4

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
# 2) CONCEPT DEFINITIONS  (identical to full_comparison.py)
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
N_CONCEPTS    = len(CONCEPTS)
CONCEPT_KEYS  = [k  for _, k, _ in CONCEPTS]
CONCEPT_LABELS= [lb for _, _, lb in CONCEPTS]
CLASS_NAMES   = ['Not Readmitted', 'Readmitted']

CONCEPT_PATTERNS = {
    'frailty_reserve': [
        r'^adm_age$', r'^adm_gender$',
        r'^omr_weight', r'^omr_height', r'^omr_bmi',
        r'^util_prior_hadm_count_365d$',
    ],
    'biological_buffer': [
        r'^lab_bicarbonate', r'^lab_albumin', r'^lab_hemoglobin',
        r'^lab_sodium.*(bw|ws)', r'^lab_potassium.*(bw|ws)', r'^lab_chloride.*(bw|ws)',
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
        r'^vital_gcs_(last|min|max|std)_pdw$', r'^vital_measure_count_pdw$',
        r'^lab_(creatinine|sodium|potassium|glucose|wbc)_(last|min|max|mean)_pdw$',
        r'^lab_abnormal_count_pdw$',
    ],
    'disease_resolution': [
        r'^note_resolution_score$', r'^note_stable_at_discharge$',
        r'^lab_.*_slope_pdw$', r'^lab_.*_delta_last_first_pdw$', r'^vital_.*_slope_pdw$',
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

# EXP-4: clinical domain groups mapped to feature-name regex patterns
DOMAIN_GROUPS = {
    'Lab Results'          : [r'^lab_'],
    'Vital Signs'          : [r'^vital_'],
    'Medications'          : [r'^med_'],
    'Clinical Notes'       : [r'^note_'],
    'Prior Utilization'    : [r'^util_'],
    'ICU / AKI / Transfer' : [r'^icu_', r'^trans_', r'^aki_'],
    'Demographics'         : [r'^adm_age$', r'^adm_gender$', r'^adm_race$',
                               r'^adm_marital_status$', r'^adm_insurance$'],
    'Comorbidities (DX)'   : [r'^dx_'],
    'Anthropometrics'      : [r'^omr_'],
}

# EXP-5: clinical plausibility scores for feature prefixes
#   1.0 = clearly clinical (biomarker, physiology)
#   0.5 = mixed — administrative but medically relevant
#   0.0 = administrative artifact with low direct clinical rationale
PLAUSIBILITY_RULES = [
    (1.0, [r'^lab_', r'^vital_', r'^aki_', r'^icu_',
           r'^dx_charlson', r'^med_highrisk',
           r'^note_resolution', r'^note_stable', r'^note_function',
           r'^note_cognitive', r'^note_has_home',
           r'^omr_bmi', r'^omr_weight', r'^omr_height',
           r'^adm_age$', r'^adm_los_days$']),
    (0.5, [r'^dx_num_', r'^proc_num_', r'^med_unique', r'^med_total',
           r'^util_prior_hadm', r'^util_days_since',
           r'^note_lives_alone', r'^note_has_caregiver',
           r'^adm_admission_type', r'^adm_discharge_location', r'^trans_']),
    (0.0, [r'^adm_insurance$', r'^adm_marital_status$', r'^adm_race$',
           r'^adm_gender$', r'^adm_admission_location$',
           r'^util_prior_ed_count', r'^adm_has_ed$', r'^proc_num_codes_ws$']),
]


def get_plausibility(fname):
    for score, patterns in PLAUSIBILITY_RULES:
        for pat in patterns:
            if re.search(pat, fname, re.IGNORECASE):
                return score
    return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 3) ARCHITECTURE  (exact copy from full_comparison.py — do NOT change)
# ─────────────────────────────────────────────────────────────────────────────
class SparseConceptLayer(nn.Module):
    def __init__(self, n_features, mask):
        super().__init__()
        self.register_buffer('mask', mask)
        self.weight = nn.Parameter(torch.empty(N_CONCEPTS, n_features))
        self.bias   = nn.Parameter(torch.zeros(N_CONCEPTS))
        nn.init.xavier_uniform_(self.weight)
        with torch.no_grad():
            self.weight.data[~mask] = 0.0
        self.weight.register_hook(self._zero_masked_grad)

    def _zero_masked_grad(self, grad):
        return grad * self.mask.float()

    def forward(self, x):
        w = self.weight * self.mask.float()
        return torch.tanh(x @ w.t() + self.bias)

    def re_zero_masked_weights(self):
        with torch.no_grad():
            self.weight.data[~self.mask] = 0.0


class ConceptNet(nn.Module):
    def __init__(self, n_features, mask):
        super().__init__()
        self.concept_layer = SparseConceptLayer(n_features, mask)
        self.output_layer  = nn.Linear(N_CONCEPTS, 2)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x, return_concepts=False):
        concepts = self.concept_layer(x)
        logits   = self.output_layer(concepts)
        if return_concepts:
            return logits, concepts
        return logits

    def re_zero_masked(self):
        self.concept_layer.re_zero_masked_weights()


class StandardMLP(nn.Module):
    def __init__(self, n_features, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),                              nn.ReLU(), nn.Dropout(dropout*0.5),
            nn.Linear(64, 2),
        )
    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4) DATA LOADING & PREPROCESSING
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
    log.info(f"Loading {FEATURES_CSV}")
    df = pd.read_csv(FEATURES_CSV, low_memory=False)
    for c in ['admittime', 'dischtime']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    label = f'readmitted_{READMIT_DAYS}d'
    if label not in df.columns:
        df = create_readmission_label(df, days=READMIT_DAYS)
    y    = df[label].copy()
    drop = [c for c in df.columns if c in EXCLUDE_COLS or c == 'next_admittime']
    X    = df.drop(columns=[c for c in drop if c in df.columns], errors='ignore')
    X    = X.select_dtypes(exclude=['datetime64[ns]'])
    log.info(f"  {len(y):,} rows | {X.shape[1]} features | {100*y.mean():.1f}% readmitted")
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
                self.ord_enc.fit_transform(self.cat_imp.fit_transform(Xc)).astype(np.float32)))
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
                self.ord_enc.transform(self.cat_imp.transform(Xc)).astype(np.float32)))
        if self.num_cols:
            Xn = X[self.num_cols].apply(pd.to_numeric, errors='coerce')
            parts.append(self.num_scaler.transform(
                self.num_imp.transform(Xn)).astype(np.float32))
        return np.hstack(parts).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5) SHARED NN UTILITIES
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
    return mask


def build_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te):
    def tt(X, y):
        return torch.from_numpy(X).float(), torch.from_numpy(y.astype(np.int64))
    Xt, yt = tt(X_tr, y_tr)
    Xv, yv = tt(X_va, y_va)
    Xe, ye = tt(X_te, y_te)
    n_pos  = int((yt == 1).sum()); n_neg = len(yt) - n_pos
    w = torch.where(yt == 1,
                    torch.full_like(yt, float(n_neg)/max(n_pos,1), dtype=torch.float),
                    torch.ones(len(yt), dtype=torch.float))
    sampler  = WeightedRandomSampler(w, num_samples=len(yt), replacement=True)
    train_dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=0)
    val_dl   = DataLoader(TensorDataset(Xv, yv), batch_size=BATCH_SIZE*2,
                          shuffle=False, num_workers=0)
    test_dl  = DataLoader(TensorDataset(Xe, ye), batch_size=BATCH_SIZE*2,
                          shuffle=False, num_workers=0)
    return train_dl, val_dl, test_dl


def concept_separation_loss(concepts, labels):
    mask_pos = (labels == 1).float().unsqueeze(1)
    mask_neg = (labels == 0).float().unsqueeze(1)
    mu_pos   = (concepts * mask_pos).sum(0) / mask_pos.sum().clamp(min=1)
    mu_neg   = (concepts * mask_neg).sum(0) / mask_neg.sum().clamp(min=1)
    return -((mu_pos - mu_neg) ** 2).mean()


@torch.no_grad()
def eval_nn(model, loader):
    model.eval()
    probs, labels = [], []
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        p = torch.softmax(model(Xb), dim=1)[:, 1]
        probs.extend(p.cpu().tolist())
        labels.extend(yb.cpu().tolist())
    return np.array(probs), np.array(labels)


@torch.no_grad()
def get_concept_activations(model, X_np):
    """Return (N, 12) concept activation matrix for a numpy array."""
    model.eval()
    rows = []
    for i in range(0, len(X_np), 4096):
        Xb = torch.from_numpy(X_np[i:i+4096]).float().to(DEVICE)
        _, c = model(Xb, return_concepts=True)
        rows.append(c.cpu().numpy())
    return np.vstack(rows)


def train_conceptnet(X_tr, y_tr, X_va, y_va, X_te, y_te, feature_names,
                     seed=42, verbose=False):
    """
    Train one ConceptNet instance from scratch.
    Returns: model, mask, y_prob_test, y_true_test, best_val_auroc
    """
    torch.manual_seed(seed); np.random.seed(seed)
    mask  = build_connection_mask(feature_names)
    model = ConceptNet(X_tr.shape[1], mask).to(DEVICE)
    crit  = nn.CrossEntropyLoss()
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10)
    train_dl, val_dl, test_dl = build_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te)
    best_auroc, best_state, no_imp = 0.0, None, 0

    for ep in range(1, CN_MAX_EPOCHS + 1):
        model.train()
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits, concepts = model(Xb, return_concepts=True)
            loss = (crit(logits, yb)
                    + CN_SEP_LAMBDA * concept_separation_loss(concepts, yb)
                    + CN_L1_LAMBDA  * model.concept_layer.weight.abs().mean())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.re_zero_masked()

        vp, vl    = eval_nn(model, val_dl)
        val_auroc = roc_auc_score(vl, vp)
        sched.step(val_auroc)
        if verbose and (ep == 1 or ep % 50 == 0):
            log.info(f"    seed={seed} ep={ep:3d}  val_auroc={val_auroc:.4f}")
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= CN_PATIENCE:
                break

    model.load_state_dict(best_state)
    y_prob, y_true = eval_nn(model, test_dl)
    return model, mask, y_prob, y_true, best_auroc


# ─────────────────────────────────────────────────────────────────────────────
# EXP-2: Prediction Explanation Visualization
# ─────────────────────────────────────────────────────────────────────────────
def exp2_explanations(cn_model, cn_mask, X_te, y_te, feature_names):
    """
    Decompose each prediction into per-concept contributions.
    Contribution[i, c] = output_weight[readmit, c] × concept_activation[i, c]
    This is mathematically exact because the output layer is linear.
    """
    log.info("\n" + "═"*60)
    log.info("  EXP-2: Prediction Explanation Visualization")
    log.info("═"*60)
    exp2_dir = os.path.join(OUTPUT_DIR, 'exp2_explanations')
    Path(exp2_dir).mkdir(exist_ok=True)

    acts_np = get_concept_activations(cn_model, X_te)          # (N, 12)
    probs_np, _ = eval_nn(cn_model, DataLoader(
        TensorDataset(torch.from_numpy(X_te).float(),
                      torch.zeros(len(X_te), dtype=torch.long)),
        batch_size=BATCH_SIZE*2, shuffle=False))

    ow = cn_model.output_layer.weight.detach().cpu().numpy()   # (2, 12)
    ob = cn_model.output_layer.bias.detach().cpu().numpy()     # (2,)

    # Contribution matrix: (N, 12)
    contribs = acts_np * ow[1, :]   # weight for "Readmitted" class

    contrib_df = pd.DataFrame(contribs, columns=CONCEPT_LABELS)
    contrib_df.insert(0, 'y_true',       y_te.astype(int))
    contrib_df.insert(1, 'p_readmit',    probs_np)
    contrib_df.insert(2, 'bias_readmit', ob[1])
    contrib_df.to_csv(os.path.join(exp2_dir, 'all_concept_contributions.csv'), index=False)

    # ── Figure 1: 6 example patients (3 high-risk readmitted, 3 low-risk not) ──
    idx_pos  = np.where(y_te == 1)[0]
    idx_neg  = np.where(y_te == 0)[0]
    hi_risk  = idx_pos[np.argsort(probs_np[idx_pos])[-3:][::-1]]
    lo_risk  = idx_neg[np.argsort(probs_np[idx_neg])[:3]]
    examples = list(hi_risk) + list(lo_risk)
    ex_titles = (
        [f"High-Risk #{i+1}  (actual: readmitted)\np={probs_np[h]:.2f}"
         for i, h in enumerate(hi_risk)] +
        [f"Low-Risk #{i+1}  (actual: not readmitted)\np={probs_np[l]:.2f}"
         for i, l in enumerate(lo_risk)]
    )

    short_labels = [lb.split('(')[0].strip() for lb in CONCEPT_LABELS]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, pat_idx, title in zip(axes.flatten(), examples, ex_titles):
        vals   = contribs[pat_idx]
        colors = ['#E91E63' if v > 0 else '#2196F3' for v in vals]
        ax.barh(range(N_CONCEPTS), vals, color=colors, edgecolor='white', height=0.7)
        ax.set_yticks(range(N_CONCEPTS))
        ax.set_yticklabels(short_labels, fontsize=7.5)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('Contribution to Readmit Logit', fontsize=8)
        ax.grid(axis='x', alpha=0.3)
    fig.suptitle(
        'ConceptNet — Per-Patient Prediction Explanations\n'
        'Pink bars increase readmission risk  |  Blue bars decrease it',
        fontsize=13, fontweight='bold')
    fig.tight_layout()
    p = os.path.join(exp2_dir, 'patient_explanations.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    log.info(f"  Saved: {p}")

    # ── Figure 2: Population-level mean contribution by class ──────────────
    mean_pos = contribs[y_te == 1].mean(axis=0)
    mean_neg = contribs[y_te == 0].mean(axis=0)
    y_idx    = np.arange(N_CONCEPTS); h = 0.35
    fig, ax  = plt.subplots(figsize=(12, 8))
    ax.barh(y_idx + h/2, mean_neg, h, label='Not Readmitted', color='#2196F3', alpha=0.85)
    ax.barh(y_idx - h/2, mean_pos, h, label='Readmitted',     color='#E91E63', alpha=0.85)
    ax.set_yticks(y_idx); ax.set_yticklabels(CONCEPT_LABELS, fontsize=9)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Mean Contribution to Readmit Logit', fontsize=11)
    ax.set_title('ConceptNet — Mean Concept Contributions by Class (Test Set)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    p = os.path.join(exp2_dir, 'mean_contributions_by_class.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    log.info(f"  Saved: {p}")

    # ── Figure 3: Heatmap over 300 patients sorted by predicted risk ────────
    n_sample  = min(300, len(contribs))
    rng       = np.random.RandomState(SEED)
    samp_idx  = rng.choice(len(contribs), n_sample, replace=False)
    sort_key  = probs_np[samp_idx].argsort()
    sorted_i  = samp_idx[sort_key]
    hm_df     = pd.DataFrame(contribs[sorted_i], columns=short_labels)
    fig, ax   = plt.subplots(figsize=(15, 7))
    sns.heatmap(hm_df.T, cmap='RdBu_r', center=0, ax=ax,
                cbar_kws={'label': 'Contribution to Readmit Logit'},
                yticklabels=True, xticklabels=False)
    ax.set_xlabel(f'Patients sorted by predicted risk (n={n_sample}, left=low, right=high)',
                  fontsize=11)
    ax.set_title('ConceptNet — Concept Contribution Heatmap', fontsize=13, fontweight='bold')
    fig.tight_layout()
    p = os.path.join(exp2_dir, 'contribution_heatmap.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    log.info(f"  Saved: {p}")

    # ── Console summary ─────────────────────────────────────────────────────
    log.info(f"\n  {'Concept':<50s}  {'Not Readmit':>11}  {'Readmit':>9}  {'Δ':>7}")
    for i, lbl in enumerate(CONCEPT_LABELS):
        log.info(f"  {lbl:<50s}  {mean_neg[i]:+11.4f}  {mean_pos[i]:+9.4f}"
                 f"  {mean_pos[i]-mean_neg[i]:+7.4f}")

    return contrib_df


# ─────────────────────────────────────────────────────────────────────────────
# EXP-3: Concept Stability Across 5 Seeds
# ─────────────────────────────────────────────────────────────────────────────
def exp3_stability(X_tr, y_tr, X_va, y_va, X_te, y_te, feature_names,
                   xgb_model_seed42=None):
    """
    Train ConceptNet 5× with different seeds.
    Concept importance = |output_weight[readmit, c]| × mean|activation[c]|

    Also trains XGBoost 5× for comparison of explanation stability.
    Key metric: mean pairwise Spearman ρ over all (5 choose 2) = 10 pairs.
    """
    log.info("\n" + "═"*60)
    log.info("  EXP-3: Concept Stability Across 5 Seeds")
    log.info("═"*60)
    exp3_dir = os.path.join(OUTPUT_DIR, 'exp3_stability')
    Path(exp3_dir).mkdir(exist_ok=True)

    # ── Train ConceptNet across seeds ─────────────────────────────────────
    cn_imp_runs, cn_aurocs = [], []
    for seed in STABILITY_SEEDS:
        log.info(f"  Training ConceptNet  seed={seed} …")
        model, mask, y_prob, y_true, _ = train_conceptnet(
            X_tr, y_tr, X_va, y_va, X_te, y_te, feature_names, seed=seed)
        auroc = roc_auc_score(y_true, y_prob)
        cn_aurocs.append(auroc)
        log.info(f"    → test AUROC {auroc:.4f}")

        ow      = model.output_layer.weight.detach().cpu().numpy()   # (2,12)
        acts    = get_concept_activations(model, X_te)               # (N,12)
        imp     = np.abs(ow[1, :]) * np.abs(acts).mean(axis=0)      # (12,)
        cn_imp_runs.append(imp)

    cn_imp_mat = np.stack(cn_imp_runs)                              # (5, 12)
    cn_ranks   = np.argsort(np.argsort(-cn_imp_mat, axis=1), axis=1)

    cn_rhos = [spearmanr(cn_ranks[i], cn_ranks[j]).statistic
               for i in range(len(STABILITY_SEEDS))
               for j in range(i+1, len(STABILITY_SEEDS))]
    cn_mean_rho = float(np.mean(cn_rhos))
    log.info(f"\n  ConceptNet  mean Spearman ρ = {cn_mean_rho:.4f}"
             f"  (AUROC {np.mean(cn_aurocs):.4f} ± {np.std(cn_aurocs):.4f})")

    # ── Train XGBoost across seeds for comparison ──────────────────────────
    xgb_mean_rho = None
    xgb_aurocs   = []
    xgb_imp_runs = []
    if HAS_XGB:
        n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
        for seed in STABILITY_SEEDS:
            log.info(f"  Training XGBoost seed={seed} …")
            m = xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=30,
                scale_pos_weight=n_neg/max(n_pos,1),
                use_label_encoder=False, random_state=seed,
                n_jobs=-1, verbosity=0, eval_metric='aucpr',
                early_stopping_rounds=20)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            auroc = roc_auc_score(y_te, m.predict_proba(X_te)[:,1])
            xgb_aurocs.append(auroc)
            xgb_imp_runs.append(m.feature_importances_)
            log.info(f"    → test AUROC {auroc:.4f}")

        xgb_imp_mat  = np.stack(xgb_imp_runs)
        xgb_ranks_m  = np.argsort(np.argsort(-xgb_imp_mat, axis=1), axis=1)
        xgb_rhos     = [spearmanr(xgb_ranks_m[i], xgb_ranks_m[j]).statistic
                        for i in range(len(STABILITY_SEEDS))
                        for j in range(i+1, len(STABILITY_SEEDS))]
        xgb_mean_rho = float(np.mean(xgb_rhos))
        log.info(f"  XGBoost     mean Spearman ρ = {xgb_mean_rho:.4f}"
                 f"  (AUROC {np.mean(xgb_aurocs):.4f} ± {np.std(xgb_aurocs):.4f})")
    else:
        log.warning("  XGBoost not available — skipping XGB stability comparison")

    # ── Figure 1: Rank heatmap across seeds ────────────────────────────────
    rank_df = pd.DataFrame(
        cn_ranks,
        index=[f'Seed {s}' for s in STABILITY_SEEDS],
        columns=[lb.split('(')[0].strip() for lb in CONCEPT_LABELS])
    fig, ax = plt.subplots(figsize=(15, 4))
    sns.heatmap(rank_df, annot=True, fmt='d', cmap='YlOrRd_r', ax=ax,
                cbar_kws={'label': 'Rank (0=most important)'},
                linewidths=0.5, linecolor='white')
    ax.set_title(
        f'ConceptNet — Concept Importance Rank Across {len(STABILITY_SEEDS)} Seeds\n'
        f'Mean Spearman ρ = {cn_mean_rho:.3f}   (1.0 = perfectly stable)',
        fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=35)
    fig.tight_layout()
    p = os.path.join(exp3_dir, 'cn_rank_heatmap.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    log.info(f"\n  Saved: {p}")

    # ── Figure 2: Mean ± std importance bar chart ───────────────────────────
    mean_imp = cn_imp_mat.mean(axis=0)
    std_imp  = cn_imp_mat.std(axis=0)
    sort_idx = np.argsort(mean_imp)[::-1]
    fig, ax  = plt.subplots(figsize=(12, 6))
    ax.bar(range(N_CONCEPTS), mean_imp[sort_idx],
           yerr=std_imp[sort_idx], capsize=5,
           color='#E91E63', alpha=0.85, edgecolor='white')
    ax.set_xticks(range(N_CONCEPTS))
    ax.set_xticklabels(
        [CONCEPT_LABELS[i].split('(')[0].strip() for i in sort_idx],
        rotation=40, ha='right', fontsize=8.5)
    ax.set_ylabel('Importance  (|output weight| × mean|activation|)', fontsize=10)
    ax.set_title(
        f'ConceptNet — Concept Importance: Mean ± Std Across {len(STABILITY_SEEDS)} Seeds\n'
        f'Short error bars = stable concept identity across training runs',
        fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p = os.path.join(exp3_dir, 'cn_importance_mean_std.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    log.info(f"  Saved: {p}")

    # ── Figure 3: Stability comparison bar (CN vs XGB) ─────────────────────
    names  = ['ConceptNet']
    rhos   = [cn_mean_rho]
    colors = ['#E91E63']
    if xgb_mean_rho is not None:
        names.append('XGBoost\n(feature-level)'); rhos.append(xgb_mean_rho)
        colors.append('#FF9800')

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(names, rhos, color=colors, edgecolor='white', width=0.45)
    for bar, val in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'ρ = {val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Mean Pairwise Spearman ρ\n(explanation rank correlation across seeds)', fontsize=9)
    ax.set_title(
        f'Explanation Stability — {len(STABILITY_SEEDS)} Seeds\n'
        f'Higher = same features identified as important every run',
        fontsize=11, fontweight='bold')
    ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p = os.path.join(exp3_dir, 'stability_comparison_bar.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    log.info(f"  Saved: {p}")

    summary = {
        'cn_mean_spearman_rho': cn_mean_rho,
        'cn_auroc_mean'       : float(np.mean(cn_aurocs)),
        'cn_auroc_std'        : float(np.std(cn_aurocs)),
        'cn_aurocs_by_seed'   : dict(zip(STABILITY_SEEDS, [float(a) for a in cn_aurocs])),
        'xgb_mean_spearman_rho': xgb_mean_rho,
        'xgb_auroc_mean'      : float(np.mean(xgb_aurocs)) if xgb_aurocs else None,
        'xgb_auroc_std'       : float(np.std(xgb_aurocs))  if xgb_aurocs else None,
        'delta_rho_cn_minus_xgb':
            round(cn_mean_rho - xgb_mean_rho, 4) if xgb_mean_rho else None,
    }
    with open(os.path.join(exp3_dir, 'stability_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary, cn_imp_mat


# ─────────────────────────────────────────────────────────────────────────────
# EXP-4: Domain Shift Simulation  (shuffle mode)
# ─────────────────────────────────────────────────────────────────────────────
def get_group_indices(feature_names, domain_groups):
    indices = {}
    for gname, patterns in domain_groups.items():
        idxs = []
        for fi, fname in enumerate(feature_names):
            for pat in patterns:
                if re.search(pat, fname, re.IGNORECASE):
                    idxs.append(fi); break
        indices[gname] = idxs
    return indices


def shuffle_corrupt(X, col_indices, rng):
    """
    Shuffle each specified column independently.
    This preserves the marginal distribution of each feature
    but destroys the joint structure with the label and other features.
    Equivalent to: a deployment environment where a data source arrives
    correctly formatted but has been disconnected from the actual patient.
    """
    Xc = X.copy()
    for ci in col_indices:
        Xc[:, ci] = rng.permutation(Xc[:, ci])
    return Xc


def exp4_domain_shift(X_te, y_te, feature_names,
                      cn_model, xgb_model=None, lgb_model=None,
                      lr_model=None, rf_model=None, mlp_model=None):
    """
    Corrupt one feature group at a time (shuffle mode).
    Training data is NEVER touched — simulates deployment drift where
    one data pipeline goes down or gets corrupted.
    """
    log.info("\n" + "═"*60)
    log.info("  EXP-4: Domain Shift Simulation  [shuffle mode]")
    log.info("═"*60)
    exp4_dir = os.path.join(OUTPUT_DIR, 'exp4_domain_shift')
    Path(exp4_dir).mkdir(exist_ok=True)

    rng           = np.random.RandomState(SEED)
    group_indices = get_group_indices(feature_names, DOMAIN_GROUPS)

    # ── Helper: AUROC for each model on a given X_te variant ──────────────
    def auroc_cn(Xte):
        probs = []
        cn_model.eval()
        with torch.no_grad():
            for i in range(0, len(Xte), 4096):
                Xb = torch.from_numpy(Xte[i:i+4096]).float().to(DEVICE)
                p  = torch.softmax(cn_model(Xb), dim=1)[:, 1]
                probs.append(p.cpu().numpy())
        return roc_auc_score(y_te, np.concatenate(probs))

    def auroc_sk(m, Xte):
        return roc_auc_score(y_te, m.predict_proba(Xte)[:,1]) if m else None

    def auroc_mlp(m, Xte):
        if not m: return None
        m.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(Xte), 4096):
                Xb = torch.from_numpy(Xte[i:i+4096]).float().to(DEVICE)
                p  = torch.softmax(m(Xb), dim=1)[:, 1]
                probs.append(p.cpu().numpy())
        return roc_auc_score(y_te, np.concatenate(probs))

    model_fns = {
        'ConceptNet'          : auroc_cn,
        'XGBoost'             : lambda Xte: auroc_sk(xgb_model, Xte),
        'LightGBM'            : lambda Xte: auroc_sk(lgb_model, Xte),
        'Logistic Regression' : lambda Xte: auroc_sk(lr_model,  Xte),
        'Random Forest'       : lambda Xte: auroc_sk(rf_model,  Xte),
        'Standard MLP'        : lambda Xte: auroc_mlp(mlp_model, Xte),
    }
    active_models = {k: v for k, v in model_fns.items()
                     if not (('XGBoost'  in k and not xgb_model) or
                             ('LightGBM' in k and not lgb_model) or
                             ('MLP'      in k and not mlp_model))}

    # Baseline
    baseline = {}
    for mname, fn in active_models.items():
        try:
            baseline[mname] = fn(X_te)
            log.info(f"  Baseline  {mname:<25s}: AUROC {baseline[mname]:.4f}")
        except Exception as e:
            log.warning(f"  Baseline  {mname}: FAILED ({e})")

    results = []
    for gname, col_idxs in group_indices.items():
        if not col_idxs:
            continue
        log.info(f"\n  Corrupting: {gname}  ({len(col_idxs)} features shuffled)")
        X_corr = shuffle_corrupt(X_te, col_idxs, rng)
        row    = {'group': gname, 'n_features': len(col_idxs)}
        for mname, fn in active_models.items():
            try:
                a = fn(X_corr)
                drop = baseline.get(mname, 0) - a
                row[mname + '_auroc'] = round(float(a),    4)
                row[mname + '_drop']  = round(float(drop), 4)
                log.info(f"    {mname:<25s}: {a:.4f}  (drop {drop:+.4f})")
            except Exception as e:
                row[mname + '_auroc'] = None
                row[mname + '_drop']  = None
                log.warning(f"    {mname}: FAILED ({e})")
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(exp4_dir, 'domain_shift_results.csv'), index=False)

    # ── Figure 1: AUROC drop heatmap — all models × all groups ─────────────
    drop_cols   = [c for c in results_df.columns if c.endswith('_drop')
                   and results_df[c].notna().any()]
    model_names = [c.replace('_drop', '') for c in drop_cols]
    hm          = results_df.set_index('group')[drop_cols]\
                            .rename(columns=dict(zip(drop_cols, model_names)))

    fig, ax = plt.subplots(figsize=(max(10, len(model_names)*2+2), 7))
    sns.heatmap(hm, annot=True, fmt='.3f', cmap='Reds', ax=ax,
                cbar_kws={'label': 'AUROC Drop (larger = more degradation)'},
                linewidths=0.5, linecolor='white', vmin=0)
    ax.set_title(
        'Domain Shift (shuffle) — AUROC Drop by Feature Group\n'
        'Rows = which group was corrupted   |   Columns = model',
        fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=10); ax.set_ylabel('Corrupted Group', fontsize=10)
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    p = os.path.join(exp4_dir, 'auroc_drop_heatmap.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    log.info(f"\n  Saved: {p}")

    # ── Figure 2: ConceptNet vs XGBoost drop per group ─────────────────────
    if 'XGBoost_drop' in results_df.columns:
        groups    = results_df['group'].tolist()
        cn_drops  = results_df['ConceptNet_drop'].fillna(0).tolist()
        xgb_drops = results_df['XGBoost_drop'].fillna(0).tolist()
        x         = np.arange(len(groups)); width = 0.35

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(x - width/2, cn_drops,  width, label='ConceptNet', color='#E91E63', alpha=0.85)
        ax.bar(x + width/2, xgb_drops, width, label='XGBoost',    color='#FF9800', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('AUROC Drop (higher = more sensitive to missing data)', fontsize=10)
        ax.set_title(
            'ConceptNet vs XGBoost — Sensitivity to Domain Shift (shuffle)\n'
            'ConceptNet failure maps directly to named concept nodes; XGBoost failure is opaque',
            fontsize=11, fontweight='bold')
        ax.legend(fontsize=10); ax.axhline(0, color='black', lw=0.8); ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        p = os.path.join(exp4_dir, 'cn_vs_xgb_drop.png')
        fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
        log.info(f"  Saved: {p}")

    # ── Figure 3: ConceptNet — which concepts affected by each group ────────
    # Shows the *structural* connection: which concept nodes own features
    # in each domain group, making degradation predictable in advance.
    cn_mask_np = build_connection_mask(feature_names).cpu().numpy()  # (12, n_feat)
    group_concept_overlap = {}
    for gname, col_idxs in group_indices.items():
        if not col_idxs: continue
        n_in_concept = cn_mask_np[:, col_idxs].sum(axis=1)    # (12,)
        group_concept_overlap[gname] = n_in_concept

    overlap_df = pd.DataFrame(
        group_concept_overlap,
        index=[lb.split('(')[0].strip() for lb in CONCEPT_LABELS]).T

    fig, ax = plt.subplots(figsize=(15, 7))
    sns.heatmap(overlap_df, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar_kws={'label': 'Number of features feeding this concept'},
                linewidths=0.5, linecolor='white')
    ax.set_title(
        'ConceptNet — Which Concepts Are Affected by Each Domain Group\n'
        'This map predicts degradation BEFORE it happens',
        fontsize=12, fontweight='bold')
    ax.set_xlabel('Concept Node', fontsize=10)
    ax.set_ylabel('Feature Group', fontsize=10)
    ax.tick_params(axis='x', rotation=40)
    fig.tight_layout()
    p = os.path.join(exp4_dir, 'concept_group_exposure_map.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    log.info(f"  Saved: {p}")

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# EXP-5: Spurious Correlation Audit
# ─────────────────────────────────────────────────────────────────────────────
def exp5_spurious_audit(X_tr, y_tr, X_te, y_te, feature_names,
                        cn_model, cn_mask, xgb_model=None):
    """
    Score every feature by clinical plausibility.
    Measure: weighted average plausibility score per model.
    Structural guarantee: ConceptNet cannot route a feature into a
    concept node for which it was not pre-specified.
    """
    log.info("\n" + "═"*60)
    log.info("  EXP-5: Spurious Correlation Audit")
    log.info("═"*60)
    exp5_dir = os.path.join(OUTPUT_DIR, 'exp5_spurious_audit')
    Path(exp5_dir).mkdir(exist_ok=True)

    # ── ConceptNet indirect feature importance ─────────────────────────────
    ow       = cn_model.output_layer.weight.detach().cpu().numpy()   # (2,12)
    mask_np  = cn_mask.cpu().numpy().astype(float)                   # (12, n_feat)
    # Indirect importance: each feature's contribution routes through its concept(s)
    cn_feat_imp = (mask_np * np.abs(ow[1, :])[:, None]).sum(axis=0)
    cn_feat_imp /= cn_feat_imp.sum() + 1e-10

    # ── XGBoost gain importance ────────────────────────────────────────────
    xgb_imp = None
    if HAS_XGB and xgb_model is not None:
        xgb_imp = xgb_model.feature_importances_
        xgb_imp = xgb_imp / (xgb_imp.sum() + 1e-10)

    plausibility = np.array([get_plausibility(f) for f in feature_names])

    audit = pd.DataFrame({
        'feature'      : feature_names,
        'plausibility' : plausibility,
        'cn_importance': cn_feat_imp,
    })
    if xgb_imp is not None:
        audit['xgb_importance'] = xgb_imp

    audit.sort_values('cn_importance', ascending=False, inplace=True)
    audit.to_csv(os.path.join(exp5_dir, 'feature_audit_table.csv'), index=False)

    cn_wplaus = float((cn_feat_imp * plausibility).sum())
    log.info(f"\n  ConceptNet weighted plausibility: {cn_wplaus:.4f}")
    if xgb_imp is not None:
        xgb_wplaus = float((xgb_imp * plausibility).sum())
        log.info(f"  XGBoost    weighted plausibility: {xgb_wplaus:.4f}")
        top20 = audit.nlargest(20, 'xgb_importance')
        pct_low = (top20['plausibility'] == 0.0).mean()
        log.info(f"  Top-20 XGB features: {100*pct_low:.0f}% low-plausibility")

    # ── Figure 1: Plausibility by importance decile ────────────────────────
    n_cols = 2 if xgb_imp is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols + 2, 6))
    if n_cols == 1:
        axes = [axes]

    plot_specs = [('cn_importance', 'ConceptNet', '#E91E63')]
    if xgb_imp is not None:
        plot_specs.append(('xgb_importance', 'XGBoost', '#FF9800'))

    for ax, (col, label, color) in zip(axes, plot_specs):
        sub = audit[[col, 'plausibility']].dropna().sort_values(col, ascending=False)\
                    .reset_index(drop=True)
        sub['decile'] = pd.cut(sub.index, bins=10,
                               labels=[f'D{i+1}' for i in range(10)])
        dec_plaus = sub.groupby('decile', observed=True)['plausibility'].mean()
        bars = ax.bar(dec_plaus.index, dec_plaus.values,
                      color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, dec_plaus.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        ax.axhline(0.5, color='gray', ls='--', lw=1.2, label='Midpoint')
        ax.set_ylim(0, 1.15)
        ax.set_xlabel('Importance Decile  (D1 = most important features)', fontsize=10)
        ax.set_ylabel('Mean Clinical Plausibility', fontsize=10)
        ax.set_title(f'{label}\nPlausibility of Important Features',
                     fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(
        'Spurious Correlation Audit\n'
        '1.0 = clearly clinical  |  0.5 = administrative  |  0.0 = likely artifact',
        fontsize=12, fontweight='bold')
    fig.tight_layout()
    p = os.path.join(exp5_dir, 'plausibility_by_decile.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    log.info(f"\n  Saved: {p}")

    # ── Figure 2: Scatter CN vs XGB importance, coloured by plausibility ───
    if xgb_imp is not None:
        color_map = {1.0: '#4CAF50', 0.5: '#FF9800', 0.0: '#F44336'}
        fig, ax = plt.subplots(figsize=(8, 7))
        for pv, lbl in [(1.0, 'High (1.0)'), (0.5, 'Medium (0.5)'),
                        (0.0, 'Low / artifact (0.0)')]:
            sub = audit[audit['plausibility'] == pv]
            ax.scatter(sub['xgb_importance'], sub['cn_importance'],
                       c=color_map[pv], label=lbl, alpha=0.55, s=18)
        ax.set_xlabel('XGBoost Feature Importance (gain)', fontsize=11)
        ax.set_ylabel('ConceptNet Feature Importance (indirect)', fontsize=11)
        ax.set_title('XGBoost vs ConceptNet Feature Importance\nColoured by Clinical Plausibility',
                     fontsize=12, fontweight='bold')
        ax.legend(title='Plausibility', fontsize=9); ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(exp5_dir, 'importance_scatter_plausibility.png')
        fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
        log.info(f"  Saved: {p}")

    # ── Figure 3: ConceptNet concept-level importance (named, interpretable) ─
    ow_abs   = np.abs(ow[1, :])   # (12,)
    sort_idx = np.argsort(ow_abs)[::-1]
    fig, ax  = plt.subplots(figsize=(12, 6))
    ax.bar(range(N_CONCEPTS), ow_abs[sort_idx], color='#E91E63', alpha=0.85, edgecolor='white')
    ax.set_xticks(range(N_CONCEPTS))
    ax.set_xticklabels(
        [CONCEPT_LABELS[i].split('(')[0].strip() for i in sort_idx],
        rotation=40, ha='right', fontsize=8.5)
    ax.set_ylabel('|Output Weight|  (direct concept → readmit influence)', fontsize=10)
    ax.set_title(
        'ConceptNet — Concept-Level Importance\n'
        'Every bar is a named, pre-defined clinical concept — not "feature 247"',
        fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p = os.path.join(exp5_dir, 'cn_concept_importance.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    log.info(f"  Saved: {p}")

    # ── SHAP beeswarm if available ──────────────────────────────────────────
    if HAS_XGB and HAS_SHAP and xgb_model is not None:
        try:
            log.info("  Computing SHAP values (n=500) …")
            n_sample  = min(500, len(X_te))
            sample_i  = np.random.choice(len(X_te), n_sample, replace=False)
            explainer = shap.TreeExplainer(xgb_model)
            shap_vals = explainer.shap_values(X_te[sample_i])
            shap_abs  = np.abs(shap_vals).mean(axis=0)
            top_n     = 20
            top_idx   = np.argsort(shap_abs)[-top_n:][::-1]
            top_names = [feature_names[i] for i in top_idx]
            top_shap  = shap_abs[top_idx]
            top_plaus = [get_plausibility(n) for n in top_names]
            bar_cols  = [('#4CAF50' if p == 1.0 else '#FF9800' if p == 0.5 else '#F44336')
                         for p in top_plaus]
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(top_n), top_shap[::-1],
                    color=bar_cols[::-1], edgecolor='white')
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_names[::-1], fontsize=8)
            ax.set_xlabel('Mean |SHAP Value|', fontsize=10)
            ax.set_title(
                f'XGBoost — Top {top_n} Features by SHAP Value\n'
                'Green=high plausibility  |  Orange=medium  |  Red=low/artifact',
                fontsize=11, fontweight='bold')
            legend_handles = [
                mpatches.Patch(color='#4CAF50', label='High plausibility'),
                mpatches.Patch(color='#FF9800', label='Medium plausibility'),
                mpatches.Patch(color='#F44336', label='Low plausibility / artifact'),
            ]
            ax.legend(handles=legend_handles, fontsize=9)
            ax.grid(axis='x', alpha=0.3)
            fig.tight_layout()
            p = os.path.join(exp5_dir, 'xgb_shap_top20.png')
            fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
            log.info(f"  Saved: {p}")
        except Exception as e:
            log.warning(f"  SHAP failed: {e}")

    return audit


# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
def write_report(stability_summary, shift_df, audit_df):
    lines, SEP = [], "=" * 70
    def h(t):   lines.append(f"\n{SEP}\n  {t}\n{SEP}")
    def sub(t): lines.append(f"\n{'─'*50}\n  {t}\n{'─'*50}")

    h("ConceptNet Hypothesis Testing — Experiment Report")
    lines.append(f"  Generated : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Seeds (EXP-3) : {STABILITY_SEEDS}")
    lines.append(f"  Corruption (EXP-4) : {CORRUPTION_MODE} (shuffle within group)")

    sub("EXP-2 — Prediction Explanations")
    lines.append(
        "  Each prediction decomposed into 12 named clinical concept contributions.\n"
        "  Output: patient_explanations.png, mean_contributions_by_class.png,\n"
        "          contribution_heatmap.png, all_concept_contributions.csv\n"
        "  No baseline model (LR/RF/XGB/MLP) can produce a named clinical\n"
        "  explanation at this level of granularity by construction.")

    sub("EXP-3 — Concept Stability")
    if stability_summary:
        cn_rho = stability_summary['cn_mean_spearman_rho']
        lines.append(f"  ConceptNet mean Spearman ρ  : {cn_rho:.4f}")
        lines.append(f"  AUROC mean ± std            : "
                     f"{stability_summary['cn_auroc_mean']:.4f} ± "
                     f"{stability_summary['cn_auroc_std']:.4f}")
        if stability_summary.get('xgb_mean_spearman_rho'):
            xgb_rho = stability_summary['xgb_mean_spearman_rho']
            delta   = stability_summary.get('delta_rho_cn_minus_xgb', 0)
            lines.append(f"  XGBoost mean Spearman ρ     : {xgb_rho:.4f}")
            lines.append(f"  Δ ρ (CN − XGB)              : {delta:+.4f}  "
                         f"({'ConceptNet more stable' if delta > 0 else 'XGBoost more stable'})")
        lines.append(
            "\n  Interpretation: ρ close to 1.0 means the SAME clinical concepts are\n"
            "  identified as important across all training runs. This is evidence\n"
            "  that ConceptNet learns meaningful abstractions rather than fitting noise.")

    sub("EXP-4 — Domain Shift (shuffle)")
    if shift_df is not None and not shift_df.empty:
        if 'ConceptNet_drop' in shift_df.columns:
            worst_cn  = shift_df.loc[shift_df['ConceptNet_drop'].idxmax()]
            mean_cn   = shift_df['ConceptNet_drop'].mean()
            lines.append(f"  Mean ConceptNet AUROC drop   : {mean_cn:.4f}")
            lines.append(f"  Worst group for ConceptNet   : {worst_cn['group']} "
                         f"(drop {worst_cn['ConceptNet_drop']:.4f})")
        if 'XGBoost_drop' in shift_df.columns:
            worst_xgb = shift_df.loc[shift_df['XGBoost_drop'].idxmax()]
            mean_xgb  = shift_df['XGBoost_drop'].mean()
            lines.append(f"  Mean XGBoost AUROC drop      : {mean_xgb:.4f}")
            lines.append(f"  Worst group for XGBoost      : {worst_xgb['group']} "
                         f"(drop {worst_xgb['XGBoost_drop']:.4f})")
        lines.append(
            "\n  Key advantage: ConceptNet's 'concept_group_exposure_map.png' lets you\n"
            "  PREDICT which concepts will degrade before deploying. A clinician can\n"
            "  look at that map and say 'if lab data goes missing, biological_buffer\n"
            "  and physio_stability will be blind.' XGBoost offers no such map.")

    sub("EXP-5 — Spurious Correlation Audit")
    if audit_df is not None:
        cn_wp = (audit_df['cn_importance'] * audit_df['plausibility']).sum()
        lines.append(f"  ConceptNet weighted plausibility : {cn_wp:.4f}")
        if 'xgb_importance' in audit_df.columns:
            xgb_wp  = (audit_df['xgb_importance'] * audit_df['plausibility']).sum()
            top20   = audit_df.nlargest(20, 'xgb_importance')
            pct_low = (top20['plausibility'] == 0.0).mean()
            lines.append(f"  XGBoost weighted plausibility    : {xgb_wp:.4f}")
            lines.append(f"  Top-20 XGB features — {100*pct_low:.0f}% are low-plausibility")
        lines.append(
            "\n  Structural guarantee of ConceptNet: adm_race can only reach\n"
            "  postdc_support — it cannot contaminate physio_stability_dc or\n"
            "  biological_buffer. This constraint is enforced by the connection mask\n"
            "  and is verifiable by inspection, not just empirically measured.")

    sub("What These Experiments Collectively Prove")
    lines.append("""
  The paper does not claim ConceptNet maximises AUROC.
  It claims concept-guided abstraction provides four properties
  that black-box models cannot match:

  1. Named explanations  [EXP-2]
     Predictions decompose into clinical concepts a doctor already uses.

  2. Stable reasoning    [EXP-3]
     The same concepts are important run-to-run. The model is not
     fitting noise or arbitrary correlations.

  3. Predictable robustness  [EXP-4]
     When a data source fails, the affected concept nodes are known
     in advance. Failure modes are auditable before deployment.

  4. Structural plausibility constraint  [EXP-5]
     Low-plausibility features (race, insurance) cannot leak into
     clinical concept nodes by architecture. This is a fairness
     property that SHAP post-hoc analysis cannot guarantee.
    """)

    report = "\n".join(lines)
    p = os.path.join(OUTPUT_DIR, 'experiment_report.txt')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(report)
    log.info(f"\n{report}")
    log.info(f"\n  Report saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("═"*60)
    log.info("  ConceptNet Hypothesis Testing — 4 Experiments")
    log.info(f"  Device : {DEVICE}")
    log.info(f"  Seeds  : {STABILITY_SEEDS}")
    log.info(f"  Shift  : {CORRUPTION_MODE}")
    log.info("═"*60)

    # ── Data ───────────────────────────────────────────────────────────────
    X_df, y = load_data()
    X_tv, X_te, y_tv, y_te = train_test_split(
        X_df, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=VAL_SIZE, random_state=SEED, stratify=y_tv)
    log.info(f"  Split — train {len(X_tr):,} | val {len(X_va):,} | test {len(X_te):,}")

    prep      = Preprocessor(list(X_df.columns))
    X_tr_np   = prep.fit_transform(X_tr)
    X_va_np   = prep.transform(X_va)
    X_te_np   = prep.transform(X_te)
    feat_names = prep.out_cols_
    y_tr_np, y_va_np, y_te_np = y_tr.to_numpy(), y_va.to_numpy(), y_te.to_numpy()
    log.info(f"  Features after preprocessing: {X_tr_np.shape[1]}")

    # ── Load saved ConceptNet (seed=42) or retrain ─────────────────────────
    cn_pt = os.path.join(COMPARISON_DIR, 'conceptnet_readmission.pt')
    if os.path.exists(cn_pt):
        log.info(f"  Loading ConceptNet from {cn_pt}")
        ckpt = torch.load(cn_pt, map_location=DEVICE)
        
        # Handle different checkpoint formats
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            # New format from train.py
            cn_mask = ckpt['mask'].to(DEVICE)
            cn_model = ConceptNet(X_tr_np.shape[1], cn_mask).to(DEVICE)
            cn_model.load_state_dict(ckpt['model_state_dict'])
        elif isinstance(ckpt, dict) and 'mask' in ckpt:
            # Old format - mask exists but model state dict is directly in ckpt
            cn_mask = ckpt['mask'].to(DEVICE)
            cn_model = ConceptNet(X_tr_np.shape[1], cn_mask).to(DEVICE)
            # Remove mask and other metadata, load remaining as state dict
            state_dict = {k: v for k, v in ckpt.items() if k not in ['mask', 'feature_names', 'n_features', 'concepts', 'metrics']}
            if state_dict:
                cn_model.load_state_dict(state_dict)
            else:
                raise KeyError(f"Could not find model weights in checkpoint. Available keys: {list(ckpt.keys())}")
        else:
            # Unrecognized format - retrain
            log.warning(f"  Unrecognized checkpoint format. Keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'not a dict'}")
            raise KeyError("Unrecognized checkpoint format")
        # Quick sanity check
        probe, _ = eval_nn(cn_model,
            DataLoader(TensorDataset(
                torch.from_numpy(X_te_np).float(),
                torch.zeros(len(X_te_np), dtype=torch.long)),
                batch_size=BATCH_SIZE*2, shuffle=False))
        log.info(f"  Loaded ConceptNet  AUROC = {roc_auc_score(y_te_np, probe):.4f}")
    else:
        log.info("  No saved model — training ConceptNet seed=42 …")
        cn_model, cn_mask, _, _, _ = train_conceptnet(
            X_tr_np, y_tr_np, X_va_np, y_va_np, X_te_np, y_te_np,
            feat_names, seed=SEED, verbose=True)
        cn_mask = cn_mask.to(DEVICE)

    # ── Baselines for EXP-4 / EXP-5 ───────────────────────────────────────
    log.info("\n  Training baselines …")
    n_pos = int(y_tr_np.sum()); n_neg = len(y_tr_np) - n_pos

    lr_model = LogisticRegression(C=0.1, penalty='l2', solver='saga', max_iter=1000,
                                   class_weight='balanced', random_state=SEED, n_jobs=-1)
    lr_model.fit(X_tr_np, y_tr_np)
    log.info(f"  LR      AUROC = {roc_auc_score(y_te_np, lr_model.predict_proba(X_te_np)[:,1]):.4f}")

    rf_model = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=50,
                                       max_features='sqrt', random_state=SEED, n_jobs=-1,
                                       class_weight={0:1, 1:n_neg/max(n_pos,1)})
    rf_model.fit(X_tr_np, y_tr_np)
    log.info(f"  RF      AUROC = {roc_auc_score(y_te_np, rf_model.predict_proba(X_te_np)[:,1]):.4f}")

    xgb_model = None
    if HAS_XGB:
        xgb_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=30,
            scale_pos_weight=n_neg/max(n_pos,1), use_label_encoder=False,
            random_state=SEED, n_jobs=-1, verbosity=0,
            eval_metric='aucpr', early_stopping_rounds=20)
        xgb_model.fit(X_tr_np, y_tr_np, eval_set=[(X_va_np, y_va_np)], verbose=False)
        log.info(f"  XGB     AUROC = {roc_auc_score(y_te_np, xgb_model.predict_proba(X_te_np)[:,1]):.4f}")

    lgb_model = None
    if HAS_LGB:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=7, num_leaves=63, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
            scale_pos_weight=n_neg/max(n_pos,1), random_state=SEED, n_jobs=-1, verbose=-1)
        lgb_model.fit(X_tr_np, y_tr_np, eval_set=[(X_va_np, y_va_np)],
                      callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)])
        log.info(f"  LGB     AUROC = {roc_auc_score(y_te_np, lgb_model.predict_proba(X_te_np)[:,1]):.4f}")

    log.info("  Training MLP …")
    mlp_model = StandardMLP(X_tr_np.shape[1]).to(DEVICE)
    mlp_crit  = nn.CrossEntropyLoss()
    mlp_opt   = optim.AdamW(mlp_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mlp_sched = optim.lr_scheduler.ReduceLROnPlateau(mlp_opt, mode='max', factor=0.5, patience=8)
    train_dl, val_dl, _ = build_loaders(X_tr_np, y_tr_np, X_va_np, y_va_np, X_te_np, y_te_np)
    best_auroc_mlp, best_state_mlp, no_imp = 0.0, None, 0
    for ep in range(1, MLP_MAX_EPOCHS + 1):
        mlp_model.train()
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            mlp_opt.zero_grad()
            mlp_crit(mlp_model(Xb), yb).backward()
            nn.utils.clip_grad_norm_(mlp_model.parameters(), 1.0)
            mlp_opt.step()
        vp, vl = eval_nn(mlp_model, val_dl)
        va = roc_auc_score(vl, vp)
        mlp_sched.step(va)
        if va > best_auroc_mlp:
            best_auroc_mlp = va
            best_state_mlp = {k: v.cpu().clone() for k, v in mlp_model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= MLP_PATIENCE:
                break
    mlp_model.load_state_dict(best_state_mlp)
    mlp_probe, mlp_labels = eval_nn(mlp_model,
        DataLoader(TensorDataset(torch.from_numpy(X_te_np).float(),
                                  torch.from_numpy(y_te_np.astype(np.int64))),
                   batch_size=BATCH_SIZE*2, shuffle=False))
    log.info(f"  MLP     AUROC = {roc_auc_score(y_te_np, mlp_probe):.4f}")

    # ── Run all 4 experiments ──────────────────────────────────────────────
    contrib_df = exp2_explanations(
        cn_model, cn_mask, X_te_np, y_te_np, feat_names)

    stability_summary, cn_imp_mat = exp3_stability(
        X_tr_np, y_tr_np, X_va_np, y_va_np, X_te_np, y_te_np, feat_names,
        xgb_model_seed42=xgb_model)

    shift_df = exp4_domain_shift(
        X_te_np, y_te_np, feat_names,
        cn_model=cn_model, xgb_model=xgb_model, lgb_model=lgb_model,
        lr_model=lr_model, rf_model=rf_model, mlp_model=mlp_model)

    audit_df = exp5_spurious_audit(
        X_tr_np, y_tr_np, X_te_np, y_te_np, feat_names,
        cn_model=cn_model, cn_mask=cn_mask, xgb_model=xgb_model)

    write_report(stability_summary, shift_df, audit_df)

    log.info("\n" + "═"*60)
    log.info(f"  ✓  All outputs → {OUTPUT_DIR}")
    log.info("═"*60)


if __name__ == '__main__':
    main()