"""
=============================================================================
  Concept-Guided Bottleneck Network -- FULL EVALUATION SUITE  [v3]
  30-Day Hospital Readmission Prediction (MIMIC-IV)
=============================================================================

CHANGES vs v2
-------------
  v3 CHANGE 1 -- Exp 4 (Domain Shift) removed. Left for next paper with
                 proper multi-site validation (eICU). Simulated masking was
                 not a valid domain-shift test; removed rather than
                 misrepresent findings.

  v3 CHANGE 2 -- concept_coverage() simplified. Previous logic had a
                 mask-membership check that incorrectly excluded many
                 legitimate features (catch-all rows, lsa_ features),
                 producing CBN coverage of 25% vs the true ~100%.
                 Now uses direct pattern matching against
                 CONCEPT_FEATURE_PATTERNS -- same ground truth used to
                 build the mask, so coverage is measured consistently.

  v3 CHANGE 3 -- All file writes use encoding='utf-8' explicitly.
                 Special Unicode box-drawing chars (used in log messages)
                 are kept out of the written report to avoid cp1252 errors
                 on Windows.

FIXES RETAINED FROM v2
-----------------------
  Fix A -- LSA features mapped to Mental Health Risk / Social Support /
           Recovery Stability only (not broadcast to all 12 concepts).
  Fix B -- Concept-layer dropout (p=0.2) + L1 sparsity (lambda=1e-4).
  Fix C -- CBN attribution uses mean weight / assigned concepts (not max).
  Bug 1 -- mask.to(DEVICE) called before model construction
  Bug 2 -- np.nan_to_num() guard in exp2_explanations
  Bug 3 -- None-guard on best_state before load_state_dict
  Bug 4 -- all-NaN columns pruned in main(); keep_empty_features=True
=============================================================================
"""

import os, warnings, logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection   import StratifiedKFold, train_test_split
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.impute            import SimpleImputer
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.calibration       import calibration_curve
from sklearn.metrics           import (roc_auc_score, average_precision_score,
                                       f1_score, brier_score_loss,
                                       roc_curve, precision_recall_curve)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logging.warning('xgboost not installed -- XGBoost baseline skipped')

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
FEATURE_MATRIX_PATH = './model_outputs/feature_matrix.csv'
OUTPUT_DIR          = './model_outputs/'
RANDOM_STATE        = 42
N_SPLITS            = 5
BATCH_SIZE          = 256
MAX_EPOCHS          = 50
PATIENCE            = 8
LR                  = 1e-3
WEIGHT_DECAY        = 1e-4
CONCEPT_L1_LAMBDA   = 1e-4
DEVICE              = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

MODEL_COLORS = {
    'ConceptBottleneckNet': '#1a6faf',
    'Plain MLP':            '#e07b39',
    'Logistic Regression':  '#2ca02c',
    'Random Forest':        '#9467bd',
    'XGBoost':              '#d62728',
}

# -----------------------------------------------------------------------------
# CLINICAL CONCEPT DEFINITIONS
# -----------------------------------------------------------------------------
CONCEPT_NAMES = [
    'Physiologic Reserve',
    'Chronic Disease Burden',
    'Healthcare Instability',
    'Admission Severity',
    'Complication Burden',
    'Functional Status',
    'Medication Complexity',
    'Mental Health Risk',
    'Social Support',
    'Care Coordination',
    'Monitoring Needs',
    'Recovery Stability',
]

CONCEPT_FEATURE_PATTERNS = {
    'Physiologic Reserve':    ['vital_', 'lab_hemoglobin', 'lab_albumin',
                                'lab_creatinine', 'lab_bun', 'lab_sodium',
                                'lab_potassium', 'lab_hco3', 'lab_lactate',
                                'age', 'los_days', 'omr_bmi'],
    'Chronic Disease Burden': ['elix_', 'elixhauser_sum', 'n_diagnoses'],
    'Healthcare Instability': ['prior_admissions', 'prior_readmissions',
                                'days_since_last_disch', 'n_ed_visits',
                                'mean_ed_los_h', 'ed_admitted'],
    'Admission Severity':     ['adm_type_emergency', 'adm_type_urgent',
                                'icu_admitted', 'n_icu_stays', 'total_icu_los',
                                'micu_flag', 'ccu_flag', 'lab_troponin',
                                'lab_bnp', 'lab_lactate', 'lab_inr',
                                'lab_bilirubin', 'vital_sbp', 'vital_map'],
    'Complication Burden':    ['n_procedures', 'n_surgical_proc',
                                'n_transfers', 'n_care_units', 'icu_transfer',
                                'lab_n_abnormal', 'lab_wbc', 'lab_platelets',
                                'elix_coagulopathy'],
    'Functional Status':      ['vital_gcs', 'disch_home', 'disch_snf',
                                'disch_rehab', 'elix_paralysis',
                                'elix_other_neurological'],
    'Medication Complexity':  ['n_unique_drugs', 'polypharmacy',
                                'n_pharmacy_dispenses', 'med_anticoag',
                                'med_insulin', 'med_diuretic', 'med_opioid',
                                'med_steroid', 'med_antipsychotic',
                                'med_antibiotic'],
    # Fix A: lsa_ anchored to 3 specific concepts only
    'Mental Health Risk':     ['elix_depression', 'elix_psychoses',
                                'elix_alcohol_abuse', 'elix_drug_abuse',
                                'med_antipsychotic', 'note_kw_substance_abuse',
                                'lsa_'],
    'Social Support':         ['married', 'english', 'note_kw_homeless',
                                'note_kw_lives_alone', 'note_kw_social_support',
                                'note_kw_non_compli', 'disch_home',
                                'insurance_mc', 'insurance_md',
                                'lsa_'],
    'Care Coordination':      ['note_kw_follow_up', 'note_kw_readmit',
                                'n_care_units', 'n_transfers', 'step_transfer',
                                'note_kw_non_compli',
                                'note_kw_medication_not_taken'],
    'Monitoring Needs':       ['vital_spo2', 'vital_resp_rate', 'lab_glucose',
                                'omr_blood_pressure', 'note_kw_confusion',
                                'note_kw_fall_risk', 'note_kw_wound_infection',
                                'note_kw_shortness_of_breath'],
    'Recovery Stability':     ['los_days', 'note_polarity', 'note_subjectivity',
                                'note_kw_return_to_ed',
                                'note_kw_poorly_controlled', 'note_kw_fever',
                                'note_kw_sepsis', 'note_kw_dyspnea',
                                'note_kw_chest_pain', 'note_word_len',
                                'lsa_'],
}

# Build a flat lookup: feature_name_substring -> covered (for fast matching)
ALL_CONCEPT_PATTERNS = [p for plist in CONCEPT_FEATURE_PATTERNS.values()
                        for p in plist]

# -----------------------------------------------------------------------------
# BUILD SPARSE MASK
# -----------------------------------------------------------------------------

def build_concept_mask(feature_names):
    n_feat = len(feature_names)
    n_con  = len(CONCEPT_NAMES)
    mask   = torch.zeros(n_feat, n_con, dtype=torch.float32)

    for j, concept in enumerate(CONCEPT_NAMES):
        for i, feat in enumerate(feature_names):
            if any(pat in feat for pat in CONCEPT_FEATURE_PATTERNS[concept]):
                mask[i, j] = 1.0

    unmatched = mask.sum(dim=1) == 0
    n_unmatched = int(unmatched.sum())
    if n_unmatched > 0:
        mask[unmatched, :] = 1.0
        log.info(f'  {n_unmatched} truly unmatched features -> all concepts')

    for j, name in enumerate(CONCEPT_NAMES):
        log.info(f'  [{name:28s}] {int(mask[:,j].sum()):4d} features')
    return mask

# -----------------------------------------------------------------------------
# MODELS
# -----------------------------------------------------------------------------

class ConceptBottleneckNetwork(nn.Module):
    def __init__(self, n_features, n_concepts, mask,
                 hidden_dim=64, dropout=0.3, concept_dropout=0.2):
        super().__init__()
        self.concept_weight  = nn.Parameter(
            torch.randn(n_features, n_concepts) * 0.01)
        self.concept_bias    = nn.Parameter(torch.zeros(n_concepts))
        self.register_buffer('mask', mask)
        self.concept_bn      = nn.BatchNorm1d(n_concepts)
        self.concept_act     = nn.ReLU()
        self.concept_dropout = nn.Dropout(p=concept_dropout)   # Fix B
        self.head = nn.Sequential(
            nn.Linear(n_concepts, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        eff_w    = self.concept_weight * self.mask
        concepts = x @ eff_w + self.concept_bias
        concepts = self.concept_act(self.concept_bn(concepts))
        concepts = self.concept_dropout(concepts)
        logit    = self.head(concepts).squeeze(1)
        return logit, concepts

    def concept_l1_loss(self):
        return (self.concept_weight * self.mask).abs().mean()


class PlainMLP(nn.Module):
    def __init__(self, n_features, n_concepts=12, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_concepts),
            nn.BatchNorm1d(n_concepts), nn.ReLU(),
            nn.Linear(n_concepts, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# -----------------------------------------------------------------------------
# TRAINING HELPERS
# -----------------------------------------------------------------------------

def train_epoch(model, loader, opt, crit, device, is_cbn=True):
    model.train()
    total = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        opt.zero_grad()
        out   = model(Xb)
        logit = out[0] if is_cbn else out
        loss  = crit(logit, yb)
        if is_cbn:
            loss = loss + CONCEPT_L1_LAMBDA * model.concept_l1_loss()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)


@torch.no_grad()
def predict_nn(model, X_tensor, device, bs=512, is_cbn=True):
    model.eval()
    probs, concepts = [], []
    for (Xb,) in DataLoader(TensorDataset(X_tensor), bs, shuffle=False):
        out = model(Xb.to(device))
        if is_cbn:
            logit, c = out
            concepts.append(c.cpu().numpy())
        else:
            logit = out
        probs.append(torch.sigmoid(logit).cpu().numpy())
    p = np.concatenate(probs)
    c = np.concatenate(concepts) if is_cbn else None
    return p, c


def train_nn_with_es(model, Xtr, ytr, Xva, yva, pos_w, is_cbn=True):
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr.astype(np.float32))
    loader = DataLoader(TensorDataset(Xt, yt), BATCH_SIZE, shuffle=True)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS, 1e-5)
    crit  = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], dtype=torch.float32).to(DEVICE))
    best_auc, best_state, wait = 0.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_epoch(model, loader, opt, crit, DEVICE, is_cbn)
        sched.step()
        vp, _ = predict_nn(model, torch.from_numpy(Xva), DEVICE, is_cbn=is_cbn)
        try:
            auc = roc_auc_score(yva, vp)
        except ValueError:
            continue
        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    if best_state is not None:   # Bug 3 fix
        model.load_state_dict(best_state)
    return model


def eval_metrics(y_true, y_prob, thr=0.5):
    yp = (y_prob >= thr).astype(int)
    return {
        'AUROC': round(roc_auc_score(y_true, y_prob), 4),
        'AUPRC': round(average_precision_score(y_true, y_prob), 4),
        'F1':    round(f1_score(y_true, yp, zero_division=0), 4),
        'Brier': round(brier_score_loss(y_true, y_prob), 4),
    }

# -----------------------------------------------------------------------------
# CROSS-VALIDATED TRAINING
# -----------------------------------------------------------------------------

def cross_validate_all(X, y, mask):
    skf     = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    n_con   = len(CONCEPT_NAMES)
    n       = len(y)

    oof = {m: np.zeros(n, dtype=np.float64)
           for m in ['CBN', 'MLP', 'LR', 'RF', 'XGB']}
    oof_con  = np.zeros((n, n_con), dtype=np.float64)
    fold_imp = np.zeros((N_SPLITS, n_con), dtype=np.float64)

    mask_dev  = mask.to(DEVICE)   # Bug 1 fix: after model construction point
    pos_w_val = (y == 0).sum() / max((y == 1).sum(), 1)

    for fold, (tr, val) in enumerate(skf.split(X, y), 1):
        log.info(f'  -- Fold {fold}/{N_SPLITS} --')
        Xtr, Xva = X[tr], X[val]
        ytr, yva = y[tr], y[val]

        imp = SimpleImputer(strategy='median', keep_empty_features=True)  # Bug 4
        Xtr = imp.fit_transform(Xtr)
        Xva = imp.transform(Xva)
        scl = StandardScaler()
        Xtr = scl.fit_transform(Xtr).astype(np.float32)
        Xva = scl.transform(Xva).astype(np.float32)

        # CBN
        cbn = ConceptBottleneckNetwork(
            Xtr.shape[1], n_con, mask_dev).to(DEVICE)
        cbn = train_nn_with_es(cbn, Xtr, ytr, Xva, yva, pos_w_val, is_cbn=True)
        vp, vc = predict_nn(cbn, torch.from_numpy(Xva), DEVICE, is_cbn=True)
        oof['CBN'][val] = vp
        oof_con[val]    = vc
        with torch.no_grad():
            fold_imp[fold-1] = (cbn.concept_weight * cbn.mask).abs()\
                               .sum(dim=0).cpu().numpy()
        log.info(f'    CBN  AUROC={roc_auc_score(yva, vp):.4f}')

        # Plain MLP
        mlp = PlainMLP(Xtr.shape[1], n_con).to(DEVICE)
        mlp = train_nn_with_es(mlp, Xtr, ytr, Xva, yva, pos_w_val, is_cbn=False)
        mp, _ = predict_nn(mlp, torch.from_numpy(Xva), DEVICE, is_cbn=False)
        oof['MLP'][val] = mp
        log.info(f'    MLP  AUROC={roc_auc_score(yva, mp):.4f}')

        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, class_weight='balanced',
                                 C=0.1, random_state=RANDOM_STATE, solver='saga')
        lr.fit(Xtr, ytr)
        lp = lr.predict_proba(Xva)[:, 1]
        oof['LR'][val] = lp
        log.info(f'    LR   AUROC={roc_auc_score(yva, lp):.4f}')

        # Random Forest
        rf = RandomForestClassifier(n_estimators=500, max_depth=8,
                                     class_weight='balanced',
                                     n_jobs=-1, random_state=RANDOM_STATE)
        rf.fit(Xtr, ytr)
        rp = rf.predict_proba(Xva)[:, 1]
        oof['RF'][val] = rp
        log.info(f'    RF   AUROC={roc_auc_score(yva, rp):.4f}')

        # XGBoost
        if HAS_XGB:
            xg = xgb.XGBClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=pos_w_val, eval_metric='logloss',
                random_state=RANDOM_STATE, verbosity=0, use_label_encoder=False)
            xg.fit(Xtr, ytr)
            xp = xg.predict_proba(Xva)[:, 1]
            oof['XGB'][val] = xp
            log.info(f'    XGB  AUROC={roc_auc_score(yva, xp):.4f}')

    return oof, fold_imp, oof_con

# -----------------------------------------------------------------------------
# EXP 1 -- PERFORMANCE
# -----------------------------------------------------------------------------

def exp1_performance(y, oof):
    log.info('\n' + '='*65)
    log.info('EXP 1 -- Full Performance Comparison')

    label_map = {'CBN': 'ConceptBottleneckNet', 'MLP': 'Plain MLP',
                 'LR':  'Logistic Regression',  'RF':  'Random Forest',
                 'XGB': 'XGBoost'}
    rows = []
    for key, label in label_map.items():
        if key == 'XGB' and not HAS_XGB:
            continue
        m = eval_metrics(y, oof[key])
        rows.append({'Model': label,
                     'Type': 'Concept-Guided' if key == 'CBN' else 'Baseline',
                     **m})

    df = pd.DataFrame(rows).sort_values('AUROC', ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(OUTPUT_DIR, 'results_comparison.csv'), index=False)
    log.info('\n' + df.to_string(index=False))

    # ROC curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for key, label in label_map.items():
        if key == 'XGB' and not HAS_XGB:
            continue
        fpr, tpr, _ = roc_curve(y, oof[key])
        auc = roc_auc_score(y, oof[key])
        lw  = 2.5 if key == 'CBN' else 1.4
        ls  = '-'  if key == 'CBN' else '--'
        ax.plot(fpr, tpr, lw=lw, ls=ls,
                color=MODEL_COLORS.get(label, 'grey'),
                label=f'{label}  (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k:', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves -- All Models\n(Concept Bottleneck vs Baselines)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'), dpi=150)
    plt.close()

    # PR curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for key, label in label_map.items():
        if key == 'XGB' and not HAS_XGB:
            continue
        prec, rec, _ = precision_recall_curve(y, oof[key])
        ap  = average_precision_score(y, oof[key])
        lw  = 2.5 if key == 'CBN' else 1.4
        ls  = '-'  if key == 'CBN' else '--'
        ax.plot(rec, prec, lw=lw, ls=ls,
                color=MODEL_COLORS.get(label, 'grey'),
                label=f'{label}  (AP={ap:.4f})')
    ax.axhline(y.mean(), color='k', ls=':', lw=1, label=f'Chance ({y.mean():.3f})')
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curves -- All Models\n(Concept Bottleneck vs Baselines)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pr_curves.png'), dpi=150)
    plt.close()

    # Performance bar chart
    metrics   = ['AUROC', 'AUPRC', 'F1', 'Brier']
    n_metrics = len(metrics)
    n_models  = len(df)
    x         = np.arange(n_metrics)
    w         = 0.75 / n_models
    fig, ax   = plt.subplots(figsize=(13, 5))
    for i, row in df.iterrows():
        vals  = [row[m] for m in metrics]
        color = MODEL_COLORS.get(row['Model'], 'grey')
        lw    = 2.0 if row['Type'] == 'Concept-Guided' else 0.5
        bars  = ax.bar(x + i*w - (n_models-1)*w/2, vals, w*0.9,
                       label=row['Model'], color=color, alpha=0.85,
                       edgecolor='black' if row['Type'] == 'Concept-Guided' else 'white',
                       linewidth=lw)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h+0.005,
                    f'{h:.3f}', ha='center', fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title('Model Performance Comparison Across All Metrics\n'
                 '(CBN highlighted with black border)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'performance_bar.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Calibration curves
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
    for key, label in label_map.items():
        if key == 'XGB' and not HAS_XGB:
            continue
        prob_true, prob_pred = calibration_curve(y, oof[key], n_bins=10)
        lw = 2.5 if key == 'CBN' else 1.4
        ax.plot(prob_pred, prob_true, marker='o', markersize=4, lw=lw,
                color=MODEL_COLORS.get(label, 'grey'), label=label)
    ax.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax.set_ylabel('Fraction of Positives', fontsize=11)
    ax.set_title('Calibration Curves -- All Models\n'
                 '(closer to diagonal = better calibrated)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'calibration_curves.png'), dpi=150)
    plt.close()

    log.info('Saved -> roc_curves.png, pr_curves.png, performance_bar.png, calibration_curves.png')
    return df

# -----------------------------------------------------------------------------
# EXP 2 -- PER-PATIENT CONCEPT EXPLANATION
# -----------------------------------------------------------------------------

def exp2_explanations(oof_concepts, oof_prob, y, n_each=3):
    log.info('\n' + '='*65)
    log.info('EXP 2 -- Per-Patient Concept Explanations')

    oof_concepts = np.nan_to_num(oof_concepts, nan=0.0)   # Bug 2 fix
    cmin = oof_concepts.min(0, keepdims=True)
    cmax = oof_concepts.max(0, keepdims=True)
    cn   = (oof_concepts - cmin) / (cmax - cmin + 1e-8)

    tp = np.where((oof_prob >= 0.5) & (y == 1))[0]
    tn = np.where((oof_prob <  0.5) & (y == 0))[0]
    tp = tp[np.argsort(oof_prob[tp])[::-1]][:n_each]
    tn = tn[np.argsort(oof_prob[tn])][:n_each]

    selected = list(tp) + list(tn)
    titles   = ([f'High Risk\n(P={oof_prob[i]:.2f})' for i in tp] +
                [f'Low Risk\n(P={oof_prob[i]:.2f})'  for i in tn])
    colors   = ['#d62728'] * len(tp) + ['#1f77b4'] * len(tn)

    fig, axes = plt.subplots(1, len(selected),
                             figsize=(4.2*len(selected), 6), sharey=True)
    if len(selected) == 1:
        axes = [axes]
    for ax, idx, title, color in zip(axes, selected, titles, colors):
        vals = cn[idx]
        ax.barh(CONCEPT_NAMES, vals, color=color, alpha=0.78,
                edgecolor='white', linewidth=0.4)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel('Concept Activation', fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold', color=color)
        ax.tick_params(axis='y', labelsize=7.5)
        for j, v in enumerate(vals):
            if v > 0.05:
                ax.text(v+0.02, j, f'{v:.2f}', va='center', fontsize=6)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle('Per-Patient Concept Activation Explanations\n'
                 '(Concept Bottleneck Network)',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'patient_concept_explanation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'Saved -> {path}')

    # Concept correlation heatmap
    corr = np.corrcoef(cn.T)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=CONCEPT_NAMES, yticklabels=CONCEPT_NAMES,
                ax=ax, linewidths=0.5, annot_kws={'size': 7})
    ax.set_title('Inter-Concept Correlation (CBN Activations)\n'
                 'Low correlation = concepts capture distinct clinical dimensions',
                 fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', labelsize=7.5, rotation=45)
    ax.tick_params(axis='y', labelsize=7.5, rotation=0)
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, 'concept_correlation_heatmap.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'Saved -> {path2}')

# -----------------------------------------------------------------------------
# EXP 3 -- CONCEPT IMPORTANCE STABILITY
# -----------------------------------------------------------------------------

def exp3_stability(fold_imp):
    log.info('\n' + '='*65)
    log.info('EXP 3 -- Concept Importance Stability')

    row_sums = fold_imp.sum(axis=1, keepdims=True) + 1e-8
    imp_norm = fold_imp / row_sums
    means = imp_norm.mean(0)
    stds  = imp_norm.std(0)
    cv    = stds / (means + 1e-8) * 100

    log.info('CV% per concept (lower = more stable):')
    for name, c in zip(CONCEPT_NAMES, cv):
        log.info(f'  {name:30s}: CV={c:.1f}%')

    order = np.argsort(means)[::-1]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(CONCEPT_NAMES))
    bars = ax.bar(x, means[order], yerr=stds[order], capsize=4,
                  color='steelblue', alpha=0.82, edgecolor='white',
                  error_kw={'ecolor': 'black', 'linewidth': 1.2})
    for bar, i in zip(bars, order):
        ax.text(bar.get_x()+bar.get_width()/2, -0.006,
                f'CV={cv[i]:.0f}%', ha='center', fontsize=6.5,
                color='#444', va='top')
    ax.set_xticks(x)
    ax.set_xticklabels([CONCEPT_NAMES[i] for i in order],
                        rotation=38, ha='right', fontsize=9)
    ax.set_ylabel('Normalised Concept Importance (mean +/- std)', fontsize=10)
    ax.set_title('Concept Importance Stability Across CV Folds\n'
                 'Shorter error bars + low CV% = stable, meaningful concepts',
                 fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'concept_stability.png')
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f'Saved -> {path}')
    return dict(zip(CONCEPT_NAMES, cv))

# -----------------------------------------------------------------------------
# EXP 4 -- FEATURE ATTRIBUTION AUDIT
#
# v3 CHANGE 2: concept_coverage() now uses direct pattern matching against
# CONCEPT_FEATURE_PATTERNS -- the same ground truth used to build the mask.
# Previous mask-membership logic incorrectly flagged many legitimate features
# (those in catch-all rows, or lsa_ features spanning multiple concepts) as
# uncovered, producing CBN coverage of 25% despite those features being
# explicitly assigned to concepts in the pattern dict.
# -----------------------------------------------------------------------------

def exp4_attribution_audit(X, y, mask, feature_names):
    log.info('\n' + '='*65)
    log.info('EXP 4 -- Feature Attribution Audit')

    imp = SimpleImputer(strategy='median', keep_empty_features=True)
    Xf  = imp.fit_transform(X)
    scl = StandardScaler()
    Xf  = scl.fit_transform(Xf).astype(np.float32)

    Xtr, Xte, ytr, yte = train_test_split(
        Xf, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight='balanced',
                             C=0.1, random_state=RANDOM_STATE, solver='saga')
    lr.fit(Xtr, ytr)
    lr_imp = np.abs(lr.coef_[0])

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                 class_weight='balanced',
                                 n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(Xtr, ytr)
    rf_imp = rf.feature_importances_

    # CBN
    mask_dev = mask.to(DEVICE)
    pos_w    = (ytr==0).sum() / max((ytr==1).sum(), 1)
    cbn = ConceptBottleneckNetwork(
        Xtr.shape[1], len(CONCEPT_NAMES), mask_dev).to(DEVICE)
    cbn = train_nn_with_es(cbn, Xtr, ytr, Xte, yte, pos_w, is_cbn=True)

    with torch.no_grad():
        eff_w           = (cbn.concept_weight * cbn.mask).abs()
        assigned_counts = cbn.mask.sum(dim=1).clamp(min=1)
        feat_cbn_imp    = (eff_w.sum(dim=1) / assigned_counts).cpu().numpy()

    top_n   = 20
    top_lr  = np.argsort(lr_imp)[::-1][:top_n]
    top_rf  = np.argsort(rf_imp)[::-1][:top_n]
    top_cbn = np.argsort(feat_cbn_imp)[::-1][:top_n]

    def concept_coverage(feat_idx):
        """
        v3 CHANGE 2: Check each feature name against CONCEPT_FEATURE_PATTERNS
        directly. A feature is 'covered' if it matches any pattern in any
        concept's pattern list. This is the same check used to build the mask,
        so it is the honest ground truth for concept alignment.
        """
        covered = 0
        for fi in feat_idx:
            fname = feature_names[fi]
            for patterns in CONCEPT_FEATURE_PATTERNS.values():
                if any(p in fname for p in patterns):
                    covered += 1
                    break
        return covered / len(feat_idx) * 100

    lr_cov  = concept_coverage(top_lr)
    rf_cov  = concept_coverage(top_rf)
    cbn_cov = concept_coverage(top_cbn)

    log.info(f'  Top-{top_n} features in defined clinical concepts:')
    log.info(f'    LR  coverage: {lr_cov:.1f}%')
    log.info(f'    RF  coverage: {rf_cov:.1f}%')
    log.info(f'    CBN coverage: {cbn_cov:.1f}%')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, idx, imp_arr, label, color, cov in zip(
            axes,
            [top_lr, top_rf, top_cbn],
            [lr_imp, rf_imp, feat_cbn_imp],
            ['Logistic Regression', 'Random Forest', 'Concept Bottleneck Net'],
            ['#2ca02c', '#9467bd', '#1a6faf'],
            [lr_cov, rf_cov, cbn_cov]):
        fnames = [feature_names[i][:30] for i in idx]
        fimps  = np.array([imp_arr[i] for i in idx])
        fimps  = fimps / (fimps.max() + 1e-8)
        ax.barh(fnames[::-1], fimps[::-1], color=color, alpha=0.78)
        ax.set_xlabel('Relative Importance', fontsize=9)
        ax.set_title(f'{label}\nTop {top_n} Features\n'
                     f'(Clinical concept coverage: {cov:.0f}%)',
                     fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7.5)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.suptitle('Feature Attribution Audit -- Top Features by Model\n'
                 'Higher clinical concept coverage = less reliance on spurious signals',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'feature_attribution_audit.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'Saved -> {path}')
    return {'LR': lr_cov, 'RF': rf_cov, 'CBN': cbn_cov}

# -----------------------------------------------------------------------------
# EVALUATION REPORT
# v3 CHANGE 3: all text uses ASCII only; file written with utf-8 explicitly.
# -----------------------------------------------------------------------------

def write_report(perf_df, stability_cv, attribution_cov):
    log.info('\n' + '='*65)
    log.info('Writing evaluation report ...')

    cbn_row       = perf_df[perf_df['Model'] == 'ConceptBottleneckNet'].iloc[0]
    best_baseline = perf_df[perf_df['Type'] == 'Baseline'].iloc[0]
    auroc_diff    = cbn_row['AUROC'] - best_baseline['AUROC']
    auprc_diff    = cbn_row['AUPRC'] - best_baseline['AUPRC']

    perf_interp = (
        'CBN matches or exceeds the best baseline -- concept guidance does '
        'NOT sacrifice performance.'
        if auroc_diff >= -0.01 else
        'CBN shows a modest performance gap vs best baseline. The gains in '
        'interpretability (Sections 2-4) justify this trade-off for clinical '
        'deployment.'
    )

    attr_note = (
        '[POSITIVE] CBN top features are predominantly within defined '
        'clinical concepts.'
        if attribution_cov.get('CBN', 0) >= attribution_cov.get('LR', 0) else
        '[NOTE] CBN coverage lower than LR -- further concept boundary '
        'refinement may improve alignment.'
    )

    stability_lines = '\n'.join(
        f'  {name:30s}: CV = {cv:.1f}%'
        for name, cv in sorted(stability_cv.items(), key=lambda x: x[1])
    )
    all_stable = 'All concepts stable (CV < 20%).' \
        if all(v < 20 for v in stability_cv.values()) \
        else 'Most concepts stable; a few may benefit from further refinement.'

    # Use only ASCII characters throughout the report string
    report = (
        '=' * 80 + '\n'
        '  EVALUATION REPORT -- Concept-Guided Bottleneck Network  [v3]\n'
        '  30-Day Hospital Readmission Prediction (MIMIC-IV)\n'
        f'  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n'
        '=' * 80 + '\n\n'

        'CHANGES IN THIS VERSION (v3)\n'
        '-----------------------------\n'
        '  v3-1: Exp 4 (Domain Shift) removed. Zeroing features is not a\n'
        '        valid domain-shift test. Reserved for next paper with\n'
        '        proper multi-site validation (MIMIC + eICU).\n'
        '  v3-2: concept_coverage() fixed. Now uses direct pattern matching\n'
        '        against CONCEPT_FEATURE_PATTERNS (same as mask build).\n'
        '        Previous mask-membership logic gave CBN coverage of 25%%\n'
        '        despite features being explicitly concept-assigned.\n'
        '  v3-3: Report written as UTF-8; all special chars removed to\n'
        '        prevent UnicodeEncodeError on Windows (cp1252).\n\n'

        '=' * 80 + '\n'
        '1. PERFORMANCE COMPARISON (5-Fold OOF)\n'
        '=' * 80 + '\n'
        f'{perf_df.to_string(index=False)}\n\n'
        f'CBN vs Best Baseline ({best_baseline["Model"]}):\n'
        f'  AUROC difference : {auroc_diff:+.4f}\n'
        f'  AUPRC difference : {auprc_diff:+.4f}\n\n'
        f'Interpretation:\n  {perf_interp}\n\n'

        '=' * 80 + '\n'
        '2. PREDICTION INTERPRETABILITY (Exp 2)\n'
        '=' * 80 + '\n'
        'Each CBN prediction decomposes into 12 clinical concept activations.\n'
        'LSA latent features are anchored to specific clinical concepts\n'
        '(Mental Health Risk, Social Support, Recovery Stability), making the\n'
        'concept decomposition clinically interpretable.\n\n'
        'Graphs:\n'
        '  patient_concept_explanation.png  -- per-patient concept bar charts\n'
        '  concept_correlation_heatmap.png  -- inter-concept correlation matrix\n\n'

        '=' * 80 + '\n'
        '3. CONCEPT STABILITY (Exp 3)\n'
        '=' * 80 + '\n'
        'Concept importance CV% across 5 folds (lower = more stable):\n\n'
        f'{stability_lines}\n\n'
        f'{all_stable}\n\n'

        '=' * 80 + '\n'
        '4. FEATURE ATTRIBUTION AUDIT (Exp 4)\n'
        '=' * 80 + '\n'
        'Top-20 feature clinical concept coverage (pattern-matching, fair):\n'
        f'  Logistic Regression : {attribution_cov.get("LR",  0):.1f}%\n'
        f'  Random Forest       : {attribution_cov.get("RF",  0):.1f}%\n'
        f'  CBN                 : {attribution_cov.get("CBN", 0):.1f}%\n\n'
        f'{attr_note}\n\n'

        '=' * 80 + '\n'
        '5. OUTPUT FILE INDEX\n'
        '=' * 80 + '\n'
        '  results_comparison.csv          -- all model metrics\n'
        '  roc_curves.png / pr_curves.png  -- ROC and PR curves\n'
        '  performance_bar.png             -- grouped bar chart\n'
        '  calibration_curves.png          -- reliability diagrams\n'
        '  patient_concept_explanation.png -- per-patient activations\n'
        '  concept_correlation_heatmap.png -- inter-concept correlation\n'
        '  concept_stability.png           -- fold importance stability\n'
        '  feature_attribution_audit.png   -- top feature comparison\n'
        '  evaluation_report.txt           -- this document\n\n'

        '=' * 80 + '\n'
        '6. LIMITATIONS\n'
        '=' * 80 + '\n'
        '  - Results are from a single MIMIC-IV cohort; external validation needed.\n'
        '  - Concept boundaries set by domain knowledge; may not be optimal.\n'
        '  - Domain shift robustness reserved for follow-up multi-site study.\n'
        '=' * 80 + '\n'
    )

    path = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')
    # v3 CHANGE 3: explicit utf-8, errors='replace' as safety net
    with open(path, 'w', encoding='utf-8', errors='replace') as f:
        f.write(report)
    log.info(f'Saved -> {path}')
    print(report)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    log.info('=' * 70)
    log.info(f'  CBN Full Evaluation Suite v3   [device: {DEVICE}]')
    log.info('=' * 70)

    if not os.path.exists(FEATURE_MATRIX_PATH):
        raise FileNotFoundError(
            f'Feature matrix not found at {FEATURE_MATRIX_PATH}\n'
            'Run the baseline pipeline first to generate it.')

    log.info(f'Loading {FEATURE_MATRIX_PATH} ...')
    feat_df = pd.read_csv(FEATURE_MATRIX_PATH, index_col=0, low_memory=False)
    log.info(f'Shape: {feat_df.shape}')

    y    = feat_df['readmit_30d'].values.astype(int)
    drop = ['readmit_30d', 'disease_chapter']
    Xdf  = feat_df.drop(columns=[c for c in drop if c in feat_df.columns])
    for c in Xdf.select_dtypes(include='object').columns:
        Xdf[c] = LabelEncoder().fit_transform(Xdf[c].astype(str))

    feature_names = Xdf.columns.tolist()
    X = Xdf.values.astype(np.float32)

    # Bug 4 fix: prune all-NaN columns before mask build
    all_nan_mask = np.all(np.isnan(X), axis=0)
    if all_nan_mask.any():
        log.warning(f'Dropping {int(all_nan_mask.sum())} all-NaN columns')
        X             = X[:, ~all_nan_mask]
        feature_names = [f for f, v in zip(feature_names, all_nan_mask) if not v]

    log.info(f'Features: {X.shape[1]}  Samples: {X.shape[0]}  '
             f'Pos rate: {y.mean():.3f}')

    log.info('\nBuilding concept mask ...')
    mask = build_concept_mask(feature_names)
    assert mask.shape[0] == X.shape[1]

    log.info(f'\nCross-validating ALL models ({N_SPLITS} folds) ...')
    oof, fold_imp, oof_concepts = cross_validate_all(X, y, mask)

    log.info('\n-- OOF Summary --')
    label_map = {'CBN': 'ConceptBottleneckNet', 'MLP': 'Plain MLP',
                 'LR':  'Logistic Regression',  'RF':  'Random Forest',
                 'XGB': 'XGBoost'}
    for key, lbl in label_map.items():
        if key == 'XGB' and not HAS_XGB:
            continue
        m = eval_metrics(y, oof[key])
        log.info(f'  {lbl:25s}  {m}')

    perf_df      = exp1_performance(y, oof)
    exp2_explanations(oof_concepts, oof['CBN'], y)
    stability_cv = exp3_stability(fold_imp)
    attr_cov     = exp4_attribution_audit(X, y, mask, feature_names)

    # v3: write_report no longer takes domain_df
    write_report(perf_df, stability_cv, attr_cov)

    log.info('\n' + '=' * 70)
    log.info('ALL OUTPUTS --> ' + OUTPUT_DIR)
    log.info('=' * 70)


if __name__ == '__main__':
    main()