"""
=============================================================================
  30-Day Hospital Readmission Prediction — MIMIC-IV (v2.2)
  State-of-the-Art ML Pipeline (No LLMs)
=============================================================================

Tables used
-----------
hosp/admissions, hosp/patients, hosp/labevents, hosp/prescriptions,
hosp/diagnoses_icd, hosp/procedures_icd, hosp/transfers, hosp/pharmacy,
hosp/omr, icu/chartevents, icu/icustays, ed/edstays, note/discharge

Pipeline
--------
1.  Load & merge core tables
2.  Apply exclusion filters (newborns, hospice/death, same-day re-admit…)
3.  Build 30-day readmission label
4.  Feature engineering (demographics, labs, meds, diagnoses/Elixhauser,
    procedures, ICU, ED, transfers, vitals/OMR, discharge-note NLP)
5.  Disease-category stratification (ICD-10 chapters)
6.  Model training: LR baseline, RF, XGBoost, LightGBM, CatBoost, Stacking
7.  Evaluation: AUROC, AUPRC, F1, Brier, calibration — overall & per-disease
8.  SHAP feature importance
9.  Save artefacts

Checkpointing
-------------
Each expensive step saves its output to CACHE_DIR as .parquet / .joblib.
On subsequent runs the cached file is loaded directly, skipping recomputation.
Delete individual cache files (or the whole cache dir) to force recomputation.

Run
---
    pip install pandas numpy scikit-learn xgboost lightgbm catboost shap \
                imbalanced-learn textblob nltk joblib tqdm matplotlib seaborn \
                pyarrow fastparquet
    python readmission_prediction.py
=============================================================================
"""

# ─── Imports ─────────────────────────────────────────────────────────────────
import os, gc, warnings, re, logging
from pathlib import Path

import numpy  as np
import pandas as pd
import joblib
from tqdm import tqdm

# ML
from sklearn.pipeline           import Pipeline
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.impute             import SimpleImputer
from sklearn.decomposition      import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, StackingClassifier
from sklearn.calibration        import CalibratedClassifierCV
from sklearn.model_selection    import StratifiedKFold
from sklearn.metrics            import (roc_auc_score, average_precision_score,
                                        f1_score, brier_score_loss,
                                        classification_report,
                                        precision_recall_curve, roc_curve)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost  as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from textblob import TextBlob
nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1)  PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH    = 'D:/Parizad/PHD/Project/Data/mimic-iv-2.2/'
OUTPUT_DIR   = './model_outputs/'
CACHE_DIR    = './pipeline_cache/'          # ← all intermediate results saved here
CHUNK_SIZE   = 500_000
RANDOM_STATE = 42
N_SPLITS     = 5          # stratified k-fold
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)

# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(name: str, ext: str = 'parquet') -> str:
    return os.path.join(CACHE_DIR, f'{name}.{ext}')


def cache_exists(name: str, ext: str = 'parquet') -> bool:
    return os.path.exists(_cache_path(name, ext))


def save_df(df: pd.DataFrame, name: str):
    """Save DataFrame to parquet cache."""
    path = _cache_path(name, 'parquet')
    df.to_parquet(path, index=True)
    log.info(f'[CACHE] Saved  → {path}')


def load_df(name: str) -> pd.DataFrame:
    """Load DataFrame from parquet cache."""
    path = _cache_path(name, 'parquet')
    log.info(f'[CACHE] Loading ← {path}')
    return pd.read_parquet(path)


def save_obj(obj, name: str):
    """Save any Python object (sklearn, etc.) to joblib cache."""
    path = _cache_path(name, 'joblib')
    joblib.dump(obj, path)
    log.info(f'[CACHE] Saved  → {path}')


def load_obj(name: str):
    """Load Python object from joblib cache."""
    path = _cache_path(name, 'joblib')
    log.info(f'[CACHE] Loading ← {path}')
    return joblib.load(path)


def _find_csv(folder: str, name: str) -> str:
    for ext in ('.csv', '.csv.gz'):
        p = os.path.join(BASE_PATH, folder, name + ext)
        if os.path.exists(p):
            return p
    return os.path.join(BASE_PATH, folder, name + '.csv')


FILE_PATHS = {
    'admissions':    _find_csv('hosp', 'admissions'),
    'patients':      _find_csv('hosp', 'patients'),
    'labevents':     _find_csv('hosp', 'labevents'),
    'prescriptions': _find_csv('hosp', 'prescriptions'),
    'diagnoses':     _find_csv('hosp', 'diagnoses_icd'),
    'procedures':    _find_csv('hosp', 'procedures_icd'),
    'transfers':     _find_csv('hosp', 'transfers'),
    'pharmacy':      _find_csv('hosp', 'pharmacy'),
    'omr':           _find_csv('hosp', 'omr'),
    'chartevents':   _find_csv('icu',  'chartevents'),
    'icustays':      _find_csv('icu',  'icustays'),
    'edstays':       _find_csv('ed',   'edstays'),
    'discharge':     _find_csv('note', 'discharge'),
}

# Vitals item-ids (MIMIC-IV chartevents)
VITAL_ITEMS = {
    'heart_rate':     [220045],
    'sbp':            [220179, 220050],
    'dbp':            [220180, 220051],
    'temperature':    [223762, 226329],
    'spo2':           [220277],
    'resp_rate':      [220210],
    'gcs_total':      [220739, 223900, 223901],
}

# Key lab item-ids
LAB_ITEMS = {
    'sodium':      [50824, 50983],
    'potassium':   [50822, 50971],
    'creatinine':  [50912],
    'bun':         [51006],
    'glucose':     [50809, 50931],
    'hemoglobin':  [51222],
    'wbc':         [51301],
    'platelets':   [51265],
    'albumin':     [50862],
    'bilirubin':   [50885],
    'inr':         [51237],
    'lactate':     [50813],
    'hco3':        [50882, 50803],
    'troponin':    [51003],
    'bnp':         [51006],
}

# ICD-10 chapter map
ICD10_CHAPTERS = {
    'A': 'infectious',   'B': 'infectious',
    'C': 'neoplasm',     'D': 'neoplasm_blood',
    'E': 'endocrine',
    'F': 'mental',
    'G': 'nervous',
    'H': 'eye_ear',
    'I': 'circulatory',
    'J': 'respiratory',
    'K': 'digestive',
    'L': 'skin',
    'M': 'musculoskeletal',
    'N': 'genitourinary',
    'O': 'pregnancy',
    'P': 'perinatal',
    'Q': 'congenital',
    'R': 'symptoms',
    'S': 'injury',       'T': 'injury',
    'U': 'covid',
    'V': 'external',     'W': 'external',
    'X': 'external',     'Y': 'external',
    'Z': 'factors',
}

# Elixhauser comorbidity ICD-10 code prefixes
ELIXHAUSER = {
    'chf':            ['I09.9','I11.0','I13.0','I13.2','I25.5','I42.0',
                       'I42.5','I42.6','I42.7','I42.8','I42.9','I43','I50',
                       'P29.0'],
    'cardiac_arrhythmia': ['I44.1','I44.2','I44.3','I45.6','I45.9','I47',
                           'I48','I49','R00.0','R00.1','R00.8','T82.1','Z45.0',
                           'Z95.0'],
    'valvular':       ['A52.0','I05','I06','I07','I08','I09.1','I09.8',
                       'I34','I35','I36','I37','I38','I39','Q23.0','Q23.1',
                       'Q23.2','Q23.3','Z95.2','Z95.3','Z95.4'],
    'pulmonary_circ': ['I26','I27','I28.0','I28.8','I28.9'],
    'pvd':            ['I70','I71','I73.1','I73.8','I73.9','I77.1','I79.0',
                       'I79.2','K55.1','K55.8','K55.9','Z95.8','Z95.9'],
    'hypertension_uncomplicated': ['I10'],
    'hypertension_complicated':   ['I11','I12','I13','I15'],
    'paralysis':      ['G04.1','G11.4','G80.1','G80.2','G81','G82',
                       'G83.0','G83.1','G83.2','G83.3','G83.4','G83.9'],
    'other_neurological': ['G10','G11','G12','G13','G20','G21','G22',
                           'G25.4','G25.5','G31.2','G31.8','G31.9','G32',
                           'G35','G36','G37','G40','G41','G93.1','G93.4',
                           'R47.0','R56'],
    'copd':           ['I27.8','I27.9','J40','J41','J42','J43','J44',
                       'J45','J46','J47','J60','J61','J62','J63','J64',
                       'J65','J66','J67','J68.0','J68.1','J68.3','J70.2',
                       'J70.3'],
    'diabetes_uncomplicated': ['E10.0','E10.1','E10.6','E10.8','E10.9',
                               'E11.0','E11.1','E11.6','E11.8','E11.9',
                               'E12','E13','E14'],
    'diabetes_complicated':   ['E10.2','E10.3','E10.4','E10.5','E10.7',
                               'E11.2','E11.3','E11.4','E11.5','E11.7'],
    'hypothyroidism': ['E00','E01','E02','E03','E89.0'],
    'renal_failure':  ['I12.0','I13.1','N18','N19','N25.0','Z49.0',
                       'Z49.1','Z49.2','Z94.0','Z99.2'],
    'liver_disease':  ['B18','I85','I86.4','I98.2','K70','K71.1',
                       'K71.3','K71.4','K71.5','K72','K73','K74',
                       'K76.0','K76.2','K76.3','K76.4','K76.5','K76.6',
                       'K76.7','K76.8','K76.9','Z94.4'],
    'pud':            ['K25','K26','K27','K28'],
    'lymphoma':       ['C81','C82','C83','C84','C85','C88','C96',
                       'C90.0','C90.2'],
    'metastatic_cancer': ['C77','C78','C79','C80'],
    'solid_tumor':    ['C00','C01','C02','C03','C04','C05','C06','C07',
                       'C08','C09','C10','C11','C12','C13','C14','C15',
                       'C16','C17','C18','C19','C20','C21','C22','C23',
                       'C24','C25','C26','C30','C31','C32','C33','C34',
                       'C37','C38','C39','C40','C41','C43','C45','C46',
                       'C47','C48','C49','C50','C51','C52','C53','C54',
                       'C55','C56','C57','C58','C60','C61','C62','C63',
                       'C64','C65','C66','C67','C68','C69','C70','C71',
                       'C72','C73','C74','C75','C76','C97'],
    'rheumatoid':     ['L40.5','M05','M06','M08','M09','M30','M31.0',
                       'M31.1','M31.2','M31.3','M32','M33','M34',
                       'M35.1','M35.3','M36'],
    'coagulopathy':   ['D65','D66','D67','D68','D69.1','D69.3','D69.4',
                       'D69.5','D69.6'],
    'obesity':        ['E66'],
    'weight_loss':    ['E40','E41','E42','E43','E44','E45','E46',
                       'R63.4','R64'],
    'fluid_electrolyte': ['E22.2','E86','E87'],
    'blood_loss_anemia': ['D50.0'],
    'deficiency_anemia': ['D50.8','D50.9','D51','D52','D53'],
    'alcohol_abuse':  ['F10','E52','G62.1','I42.6','K29.2','K70.0',
                       'K70.3','K70.9','T51','Z50.2','Z71.4','Z72.1'],
    'drug_abuse':     ['F11','F12','F13','F14','F15','F16','F18','F19',
                       'Z71.5','Z72.2'],
    'psychoses':      ['F20','F22','F23','F24','F25','F28','F29',
                       'F30.2','F31.2','F31.5'],
    'depression':     ['F20.4','F31.3','F31.4','F31.5','F32','F33',
                       'F34.1','F41.2','F43.2'],
}


# ─────────────────────────────────────────────────────────────────────────────
# 2)  DATA LOADING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(key: str, cols=None, dtype=None) -> pd.DataFrame:
    """Load a CSV, using a parquet cache when available."""
    cache_name = f'raw_{key}'
    # Only cache full-table loads (cols=None) to keep cache simple.
    # Selective column loads are fast enough and vary by caller.
    if cols is None and cache_exists(cache_name):
        return load_df(cache_name)

    path = FILE_PATHS[key]
    log.info(f'Loading {key}  ({path})')
    df = pd.read_csv(path, usecols=cols, dtype=dtype, low_memory=False)

    if cols is None:
        save_df(df, cache_name)
    return df


def load_large_csv(key: str, subject_ids: set, id_col: str,
                   cols=None, dtype=None, agg_fn=None,
                   cache_name: str = None) -> pd.DataFrame:
    """
    Chunk-read a large table, keeping only rows in subject_ids.
    Saves the filtered result to cache so subsequent runs skip the scan.
    """
    if cache_name and cache_exists(cache_name):
        log.info(f'[CACHE] Skipping chunk-scan for {key}, loading ← {_cache_path(cache_name)}')
        return load_df(cache_name)

    path  = FILE_PATHS[key]
    log.info(f'Chunk-loading {key}  ({path})')
    parts = []
    for chunk in pd.read_csv(path, usecols=cols, dtype=dtype,
                             chunksize=CHUNK_SIZE, low_memory=False):
        sub = chunk[chunk[id_col].isin(subject_ids)]
        if agg_fn:
            sub = agg_fn(sub)
        parts.append(sub)
        gc.collect()
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    if cache_name:
        save_df(result, cache_name)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3)  CORE TABLES: ADMISSIONS + PATIENTS
# ─────────────────────────────────────────────────────────────────────────────

def build_index() -> pd.DataFrame:
    """
    Load admissions & patients, compute age, LOS, label, apply exclusions.
    Returns one row per admission (index cohort).
    Cached to 'index_cohort.parquet'.
    """
    if cache_exists('index_cohort'):
        log.info('[CACHE] Loading pre-built index cohort.')
        return load_df('index_cohort')

    adm = load_csv('admissions')
    pat = load_csv('patients', cols=['subject_id','gender','dob'
                                     if 'dob' in pd.read_csv(
                                         FILE_PATHS['patients'], nrows=1).columns
                                     else 'anchor_age',
                                     'anchor_age','anchor_year',
                                     'anchor_year_group','dod'])

    if 'dob' in pat.columns:
        pat['anchor_age'] = np.nan
    pat.columns = pat.columns.str.lower()
    adm.columns = adm.columns.str.lower()

    for c in ['admittime','dischtime','edregtime','edouttime','deathtime']:
        if c in adm.columns:
            adm[c] = pd.to_datetime(adm[c], errors='coerce')

    if 'anchor_age' in pat.columns and pat['anchor_age'].notna().any():
        adm = adm.merge(pat[['subject_id','gender','anchor_age',
                              'anchor_year','dod']], on='subject_id', how='left')
        adm['admit_year'] = adm['admittime'].dt.year
        adm['age'] = adm['anchor_age'] + (adm['admit_year'] - adm['anchor_year'])
    else:
        adm = adm.merge(pat[['subject_id','gender','dod']], on='subject_id', how='left')
        adm['age'] = np.nan

    adm['los_days'] = (adm['dischtime'] - adm['admittime']).dt.total_seconds() / 86400

    n0 = len(adm)
    log.info(f'Total admissions before exclusions: {n0}')

    newborn_mask = (
        (adm['age'] < 1) |
        adm['admission_type'].str.contains(
            r'newborn|neonatal|neonate|birth', case=False, na=False)
    )
    adm = adm[~newborn_mask].copy()
    log.info(f'  After newborn exclusion: {len(adm)}')

    adm = adm[adm['los_days'] >= (1/24)].copy()
    log.info(f'  After LOS<1h exclusion: {len(adm)}')

    if 'hospital_expire_flag' in adm.columns:
        adm = adm[adm['hospital_expire_flag'] != 1].copy()
    log.info(f'  After in-hospital death exclusion: {len(adm)}')

    if 'discharge_location' in adm.columns:
        hospice_mask = adm['discharge_location'].str.contains(
            r'hospice|deceased', case=False, na=False)
        adm = adm[~hospice_mask].copy()
    log.info(f'  After hospice discharge exclusion: {len(adm)}')

    adm = adm[adm['age'] <= 89].copy()
    log.info(f'  After age>89 exclusion: {len(adm)}')

    adm = adm.sort_values(['subject_id', 'admittime']).reset_index(drop=True)
    adm['next_admittime'] = adm.groupby('subject_id')['admittime'].shift(-1)
    adm['days_to_next'] = (
        (adm['next_admittime'] - adm['dischtime']).dt.total_seconds() / 86400
    )
    adm['next_admission_type'] = adm.groupby('subject_id')['admission_type'].shift(-1)
    planned_mask = adm['next_admission_type'].str.contains(
        r'elective|scheduled|planned', case=False, na=False)

    adm['readmit_30d'] = (
        (adm['days_to_next'] > 0) &
        (adm['days_to_next'] <= 30) &
        (~planned_mask)
    ).astype(int)

    log.info(f'Label distribution:\n{adm["readmit_30d"].value_counts()}')
    log.info(f'Readmission rate: {adm["readmit_30d"].mean():.3f}')

    save_df(adm, 'index_cohort')
    return adm


# ─────────────────────────────────────────────────────────────────────────────
# 4)  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def feat_demographics(adm: pd.DataFrame) -> pd.DataFrame:
    cols = ['hadm_id','age','gender','marital_status','insurance',
            'language','ethnicity','admission_type','admission_location',
            'discharge_location','los_days']
    cols = [c for c in cols if c in adm.columns]
    df = adm[cols].copy()

    df['gender_m']    = (df['gender'] == 'M').astype(int)
    df['insurance_mc']= df['insurance'].str.contains(
        r'medicare', case=False, na=False).astype(int)
    df['insurance_md']= df['insurance'].str.contains(
        r'medicaid', case=False, na=False).astype(int)
    df['insured_self'] = df['insurance'].str.contains(
        r'self|private', case=False, na=False).astype(int)
    df['married']     = df['marital_status'].str.contains(
        r'married', case=False, na=False).astype(int) \
        if 'marital_status' in df.columns else 0
    df['english']     = df['language'].str.upper().eq('ENGLISH').astype(int) \
        if 'language' in df.columns else 1

    for v in ['emergency','urgent','elective']:
        df[f'adm_type_{v}'] = df['admission_type'].str.contains(
            v, case=False, na=False).astype(int)

    if 'discharge_location' in df.columns:
        df['disch_home']  = df['discharge_location'].str.contains(
            r'home|self', case=False, na=False).astype(int)
        df['disch_snf']   = df['discharge_location'].str.contains(
            r'skilled|SNF|nursing', case=False, na=False).astype(int)
        df['disch_rehab'] = df['discharge_location'].str.contains(
            r'rehab', case=False, na=False).astype(int)

    drop = ['gender','marital_status','insurance','language',
            'ethnicity','admission_type','admission_location','discharge_location']
    df.drop(columns=[c for c in drop if c in df.columns], inplace=True)
    return df.set_index('hadm_id')


def feat_prior_utilisation(adm: pd.DataFrame) -> pd.DataFrame:
    adm = adm.sort_values(['subject_id','admittime'])
    grp = adm.groupby('subject_id')

    adm['prior_admissions']  = grp.cumcount()
    adm['prior_readmissions']= grp['readmit_30d'].cumsum().shift(1).fillna(0)
    adm['prev_dischtime']    = grp['dischtime'].shift(1)
    adm['days_since_last_disch'] = (
        (adm['admittime'] - adm['prev_dischtime']).dt.total_seconds() / 86400
    ).clip(upper=365)

    cols = ['hadm_id','prior_admissions','prior_readmissions','days_since_last_disch']
    return adm[cols].set_index('hadm_id')


def feat_diagnoses(adm_ids: pd.Index) -> pd.DataFrame:
    if cache_exists('feat_diagnoses'):
        return load_df('feat_diagnoses')

    df = load_csv('diagnoses',
                  cols=['hadm_id','icd_code','icd_version','seq_num'])
    df = df[df['hadm_id'].isin(adm_ids)].copy()
    df['icd_code'] = df['icd_code'].astype(str).str.strip()

    n_diag  = df.groupby('hadm_id')['icd_code'].nunique().rename('n_diagnoses')
    primary = (df[df['seq_num'] == 1]
               .drop_duplicates('hadm_id')
               .set_index('hadm_id')['icd_code'])

    def _chapter(code):
        c = str(code).upper()
        if not c:
            return 'unknown'
        first = c[0]
        if first in ICD10_CHAPTERS:
            return ICD10_CHAPTERS[first]
        try:
            n = int(c[:3])
            if   1   <= n <= 139:  return 'infectious'
            elif 140 <= n <= 239:  return 'neoplasm'
            elif 240 <= n <= 279:  return 'endocrine'
            elif 280 <= n <= 289:  return 'neoplasm_blood'
            elif 290 <= n <= 319:  return 'mental'
            elif 320 <= n <= 389:  return 'nervous'
            elif 390 <= n <= 459:  return 'circulatory'
            elif 460 <= n <= 519:  return 'respiratory'
            elif 520 <= n <= 579:  return 'digestive'
            elif 580 <= n <= 629:  return 'genitourinary'
            elif 630 <= n <= 679:  return 'pregnancy'
            elif 680 <= n <= 709:  return 'skin'
            elif 710 <= n <= 739:  return 'musculoskeletal'
            elif 740 <= n <= 759:  return 'congenital'
            elif 760 <= n <= 779:  return 'perinatal'
            elif 780 <= n <= 799:  return 'symptoms'
            elif 800 <= n <= 999:  return 'injury'
        except ValueError:
            pass
        return 'other'

    chapter_sr = primary.apply(_chapter).rename('disease_chapter')

    elix_rows = {}
    for hadm, grp in df.groupby('hadm_id'):
        row = {}
        codes = grp['icd_code'].tolist()
        for cond, prefixes in ELIXHAUSER.items():
            row[f'elix_{cond}'] = int(
                any(any(c.startswith(p) for p in prefixes) for c in codes)
            )
        row['elixhauser_sum'] = sum(row.values())
        elix_rows[hadm] = row

    elix_df = pd.DataFrame.from_dict(elix_rows, orient='index')
    elix_df.index.name = 'hadm_id'

    out = (pd.concat([n_diag, chapter_sr, elix_df], axis=1)
           .reindex(adm_ids))
    out['disease_chapter'] = out['disease_chapter'].fillna('unknown')
    out = out.fillna(0)
    save_df(out, 'feat_diagnoses')
    return out


def feat_procedures(adm_ids: pd.Index) -> pd.DataFrame:
    if cache_exists('feat_procedures'):
        return load_df('feat_procedures')

    df = load_csv('procedures', cols=['hadm_id','icd_code','icd_version'])
    df = df[df['hadm_id'].isin(adm_ids)]

    n_proc = df.groupby('hadm_id')['icd_code'].nunique().rename('n_procedures')
    df['surgical'] = df['icd_code'].apply(
        lambda c: int(str(c)[0] == '0') if str(c)[0].isdigit() else 0)
    n_surgical = df.groupby('hadm_id')['surgical'].sum().rename('n_surgical_proc')

    out = (pd.concat([n_proc, n_surgical], axis=1)
           .reindex(adm_ids).fillna(0))
    save_df(out, 'feat_procedures')
    return out


def feat_labs(adm_ids: pd.Index, hadm_admittime: pd.Series) -> pd.DataFrame:
    """Last-value, mean, std, flag count for key labs."""
    if cache_exists('feat_labs'):
        return load_df('feat_labs')

    all_item_ids = [iid for ids in LAB_ITEMS.values() for iid in ids]
    cols = ['hadm_id','itemid','valuenum','flag']

    def _chunk_agg(chunk):
        return chunk[chunk['hadm_id'].isin(adm_ids) &
                     chunk['itemid'].isin(all_item_ids)].copy()

    df = load_large_csv('labevents', adm_ids, 'hadm_id',
                        cols=cols, agg_fn=_chunk_agg,
                        cache_name='raw_labevents_filtered')
    if df.empty:
        return pd.DataFrame(index=adm_ids)

    df['valuenum'] = pd.to_numeric(df['valuenum'], errors='coerce')
    df['abnormal'] = df['flag'].notna() & df['flag'].str.contains(
        r'abnormal|delta', case=False, na=False)

    rows = {}
    for name, iids in LAB_ITEMS.items():
        sub = df[df['itemid'].isin(iids)]
        grp = sub.groupby('hadm_id')['valuenum']
        rows[f'lab_{name}_last']  = grp.last()
        rows[f'lab_{name}_mean']  = grp.mean()
        rows[f'lab_{name}_std']   = grp.std()
        rows[f'lab_{name}_n']     = grp.count()

    abn    = df.groupby('hadm_id')['abnormal'].sum().rename('lab_n_abnormal')
    lab_df = pd.DataFrame(rows).reindex(adm_ids)
    lab_df['lab_n_abnormal'] = abn.reindex(adm_ids).fillna(0)

    save_df(lab_df, 'feat_labs')
    return lab_df


def feat_medications(adm_ids: pd.Index) -> pd.DataFrame:
    if cache_exists('feat_medications'):
        return load_df('feat_medications')

    rx = load_csv('prescriptions',
                  cols=['hadm_id','drug','formulary_drug_cd','route'])
    rx = rx[rx['hadm_id'].isin(adm_ids)].copy()

    n_drugs    = rx.groupby('hadm_id')['drug'].nunique().rename('n_unique_drugs')
    polypharmacy = (n_drugs >= 10).astype(int).rename('polypharmacy')

    for cls, pattern in {
        'antibiotic':   r'mycin|cillin|cycline|floxacin|cef|meropenem|vanco',
        'anticoag':     r'warfarin|heparin|enoxaparin|apixaban|rivaroxaban',
        'insulin':      r'insulin',
        'diuretic':     r'furosemide|lasix|torsemide|spiro|hydrochloro',
        'statin':       r'statin|vastatin',
        'opioid':       r'morphine|oxycodone|hydrocodone|fentanyl|tramadol',
        'steroid':      r'prednis|dexameth|hydrocortisone|methylpredni',
        'antipsychotic':r'haloperidol|quetiapine|olanzapine|risperidone',
    }.items():
        flag = rx.groupby('hadm_id')['drug'].apply(
            lambda s: int(s.str.contains(pattern, case=False, na=False).any())
        ).rename(f'med_{cls}')
        n_drugs = pd.concat([n_drugs, flag], axis=1)

    try:
        ph     = load_csv('pharmacy', cols=['hadm_id','dispensation'])
        ph     = ph[ph['hadm_id'].isin(adm_ids)]
        n_disp = ph.groupby('hadm_id').size().rename('n_pharmacy_dispenses')
    except Exception:
        n_disp = pd.Series(dtype=float, name='n_pharmacy_dispenses')

    out = pd.concat([n_drugs, polypharmacy, n_disp], axis=1).reindex(adm_ids).fillna(0)
    save_df(out, 'feat_medications')
    return out


def feat_icu(adm_ids: pd.Index) -> pd.DataFrame:
    if cache_exists('feat_icu'):
        return load_df('feat_icu')

    icu = load_csv('icustays',
                   cols=['hadm_id','los','first_careunit','last_careunit'])
    icu = icu[icu['hadm_id'].isin(adm_ids)].copy()
    icu['los'] = pd.to_numeric(icu['los'], errors='coerce')

    icu_flag  = icu.groupby('hadm_id').size().rename('n_icu_stays').gt(0).astype(int)
    icu_los   = icu.groupby('hadm_id')['los'].sum().rename('total_icu_los')
    micu_flag = icu['first_careunit'].str.contains(r'MICU|Medical', case=False, na=False)
    micu_sr   = icu[micu_flag].groupby('hadm_id').size().gt(0).astype(int).rename('micu_flag')
    ccu_flag  = icu['first_careunit'].str.contains(r'CCU|cardiac|coronary', case=False, na=False)
    ccu_sr    = icu[ccu_flag].groupby('hadm_id').size().gt(0).astype(int).rename('ccu_flag')

    out = pd.concat([icu_flag.astype(int), icu_los, micu_sr, ccu_sr], axis=1)
    out = out.reindex(adm_ids).fillna(0)
    out['icu_admitted'] = (out['n_icu_stays'] > 0).astype(int)

    save_df(out, 'feat_icu')
    return out


def feat_ed(subject_ids: pd.Index, hadm_subject: pd.Series) -> pd.DataFrame:
    if cache_exists('feat_ed'):
        return load_df('feat_ed')

    ed = load_csv('edstays', cols=['subject_id','hadm_id','intime','outtime',
                                   'disposition'])
    ed = ed[ed['subject_id'].isin(subject_ids)].copy()
    ed['los_ed_h'] = (
        (pd.to_datetime(ed['outtime']) - pd.to_datetime(ed['intime']))
        .dt.total_seconds() / 3600
    )

    n_ed    = ed.groupby('hadm_id').size().rename('n_ed_visits')
    ed_los  = ed.groupby('hadm_id')['los_ed_h'].mean().rename('mean_ed_los_h')
    ed_admit= (ed.groupby('hadm_id')['disposition'].apply(
        lambda s: int(s.str.contains(r'admit', case=False, na=False).any())
    ).rename('ed_admitted'))

    out = pd.concat([n_ed, ed_los, ed_admit], axis=1)
    hadm_subj_df = hadm_subject.reset_index().set_index('hadm_id')
    out = out.reindex(hadm_subj_df.index).fillna(0)

    save_df(out, 'feat_ed')
    return out


def feat_transfers(adm_ids: pd.Index) -> pd.DataFrame:
    if cache_exists('feat_transfers'):
        return load_df('feat_transfers')

    tf = load_csv('transfers', cols=['hadm_id','careunit','eventtype'])
    tf = tf[tf['hadm_id'].isin(adm_ids)].copy()

    n_transfers = tf.groupby('hadm_id').size().rename('n_transfers')
    n_units     = tf.groupby('hadm_id')['careunit'].nunique().rename('n_care_units')

    for unit, patt in {'icu_transfer':  r'ICU|intensive',
                       'step_transfer': r'step|step-down|intermediate'}.items():
        flag = (tf[tf['careunit'].str.contains(patt, case=False, na=False)]
                .groupby('hadm_id').size().gt(0).astype(int).rename(unit))
        n_transfers = pd.concat([n_transfers, flag], axis=1)

    out = pd.concat([n_transfers, n_units], axis=1).reindex(adm_ids).fillna(0)
    save_df(out, 'feat_transfers')
    return out


def feat_vitals(adm_ids: pd.Index) -> pd.DataFrame:
    """Load/aggregate vitals from chartevents, with caching."""
    if cache_exists('feat_vitals'):
        return load_df('feat_vitals')

    all_item_ids = [iid for ids in VITAL_ITEMS.values() for iid in ids]
    cols = ['hadm_id','itemid','valuenum']

    def _chunk_filter(chunk):
        return chunk[chunk['hadm_id'].isin(adm_ids) &
                     chunk['itemid'].isin(all_item_ids)]

    df = load_large_csv('chartevents', adm_ids, 'hadm_id',
                        cols=cols, agg_fn=_chunk_filter,
                        cache_name='raw_chartevents_filtered')
    if df.empty:
        return pd.DataFrame(index=adm_ids)

    df['valuenum'] = pd.to_numeric(df['valuenum'], errors='coerce')
    rows = {}
    for name, iids in VITAL_ITEMS.items():
        sub = df[df['itemid'].isin(iids)]
        grp = sub.groupby('hadm_id')['valuenum']
        rows[f'vital_{name}_mean'] = grp.mean()
        rows[f'vital_{name}_min']  = grp.min()
        rows[f'vital_{name}_max']  = grp.max()
        rows[f'vital_{name}_std']  = grp.std()

    vdf = pd.DataFrame(rows).reindex(adm_ids)
    if {'vital_sbp_mean','vital_dbp_mean'}.issubset(vdf.columns):
        vdf['vital_pulse_pressure'] = vdf['vital_sbp_mean'] - vdf['vital_dbp_mean']
        vdf['vital_map'] = vdf['vital_dbp_mean'] + vdf['vital_pulse_pressure'] / 3

    save_df(vdf, 'feat_vitals')
    return vdf


def feat_omr(adm_ids: pd.Index, hadm_subject: pd.Series) -> pd.DataFrame:
    if cache_exists('feat_omr'):
        return load_df('feat_omr')

    omr = load_csv('omr', cols=['subject_id','chartdate','result_name','result_value'])
    subj_ids = set(hadm_subject.values)
    omr = omr[omr['subject_id'].isin(subj_ids)].copy()
    omr['result_value'] = pd.to_numeric(omr['result_value'], errors='coerce')

    omr_pivot = (omr.groupby(['subject_id','result_name'])['result_value']
                 .last().unstack(fill_value=np.nan))
    omr_pivot.columns = [f'omr_{c.lower().replace(" ","_")}' for c in omr_pivot.columns]

    subj_to_hadm = (hadm_subject.reset_index()
                    .rename(columns={'index':'hadm_id','subject_id':'subject_id'})
                    .set_index('subject_id'))
    out = (omr_pivot
           .join(subj_to_hadm, how='inner')
           .reset_index(drop=True)
           .set_index('hadm_id')
           .reindex(adm_ids))
    save_df(out, 'feat_omr')
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4l  Discharge Notes — NLP Features  (most expensive step)
# ─────────────────────────────────────────────────────────────────────────────

_STOP = set(stopwords.words('english'))

def _clean_note(text: str) -> str:
    text = re.sub(r'\[\*\*.*?\*\*\]', ' ', str(text))
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = ' '.join(w for w in text.lower().split() if w not in _STOP and len(w) > 2)
    return text

READMIT_KEYWORDS = [
    'readmit','re-admit','return to ed','follow.up','follow up',
    'non-compli','noncompliant','non compli',
    'poorly controlled','uncontrolled',
    'medication not taken','missed dose','substance abuse',
    'homeless','social support','lives alone',
    'wound infection','dehiscence','sepsis','fever',
    'shortness of breath','dyspnea','chest pain',
    'confusion','altered mental','fall risk',
]


def feat_notes(adm_ids: pd.Index) -> tuple:
    """
    Returns (structured_df, raw_texts_series).
    Both are cached independently so re-runs skip text processing.
    """
    struct_cached = cache_exists('feat_notes_struct')
    texts_cached  = cache_exists('feat_notes_raw_texts')

    if struct_cached and texts_cached:
        log.info('[CACHE] Loading note features and raw texts.')
        return load_df('feat_notes_struct'), load_df('feat_notes_raw_texts')['clean']

    notes = load_csv('discharge',
                     cols=['hadm_id','text','charttime'] if 'charttime' in
                          pd.read_csv(FILE_PATHS['discharge'], nrows=1).columns
                          else ['hadm_id','text'])
    notes = notes[notes['hadm_id'].isin(adm_ids)].drop_duplicates('hadm_id')
    notes['text'] = notes['text'].fillna('')
    notes = notes.set_index('hadm_id')

    log.info('Extracting structured NLP features from discharge notes…')
    feats = pd.DataFrame(index=notes.index)
    feats['note_char_len'] = notes['text'].str.len()
    feats['note_word_len'] = notes['text'].apply(lambda t: len(str(t).split()))

    for kw in READMIT_KEYWORDS:
        col = 'note_kw_' + re.sub(r'\W+', '_', kw).strip('_')
        feats[col] = notes['text'].str.contains(kw, case=False, na=False).astype(int)
    feats['note_kw_total'] = feats[[c for c in feats.columns
                                    if c.startswith('note_kw_')]].sum(axis=1)

    log.info('Running sentiment analysis…')
    def _sentiment(t):
        blob = TextBlob(str(t)[:5000])
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    sent = notes['text'].apply(_sentiment)
    feats['note_polarity']     = sent.apply(lambda x: x[0])
    feats['note_subjectivity'] = sent.apply(lambda x: x[1])

    log.info('Cleaning notes for TF-IDF…')
    notes['clean'] = notes['text'].apply(_clean_note)
    raw_texts      = notes[['clean']].reindex(adm_ids)
    raw_texts['clean'] = raw_texts['clean'].fillna('')

    feats = feats.reindex(adm_ids).fillna(0)

    save_df(feats,     'feat_notes_struct')
    save_df(raw_texts, 'feat_notes_raw_texts')
    return feats, raw_texts['clean']


def build_tfidf_lsa(raw_texts: pd.Series, n_components: int = 50):
    """
    Fit TF-IDF + LSA and return (vectorizer, svd, lsa_df).
    Both the fitted objects and the LSA matrix are cached.
    On subsequent runs only the LSA DataFrame is loaded (no re-fitting).
    """
    lsa_cached    = cache_exists('feat_lsa')
    objects_cached = cache_exists('tfidf_svd', ext='joblib')

    if lsa_cached and objects_cached:
        log.info('[CACHE] Loading LSA features and TF-IDF/SVD objects.')
        lsa_df          = load_df('feat_lsa')
        tfidf, svd      = load_obj('tfidf_svd')
        return tfidf, svd, lsa_df

    log.info('Fitting TF-IDF + LSA on discharge notes…')
    tfidf = TfidfVectorizer(max_features=30_000, ngram_range=(1, 2),
                            min_df=5, sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(raw_texts.fillna(''))
    svd     = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    X_lsa   = svd.fit_transform(X_tfidf)
    lsa_df  = pd.DataFrame(X_lsa,
                           index=raw_texts.index,
                           columns=[f'lsa_{i}' for i in range(n_components)])

    save_df(lsa_df,        'feat_lsa')
    save_obj((tfidf, svd), 'tfidf_svd')
    return tfidf, svd, lsa_df


# ─────────────────────────────────────────────────────────────────────────────
# 5)  ASSEMBLE FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def assemble_features(adm: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the full feature matrix.
    The assembled matrix is also cached so the entire feature engineering
    step can be skipped on subsequent runs.
    """
    if cache_exists('feature_matrix'):
        log.info('[CACHE] Loading pre-assembled feature matrix.')
        feat_df = load_df('feature_matrix')
        # Still need to return tfidf/svd objects for potential SHAP use
        if cache_exists('tfidf_svd', ext='joblib'):
            tfidf, svd = load_obj('tfidf_svd')
        else:
            tfidf, svd = None, None
        return feat_df, tfidf, svd

    adm_ids      = pd.Index(adm['hadm_id'].unique())
    hadm_subject = adm.set_index('hadm_id')['subject_id']
    subject_ids  = set(hadm_subject.values)

    log.info('=== Feature Engineering ===')
    dfs = {}
    dfs['demog']   = feat_demographics(adm)
    dfs['prior']   = feat_prior_utilisation(adm)
    dfs['diag']    = feat_diagnoses(adm_ids)
    dfs['proc']    = feat_procedures(adm_ids)
    dfs['lab']     = feat_labs(adm_ids, adm.set_index('hadm_id')['admittime'])
    dfs['med']     = feat_medications(adm_ids)
    dfs['icu']     = feat_icu(adm_ids)
    dfs['ed']      = feat_ed(subject_ids, hadm_subject)
    dfs['transfer']= feat_transfers(adm_ids)
    dfs['vitals']  = feat_vitals(adm_ids)
    dfs['omr']     = feat_omr(adm_ids, hadm_subject)

    note_feats, raw_texts      = feat_notes(adm_ids)
    tfidf, svd, lsa_df         = build_tfidf_lsa(raw_texts)
    dfs['note_struct'] = note_feats
    dfs['note_lsa']    = lsa_df

    log.info('Merging feature blocks…')
    feat_df = pd.concat(list(dfs.values()), axis=1)
    feat_df.index.name = 'hadm_id'

    if 'disease_chapter' not in feat_df.columns:
        diag_chapter = dfs['diag']['disease_chapter'].astype(str) \
            if 'disease_chapter' in dfs['diag'].columns \
            else pd.Series('unknown', index=feat_df.index, name='disease_chapter')
        feat_df['disease_chapter'] = diag_chapter.astype(str)

    feat_df['readmit_30d'] = (adm.set_index('hadm_id')['readmit_30d']
                               .reindex(feat_df.index))

    log.info(f'Feature matrix shape: {feat_df.shape}')
    save_df(feat_df, 'feature_matrix')
    return feat_df, tfidf, svd


# ─────────────────────────────────────────────────────────────────────────────
# 6)  MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_models():
    lr = LogisticRegression(max_iter=1000, solver='lbfgs',
                            class_weight='balanced', C=0.1,
                            random_state=RANDOM_STATE)

    rf = RandomForestClassifier(n_estimators=500, max_depth=12,
                                min_samples_leaf=20, n_jobs=-1,
                                class_weight='balanced',
                                random_state=RANDOM_STATE)

    xgb_m = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=5,
        eval_metric='aucpr',
        n_jobs=-1, random_state=RANDOM_STATE,
    )

    xgb_stack = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=5,
        eval_metric='aucpr',
        n_jobs=-1, random_state=RANDOM_STATE,
    )

    lgbm = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, min_split_gain=0.01,
        class_weight='balanced', n_jobs=-1,
        random_state=RANDOM_STATE, verbose=-1,
    )

    cat = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6,
        auto_class_weights='Balanced', eval_metric='AUC',
        early_stopping_rounds=50, verbose=0,
        random_seed=RANDOM_STATE,
    )

    cat_stack = CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6,
        auto_class_weights='Balanced', eval_metric='AUC',
        verbose=0, random_seed=RANDOM_STATE,
    )

    stack = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_stack), ('lgbm', lgbm)],
        final_estimator=LogisticRegression(max_iter=500,
                                           class_weight='balanced',
                                           random_state=RANDOM_STATE),
        cv=3, n_jobs=-1, passthrough=False,
    )

    return {
        'LogisticRegression': lr,
        'RandomForest':       rf,
        'XGBoost':            xgb_m,
        'LightGBM':           lgbm,
        'CatBoost':           cat,
        'StackingEnsemble':   stack,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7)  TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'AUROC':  roc_auc_score(y_true, y_prob),
        'AUPRC':  average_precision_score(y_true, y_prob),
        'F1':     f1_score(y_true, y_pred, zero_division=0),
        'Brier':  brier_score_loss(y_true, y_prob),
    }


def train_evaluate(feat_df: pd.DataFrame) -> dict:
    """
    Cross-validated training.
    Per-model OOF predictions are cached so individual models can be re-run
    without repeating all others.
    """
    y       = feat_df['readmit_30d'].values.astype(int)
    chapter = feat_df['disease_chapter'].astype(str).values \
              if 'disease_chapter' in feat_df.columns \
              else np.array(['unknown'] * len(y))

    drop_cols = ['readmit_30d', 'disease_chapter']
    X_df      = feat_df.drop(columns=[c for c in drop_cols if c in feat_df.columns])

    cat_cols = X_df.select_dtypes(include='object').columns.tolist()
    for c in cat_cols:
        le = LabelEncoder()
        X_df[c] = le.fit_transform(X_df[c].astype(str))

    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(np.float32)

    skf    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = get_models()

    all_results = {}
    oof_preds   = {}

    for model_name, model in models.items():
        oof_cache = f'oof_{model_name}'

        # ── Load cached OOF if available ──────────────────────────────────────
        if cache_exists(oof_cache):
            log.info(f'[CACHE] Skipping training for {model_name}, loading OOF predictions.')
            oof_prob = load_df(oof_cache)['oof_prob'].values
        else:
            log.info(f'\n>>> Training {model_name}')
            oof_prob = np.zeros(len(y), dtype=np.float64)

            for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
                X_tr, X_val = X[tr_idx], X[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]

                imp   = SimpleImputer(strategy='median')
                X_tr  = imp.fit_transform(X_tr)
                X_val = imp.transform(X_val)

                if model_name == 'LogisticRegression':
                    scl   = StandardScaler()
                    X_tr  = scl.fit_transform(X_tr)
                    X_val = scl.transform(X_val)

                if HAS_SMOTE and model_name in ('RandomForest',):
                    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
                    try:
                        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                    except Exception:
                        pass

                fit_kwargs = {}
                if model_name in ('XGBoost',):
                    fit_kwargs = {'eval_set': [(X_val, y_val)], 'verbose': False, 'early_stopping_rounds': 50}
                elif model_name in ('CatBoost',):
                    fit_kwargs = {'eval_set': (X_val, y_val)}

                try:
                    model.fit(X_tr, y_tr, **fit_kwargs)
                except TypeError:
                    model.fit(X_tr, y_tr)

                prob = model.predict_proba(X_val)[:, 1]
                oof_prob[val_idx] = prob
                m = evaluate(y_val, prob)
                log.info(f'  Fold {fold}: AUROC={m["AUROC"]:.4f}  AUPRC={m["AUPRC"]:.4f}')

            # Cache OOF predictions for this model
            oof_df = pd.DataFrame({'oof_prob': oof_prob})
            save_df(oof_df, oof_cache)

        oof_preds[model_name] = oof_prob
        overall = evaluate(y, oof_prob)

        chapter_results = {}
        for chap in np.unique(chapter):
            mask = chapter == chap
            if mask.sum() < 30 or y[mask].sum() < 5:
                continue
            try:
                chapter_results[chap] = evaluate(y[mask], oof_prob[mask])
            except Exception:
                pass

        all_results[model_name] = {
            'overall':    overall,
            'by_disease': chapter_results,
            'oof_prob':   oof_prob,
        }
        log.info(f'{model_name} OOF → '
                 f'AUROC={overall["AUROC"]:.4f}  '
                 f'AUPRC={overall["AUPRC"]:.4f}  '
                 f'F1={overall["F1"]:.4f}  '
                 f'Brier={overall["Brier"]:.4f}')

    return all_results, feature_names, X, y, chapter


# ─────────────────────────────────────────────────────────────────────────────
# 8)  SHAP FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(model, X: np.ndarray, feature_names: list,
                 n_samples: int = 2000):
    if not HAS_SHAP:
        log.warning('shap not installed — skipping')
        return
    log.info('Computing SHAP values…')
    idx   = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_sub = X[idx]
    imp   = SimpleImputer(strategy='median')
    X_sub = imp.fit_transform(X_sub)
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.LinearExplainer(model, X_sub)
    shap_vals = explainer.shap_values(X_sub)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_sub, feature_names=feature_names,
                      show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'), dpi=150)
    plt.close()
    log.info('SHAP plot saved.')


# ─────────────────────────────────────────────────────────────────────────────
# 9)  REPORTING & PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_pr(all_results: dict, y: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette   = plt.cm.tab10.colors
    for i, (name, res) in enumerate(all_results.items()):
        prob  = res['oof_prob']
        fpr, tpr, _ = roc_curve(y, prob)
        prec, rec, _ = precision_recall_curve(y, prob)
        auc   = res['overall']['AUROC']
        apr   = res['overall']['AUPRC']
        col   = palette[i % len(palette)]
        axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=col)
        axes[1].plot(rec, prec, label=f'{name} (APR={apr:.3f})', color=col)

    axes[0].plot([0,1],[0,1],'k--'); axes[0].set_title('ROC Curve')
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].legend(fontsize=7)

    base_rate = y.mean()
    axes[1].axhline(base_rate, color='k', linestyle='--',
                    label=f'Baseline ({base_rate:.3f})')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_pr_curves.png'), dpi=150)
    plt.close()
    log.info('ROC/PR curves saved.')


def plot_disease_heatmap(all_results: dict):
    records = []
    for model, res in all_results.items():
        for chap, metrics in res['by_disease'].items():
            records.append({'model': model, 'chapter': chap,
                            **{k: round(v, 3) for k, v in metrics.items()}})
    if not records:
        return
    df = pd.DataFrame(records)
    for metric in ('AUROC', 'AUPRC', 'F1'):
        if metric not in df.columns:
            continue
        pivot = df.pivot(index='chapter', columns='model', values=metric)
        plt.figure(figsize=(max(8, len(pivot.columns)*2), max(6, len(pivot)*0.5)))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                    vmin=0.5, vmax=0.9, linewidths=0.5)
        plt.title(f'{metric} by Disease Chapter')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'disease_heatmap_{metric}.png'), dpi=150)
        plt.close()
    log.info('Disease heatmaps saved.')


def save_summary(all_results: dict):
    rows = []
    for model, res in all_results.items():
        rows.append({'Model': model, 'Set': 'Overall', 'Chapter': '—',
                     **res['overall']})
        for chap, metrics in res['by_disease'].items():
            rows.append({'Model': model, 'Set': 'ByDisease',
                         'Chapter': chap, **metrics})
    summary   = pd.DataFrame(rows)
    out_path  = os.path.join(OUTPUT_DIR, 'results_summary.csv')
    summary.to_csv(out_path, index=False)
    log.info(f'Results summary saved → {out_path}')
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 10)  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info('═' * 70)
    log.info('  30-Day Readmission Prediction — MIMIC-IV   SOA ML Pipeline')
    log.info(f'  Cache directory: {os.path.abspath(CACHE_DIR)}')
    log.info('  Delete cache files to force recomputation of any step.')
    log.info('═' * 70)

    # ── Step 1: Build index cohort ────────────────────────────────────────────
    adm = build_index()

    # ── Step 2: Assemble feature matrix ──────────────────────────────────────
    feat_df, tfidf, svd = assemble_features(adm)

    # Save human-readable copy for inspection
    feat_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'))
    log.info('Feature matrix CSV saved.')

    # ── Step 3: Train & evaluate ──────────────────────────────────────────────
    all_results, feature_names, X, y, chapter = train_evaluate(feat_df)

    # ── Step 4: Plots ─────────────────────────────────────────────────────────
    plot_roc_pr(all_results, y)
    plot_disease_heatmap(all_results)

    # ── Step 5: SHAP on best model ────────────────────────────────────────────
    best_name = max(all_results,
                    key=lambda k: all_results[k]['overall']['AUROC'])
    log.info(f'Best model: {best_name}')
    imp_full = SimpleImputer(strategy='median')
    X_imp    = imp_full.fit_transform(X)
    best_model = get_models()[best_name]
    try:
        best_model.fit(X_imp, y)
        compute_shap(best_model, X_imp, feature_names)
    except Exception as e:
        log.warning(f'SHAP failed: {e}')

    # ── Step 6: Save summary ─────────────────────────────────────────────────
    summary = save_summary(all_results)

    log.info('\n' + '═' * 70)
    log.info('OVERALL RESULTS')
    log.info('═' * 70)
    overall_df = (summary[summary['Set'] == 'Overall']
                  [['Model','AUROC','AUPRC','F1','Brier']]
                  .sort_values('AUROC', ascending=False))
    print(overall_df.to_string(index=False))
    log.info('═' * 70)
    log.info('All outputs written to:  ' + OUTPUT_DIR)


if __name__ == '__main__':
    main()