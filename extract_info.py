"""
MIMIC-IV Feature Extraction Pipeline — v1.3  (LEAKAGE-FIXED)
=============================================================
Changes vs v1.2
  LEAK-01  build_c1()  adm_discharge_location REMOVED from keep list
                       → REPLACE with: drop it entirely (it encodes "DIED", "HOSPICE" etc.)
  LEAK-02  build_c1()  adm_los_days REMOVED from keep list
                       → REPLACE with: retain only for cohort filtering in load_base_tables(), not as model feature
  LEAK-03  build_c1()  adm_inhospital_mortality kept only as label column, not feature
                       → REPLACE with: exclude from X; use only as y
  LEAK-04  build_c2()  util_prior_hadm_count_* are all 0 after first-admission filter
                       → REPLACE with: removed; only util_days_since_last_discharge and ED counts kept
  LEAK-05  build_c3()  trans_last_careunit REMOVED
                       → REPLACE with: trans_first_careunit only (last careunit = where patient died/discharged)
                       icu_total_los_days KEPT but renamed _dpt suffix (discharge-point-time)
  LEAK-06  build_c4()  Charlson computed from THIS admission's ICD codes (billed at discharge)
                       → REPLACE with: Charlson computed from PRIOR admissions only
                       dx/proc counts from this admission kept and explicitly labelled _ws
  LEAK-07  build_c5()  _ws lab columns (whole-stay min/max) REMOVED from output
                       → REPLACE with: only bw (first 24h) and pdw (last 24h) windows exported
                       aki_creat_max_ws removed; AKI ratio now uses bw max instead
  LEAK-08  build_d()   note_stable_at_discharge and note_resolution_score REMOVED
                       → REPLACE with: these encode clinical outcome; only social/logistic features kept
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0) IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import sys
import warnings
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1) PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH   = 'D:/Parizad/PHD/Project/Data/mimic-iv-2.2/'
OUTPUT_FILE = './model_outputs/extracted_features2.csv'
CHUNK_SIZE  = 500_000

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def _find_csv_file(folder: str, name: str) -> str:
    csv = os.path.join(BASE_PATH, folder, name + '.csv')
    gz  = os.path.join(BASE_PATH, folder, name + '.csv.gz')
    return csv if os.path.exists(csv) else (gz if os.path.exists(gz) else csv)

FILE_PATHS = {
    'admissions':    _find_csv_file('hosp', 'admissions'),
    'patients':      _find_csv_file('hosp', 'patients'),
    'labevents':     _find_csv_file('hosp', 'labevents'),
    'prescriptions': _find_csv_file('hosp', 'prescriptions'),
    'diagnoses':     _find_csv_file('hosp', 'diagnoses_icd'),
    'procedures':    _find_csv_file('hosp', 'procedures_icd'),
    'transfers':     _find_csv_file('hosp', 'transfers'),
    'pharmacy':      _find_csv_file('hosp', 'pharmacy'),
    'omr':           _find_csv_file('hosp', 'omr'),
    'chartevents':   _find_csv_file('icu',  'chartevents'),
    'icustays':      _find_csv_file('icu',  'icustays'),
    'edstays':       _find_csv_file('ed',   'edstays'),
    'discharge':     _find_csv_file('note', 'discharge'),
}

# ─────────────────────────────────────────────────────────────────────────────
# 2) DOMAIN DICTIONARIES
# ─────────────────────────────────────────────────────────────────────────────
LAB_ITEMIDS = {
    "creatinine":  {50912, 52024},
    "bun":         {51006},
    "sodium":      {50983, 52623},
    "potassium":   {50971, 52610},
    "bicarbonate": {50882},
    "chloride":    {50902},
    "hemoglobin":  {51222},
    "platelets":   {51265},
    "wbc":         {51301},
    "glucose":     {50931, 50809, 52027},
    "albumin":     {50862},
    "bilirubin":   {50885},
    "lactate":     {50813},
}
ITEMID_TO_ANALYTE = {iid: a for a, ids in LAB_ITEMIDS.items() for iid in ids}
ALL_LAB_ITEMIDS   = set(ITEMID_TO_ANALYTE.keys())
ALL_ANALYTES      = list(LAB_ITEMIDS.keys())

VITAL_ITEMIDS = {
    "sbp":  {220179, 220050},
    "dbp":  {220180, 220051},
    "map":  {220052, 220181},
    "hr":   {220045},
    "rr":   {220210, 224690},
    "spo2": {220277},
    "temp": {223761, 223762},
}
ITEMID_TO_VITAL   = {iid: v for v, ids in VITAL_ITEMIDS.items() for iid in ids}
ALL_VITAL_ITEMIDS = set(ITEMID_TO_VITAL.keys())
ALL_VITALS        = list(VITAL_ITEMIDS.keys())

GCS_ITEMIDS = {220739, 223900, 223901}  # eye opening, verbal, motor

ICU_CAREUNIT_KEYWORDS = {
    "micu", "sicu", "cicu", "csru", "nicu", "icu",
    "intensive care", "coronary care", "neuro icu",
    "trauma icu", "surgical icu", "cardiac icu",
    "medical icu", "pediatric icu", "burn icu",
}

# ─────────────────────────────────────────────────────────────────────────────
# 2b) CHARLSON ICD MAPPINGS  (Quan 2005 + ICD-10 update)
# ─────────────────────────────────────────────────────────────────────────────
CHARLSON_ICD9 = {
    "mi":               ["410","412"],
    "chf":              ["39891","40201","40211","40291","40401","40403","40411",
                         "40413","40491","40493","4254","4255","4257","4258","4259","428"],
    "pvd":              ["0930","4373","440","4471","5571","5579","V434"],
    "cvd":              ["36234","430","431","432","433","434","435","436","437","438"],
    "dementia":         ["290","2941","3312"],
    "copd":             ["4168","4169","490","491","492","493","494","495","496",
                         "500","501","502","503","504","505","5064","5081","5088"],
    "rheum":            ["4465","7100","7101","7102","7103","7104","7108","7109",
                         "7112","714","7193","720","725","7285","72889","72930"],
    "pud":              ["531","532","533","534"],
    "mild_liver":       ["07022","07023","07032","07033","07044","07054","0706",
                         "0709","570","571","5733","5734","5738","5739","V427"],
    "diabetes":         ["2500","2501","2502","2503","2508","2509"],
    "diabetes_comp":    ["2504","2505","2506","2507"],
    "hemi_para":        ["3341","342","343","3440","3441","3442","3443","3444",
                         "3445","3446","3449"],
    "renal":            ["40301","40311","40391","40402","40403","40412","40413",
                         "40492","40493","582","5830","5831","5832","5834","5836",
                         "5837","585","586","5880","V420","V451","V56"],
    "cancer":           ["140","141","142","143","144","145","146","147","148",
                         "149","150","151","152","153","154","155","156","157",
                         "158","159","160","161","162","163","164","165","170",
                         "171","172","174","175","176","179","180","181","182",
                         "183","184","185","186","187","188","189","190","191",
                         "192","193","194","195","200","201","202","203","204",
                         "205","206","207","208","2386"],
    "mod_severe_liver": ["4560","4561","4562","5722","5723","5724","5725","5726",
                         "5727","5728"],
    "metastatic":       ["196","197","198","199"],
    "hiv":              ["042","043","044"],
}

CHARLSON_ICD10 = {
    "mi":               ["I21","I22","I252"],
    "chf":              ["I099","I110","I130","I132","I255","I420","I425","I426",
                         "I427","I428","I429","I43","I50","P290"],
    "pvd":              ["I70","I71","I731","I738","I739","I771","I790","I792",
                         "K551","K558","K559","Z958","Z959"],
    "cvd":              ["G45","G46","H340","I60","I61","I62","I63","I64","I65",
                         "I66","I67","I68","I69"],
    "dementia":         ["F00","F01","F02","F03","F051","G30","G311"],
    "copd":             ["I278","I279","J40","J41","J42","J43","J44","J45","J46",
                         "J47","J60","J61","J62","J63","J64","J65","J66","J67",
                         "J684","J701","J703"],
    "rheum":            ["M05","M06","M315","M32","M33","M34","M351","M353","M360"],
    "pud":              ["K25","K26","K27","K28"],
    "mild_liver":       ["B18","K700","K701","K702","K703","K709","K713","K714",
                         "K715","K717","K73","K74","K760","K762","K763","K764",
                         "K768","K769","Z944"],
    "diabetes":         ["E100","E101","E106","E108","E109","E110","E111","E116",
                         "E118","E119","E120","E121","E126","E128","E129","E130",
                         "E131","E136","E138","E139","E140","E141","E146","E148","E149"],
    "diabetes_comp":    ["E102","E103","E104","E105","E107","E112","E113","E114",
                         "E115","E117","E122","E123","E124","E125","E127","E132",
                         "E133","E134","E135","E137","E142","E143","E144","E145","E147"],
    "hemi_para":        ["G041","G114","G801","G802","G81","G82","G830","G831",
                         "G832","G833","G834","G839"],
    "renal":            ["I120","I131","N03","N05","N18","N19","N250","Z490",
                         "Z491","Z492","Z940","Z992"],
    "cancer":           ["C00","C01","C02","C03","C04","C05","C06","C07","C08",
                         "C09","C10","C11","C12","C13","C14","C15","C16","C17",
                         "C18","C19","C20","C21","C22","C23","C24","C25","C26",
                         "C30","C31","C32","C33","C34","C37","C38","C39","C40",
                         "C41","C43","C45","C46","C47","C48","C49","C50","C51",
                         "C52","C53","C54","C55","C56","C57","C58","C60","C61",
                         "C62","C63","C64","C65","C66","C67","C68","C69","C70",
                         "C71","C72","C73","C74","C75","C76","C81","C82","C83",
                         "C84","C85","C88","C90","C91","C92","C93","C94","C95",
                         "C96","C97"],
    "mod_severe_liver": ["I850","I859","I864","I982","K704","K711","K721","K729",
                         "K765","K766","K767"],
    "metastatic":       ["C77","C78","C79","C80"],
    "hiv":              ["B20","B21","B22","B24"],
}

CHARLSON_WEIGHTS = {
    "mi":1, "chf":1, "pvd":1, "cvd":1, "dementia":1, "copd":1, "rheum":1, "pud":1,
    "mild_liver":1, "diabetes":1, "hemi_para":2, "renal":2, "cancer":2,
    "diabetes_comp":2, "mod_severe_liver":3, "metastatic":6, "hiv":6,
}

# ─────────────────────────────────────────────────────────────────────────────
# 2c) MED NORMALISATION & HIGH-RISK CLASSES
# ─────────────────────────────────────────────────────────────────────────────
def normalize_drug_name(name: str) -> str:
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()
    name = re.sub(r'\s*\d+[\.\d]*\s*(mg|mcg|meq|g|ml|unit|iu|%|units).*', '', name, flags=re.I)
    name = re.sub(r'\b(tablet|capsule|injection|infusion|solution|oral|iv|po|im|sq|subl|'
                  r'patch|cream|ointment|drops?|syrup|suspension|er|sr|xr|hcl|sodium|sulfate)\b',
                  '', name, flags=re.I)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

HIGH_RISK_CLASSES = {
    "anticoagulant":  {"warfarin","heparin","enoxaparin","fondaparinux","rivaroxaban",
                       "apixaban","dabigatran","argatroban","bivalirudin","dalteparin",
                       "tinzaparin","edoxaban"},
    "insulin":        {"insulin","insulin regular","insulin lispro","insulin aspart",
                       "insulin glargine","insulin detemir","insulin nph","insulin degludec",
                       "insulin glulisine","humulin","novolog","lantus","levemir","toujeo"},
    "opioid":         {"morphine","oxycodone","hydromorphone","fentanyl","methadone",
                       "codeine","tramadol","hydrocodone","oxymorphone","buprenorphine",
                       "meperidine","tapentadol"},
    "benzo":          {"lorazepam","diazepam","midazolam","alprazolam","clonazepam",
                       "temazepam","triazolam","oxazepam","chlordiazepoxide"},
    "antipsychotic":  {"haloperidol","quetiapine","olanzapine","risperidone","ziprasidone",
                       "aripiprazole","clozapine","lurasidone","paliperidone"},
    "antiarrhythmic": {"amiodarone","lidocaine","procainamide","quinidine","flecainide",
                       "propafenone","sotalol","dofetilide","digoxin","adenosine",
                       "verapamil","diltiazem"},
    "steroid":        {"dexamethasone","prednisone","methylprednisolone","hydrocortisone",
                       "prednisolone","budesonide","fludrocortisone","triamcinolone",
                       "betamethasone"},
    "chemo":          {"cisplatin","carboplatin","oxaliplatin","cyclophosphamide",
                       "methotrexate","doxorubicin","vincristine","paclitaxel","docetaxel",
                       "gemcitabine","fluorouracil","capecitabine","irinotecan","etoposide",
                       "temozolomide","rituximab","trastuzumab"},
}

FREQ_DAILY_MAP = {
    "qd":1,"daily":1,"once":1,"q24h":1,"every 24":1,
    "bid":2,"twice":2,"q12h":2,"every 12":2,
    "tid":3,"three times":3,"q8h":3,"every 8":3,
    "qid":4,"four times":4,"q6h":4,"every 6":4,
    "q4h":6,"every 4":6,
    "q2h":12,"every 2":12,
    "q1h":24,"every hour":24,
    "prn":1,
}

def map_frequency_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    out = pd.Series(np.ones(len(s), dtype=float), index=s.index)
    for k, v in FREQ_DAILY_MAP.items():
        out = np.where(s.str.contains(re.escape(k), na=False), float(v), out)
        out = pd.Series(out, index=s.index)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 3) SMALL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def is_icu_unit(unit_name) -> bool:
    if pd.isna(unit_name):
        return False
    return any(kw in str(unit_name).lower() for kw in ICU_CAREUNIT_KEYWORDS)

def _safe_div(a, b):
    return np.where((b != 0) & (~pd.isna(b)) & (~pd.isna(a)), a / b, np.nan)

def _icd_match(code_series: pd.Series, prefixes: list) -> pd.Series:
    if not prefixes:
        return pd.Series(False, index=code_series.index)
    s   = code_series.str.replace('.', '', regex=False).str.upper().fillna('')
    pat = '|'.join('^' + re.escape(p.upper()) for p in prefixes)
    return s.str.match(pat, na=False)

# ─────────────────────────────────────────────────────────────────────────────
# 4) LOAD BASE TABLES
# ─────────────────────────────────────────────────────────────────────────────
def load_base_tables() -> pd.DataFrame:
    log.info("Loading admissions …")
    adm = pd.read_csv(
        FILE_PATHS['admissions'],
        usecols=['subject_id','hadm_id','admittime','dischtime',
                 'admission_type','admission_location','discharge_location',
                 'insurance','marital_status','race',
                 'edregtime','edouttime','hospital_expire_flag'],
        parse_dates=['admittime','dischtime','edregtime','edouttime'],
    )
    log.info(f"  admissions rows (raw): {len(adm):,}")

    bad_ts = adm['dischtime'] <= adm['admittime']
    if bad_ts.sum():
        log.warning(f"  Dropping {bad_ts.sum()} rows where dischtime <= admittime")
        adm = adm[~bad_ts].copy()

    neg_ed = (adm['edouttime'].notna() & adm['edregtime'].notna() &
              (adm['edouttime'] < adm['edregtime']))
    if neg_ed.sum():
        log.warning(f"  Nulling {neg_ed.sum()} rows with negative ED LOS")
        adm.loc[neg_ed, ['edouttime','edregtime']] = np.nan

    log.info("Loading patients …")
    pts = pd.read_csv(
        FILE_PATHS['patients'],
        usecols=['subject_id','gender','anchor_age','anchor_year','anchor_year_group'],
    )

    base = adm.merge(pts, on='subject_id', how='left')
    base = base.sort_values(['subject_id','admittime']).reset_index(drop=True)

    base["yob"]     = base["anchor_year"] - base["anchor_age"]
    base["adm_age"] = base["admittime"].dt.year - base["yob"]
    base.loc[base["anchor_age"] >= 91, "adm_age"] = 91

    # calculate readmission BEFORE dropping any future admissions
    base = base.sort_values(['subject_id', 'admittime']).copy()
    base['next_admittime'] = base.groupby('subject_id')['admittime'].shift(-1)
    base['days_to_readmit'] = (base['next_admittime'] - base['dischtime']).dt.total_seconds() / 86400.0
    base['readmitted_30d'] = ((base['days_to_readmit'] > 0) & (base['days_to_readmit'] <= 30)).astype(int)
    base.loc[base['hospital_expire_flag'] == 1, 'readmitted_30d'] = 0

    log.info(f"  Pre-filter cohort: {len(base):,} admissions")
    base = base[base['adm_age'] >= 18].copy()
    log.info(f"  After age>=18:            {len(base):,}")
    base = base[~base['admission_type'].str.upper().str.contains('NEWBORN', na=False)].copy()
    log.info(f"  After newborn exclusion:  {len(base):,}")

    # adm_los_days kept here ONLY for cohort filter — NOT forwarded to build_c1
    base['adm_los_days'] = (base['dischtime'] - base['admittime']).dt.total_seconds() / 86400.0
    base = base[base['adm_los_days'] >= 0.5].copy()
    log.info(f"  After min LOS>=0.5d:      {len(base):,}")

    transfer_in = ['TRANSFER FROM HOSPITAL', 'TRANSFER FROM SKILLED NURSING FACILITY']
    base = base[~base['admission_location'].isin(transfer_in)].copy()
    log.info(f"  After transfer-in excl.:  {len(base):,}")

    base = base.sort_values(['subject_id', 'admittime'])
    base = base.drop_duplicates(subset='subject_id', keep='first')
    log.info(f"  After first-admission:    {len(base):,}")
    log.info(f"  Final base cohort:        {len(base):,} admissions")
    return base

# ─────────────────────────────────────────────────────────────────────────────
# 5) C1 — DEMOGRAPHICS & ADMISSION CONTEXT
# ─────────────────────────────────────────────────────────────────────────────
def build_c1(base: pd.DataFrame) -> pd.DataFrame:
    """
    LEAK-01: REPLACE 'adm_discharge_location' → REMOVED from keep list.
             Values like "DIED", "HOSPICE", "SNF" are direct proxies for the outcome.

    LEAK-02: REPLACE 'adm_los_days' in keep list → REMOVED.
             LOS is a whole-stay summary unknown at admission time.
             It is computed in load_base_tables() for cohort filtering only.

    LEAK-03: 'adm_inhospital_mortality' is kept as a LABEL COLUMN only.
             → Do NOT include it in your X feature matrix when training.
    """
    log.info("C1 — Demographics & admission context …")
    df = base.copy()
    df['adm_gender']             = df['gender']
    df['adm_race']               = df['race']
    df['adm_insurance']          = df['insurance']
    df['adm_marital_status']     = df['marital_status']
    df['adm_has_ed']             = df['edregtime'].notna().astype(int)

    # Cap ED LOS at admittime so it never includes time after admission
    # REPLACE: raw (edouttime - edregtime) → clipped so edouttime <= admittime
    ed_duration_hours = np.where(
        df['edregtime'].notna() & df['edouttime'].notna(),
        (df[['edouttime','admittime']].min(axis=1) - df['edregtime']
         ).dt.total_seconds() / 3600.0,
        np.nan,
    )
    df['adm_ed_los_hours']       = np.where(
        np.array(ed_duration_hours, dtype=float) < 0, np.nan, ed_duration_hours)

    df['adm_admission_type']     = df['admission_type']
    df['adm_admission_location'] = df['admission_location']
    df['adm_inhospital_mortality'] = df['hospital_expire_flag']  # LABEL — exclude from X

    # REPLACE old keep list → adm_discharge_location and adm_los_days removed
    keep = [
        'subject_id', 'hadm_id', 'admittime', 'dischtime',
        'adm_age', 'adm_gender', 'adm_race', 'adm_insurance', 'adm_marital_status',
        'adm_has_ed', 'adm_ed_los_hours',
        'adm_admission_type', 'adm_admission_location',
        'adm_inhospital_mortality',   # label column — DO NOT use as feature
    ]
    return df[keep]

# ─────────────────────────────────────────────────────────────────────────────
# 6) C2 — UTILIZATION HISTORY
# ─────────────────────────────────────────────────────────────────────────────
def build_c2(base: pd.DataFrame) -> pd.DataFrame:
    """
    LEAK-04: REPLACE util_prior_hadm_count_30/90/180/365d → REMOVED.
             After the first-admission-only filter in load_base_tables() every patient
             has exactly one row, so all four lookback counts are always 0 — zero-variance
             columns that waste space and can distort feature-importance scores.

             KEPT: util_days_since_last_discharge (NaN for everyone here, but structurally
             correct if you later relax the first-admission filter).
             KEPT: util_prior_ed_count_30d / _180d (from the separate edstays table, valid).
    """
    log.info("C2 — Utilization history …")

    b = base[['subject_id','hadm_id','admittime','dischtime']].sort_values(
        ['subject_id','admittime']).copy()

    out_rows = []
    for sid, g in tqdm(b.groupby('subject_id', sort=False),
                       total=b['subject_id'].nunique(), desc="  per-subject util"):
        t_adm = g['admittime'].to_numpy(dtype='datetime64[ns]')
        t_dis = g['dischtime'].to_numpy(dtype='datetime64[ns]')
        hadm  = g['hadm_id'].to_numpy()

        prior_max_dis    = np.maximum.accumulate(t_dis)
        prior_max_dis    = np.roll(prior_max_dis, 1)
        prior_max_dis[0] = np.datetime64('NaT')

        days_since = (t_adm - prior_max_dis) / np.timedelta64(1, 'D')
        days_since = days_since.astype(float)
        days_since[np.isnat(prior_max_dis)] = np.nan
        days_since = np.where(days_since < 0, np.nan, days_since)

        # REPLACE: util_prior_hadm_count_* columns removed entirely
        out_rows.append(pd.DataFrame({
            'hadm_id':                        hadm,
            'util_days_since_last_discharge': days_since,
        }))

    util = pd.concat(out_rows, ignore_index=True)

    if os.path.exists(FILE_PATHS['edstays']):
        log.info("  ED lookback via ed.edstays …")
        ed = pd.read_csv(FILE_PATHS['edstays'], usecols=['subject_id','intime'],
                         parse_dates=['intime'])
        ed = ed.sort_values(['subject_id','intime'])
        ed_groups = {sid: s.to_numpy(dtype='datetime64[ns]')
                     for sid, s in ed.groupby('subject_id')['intime']}

        ed30  = np.zeros(len(b), dtype=int)
        ed180 = np.zeros(len(b), dtype=int)
        b_idx = b.reset_index(drop=True)

        for sid, g in tqdm(b_idx.groupby('subject_id', sort=False),
                           total=b_idx['subject_id'].nunique(), desc="  per-subject ED"):
            times = ed_groups.get(sid)
            if times is None or len(times) == 0:
                continue
            t_adm = g['admittime'].to_numpy(dtype='datetime64[ns]')
            pos   = g.index.to_numpy()

            def cnt(times_arr, t, lb_days):
                lb = np.array(lb_days, dtype='timedelta64[D]')
                lo = np.searchsorted(times_arr, t - lb, side='right')
                hi = np.searchsorted(times_arr, t,       side='left')
                return (hi - lo).astype(int)

            ed30[pos]  = cnt(times, t_adm, 30)
            ed180[pos] = cnt(times, t_adm, 180)

        util = util.merge(pd.DataFrame({
            'hadm_id':                  b_idx['hadm_id'].to_numpy(),
            'util_prior_ed_count_30d':  ed30,
            'util_prior_ed_count_180d': ed180,
        }), on='hadm_id', how='left')
    else:
        log.info("  edstays not found — ED lookback skipped")
        util['util_prior_ed_count_30d']  = np.nan
        util['util_prior_ed_count_180d'] = np.nan

    return util

# ─────────────────────────────────────────────────────────────────────────────
# 7) C3 — TRANSFERS + ICU
# ─────────────────────────────────────────────────────────────────────────────
def build_c3(base: pd.DataFrame) -> pd.DataFrame:
    """
    LEAK-05: REPLACE trans_last_careunit → REMOVED.
             The last careunit is where the patient was at discharge or death.
             For mortality prediction this leaks outcome directly (e.g. "CCU" at death).
             REPLACE with: trans_first_careunit only.

             icu_total_los_days renamed → icu_total_los_days_dpt (_dpt = discharge-point-time)
             to make clear it is only valid for discharge-time prediction models.
    """
    log.info("C3 — Care transitions & ICU exposure …")

    log.info("  Loading transfers …")
    tr = pd.read_csv(
        FILE_PATHS['transfers'],
        usecols=['hadm_id','careunit','intime','outtime'],
        parse_dates=['intime','outtime'],
    )
    tr = tr[tr['hadm_id'].notna()].copy()
    tr['hadm_id'] = tr['hadm_id'].astype(int)
    tr = tr.merge(base[['hadm_id','admittime','dischtime']], on='hadm_id', how='inner')
    tr_ws = tr[(tr['intime'] >= tr['admittime']) & (tr['intime'] <= tr['dischtime'])].copy()

    feats = (tr_ws.sort_values(['hadm_id','intime'])
                  .groupby('hadm_id')
                  .agg(
                      icu_any_transfer=('careunit',
                                        lambda s: int(pd.Series(s).apply(is_icu_unit).any())),
                      trans_num_transfers_ws=('careunit', 'size'),
                      trans_first_careunit=('careunit', 'first'),
                      # REPLACE: trans_last_careunit removed — leaks discharge/death unit
                  )
                  .reset_index())

    tmp = tr_ws.sort_values(['hadm_id','intime'])[['hadm_id','careunit']].copy()
    tmp['prev'] = tmp.groupby('hadm_id')['careunit'].shift(1)
    chg = (tmp.assign(changed=lambda x: (x['careunit'] != x['prev']).astype(int))
              .groupby('hadm_id')['changed'].sum()
              .reset_index(name='trans_num_careunit_changes_ws'))
    feats = feats.merge(chg, on='hadm_id', how='left')

    feats = base[['hadm_id']].merge(feats, on='hadm_id', how='left')
    feats['icu_any_transfer']              = feats['icu_any_transfer'].fillna(0).astype(int)
    feats['trans_num_transfers_ws']        = feats['trans_num_transfers_ws'].fillna(0).astype(int)
    feats['trans_num_careunit_changes_ws'] = feats['trans_num_careunit_changes_ws'].fillna(0).astype(int)

    if os.path.exists(FILE_PATHS['icustays']):
        log.info("  Loading icustays …")
        icu = pd.read_csv(
            FILE_PATHS['icustays'],
            usecols=['hadm_id','stay_id','intime','outtime','los'],
            parse_dates=['intime','outtime'],
        )
        icu['hadm_id'] = icu['hadm_id'].astype(int)
        icu_agg = (icu.groupby('hadm_id')
                      .agg(
                          icu_num_stays=('stay_id', 'count'),
                          icu_total_los_days_dpt=('los', 'sum'),  # REPLACE: renamed from icu_total_los_days → _dpt suffix
                          icu_first_intime=('intime', 'min'),
                      )
                      .reset_index())
        icu_agg['icu_any'] = 1
        icu_agg = icu_agg.merge(base[['hadm_id','admittime']], on='hadm_id', how='left')
        icu_agg['icu_time_to_first_icu_hours'] = (
            (icu_agg['icu_first_intime'] - icu_agg['admittime']).dt.total_seconds() / 3600.0
        ).clip(lower=0)
        icu_agg = icu_agg.drop(columns=['admittime'])

        icu_df = base[['hadm_id']].merge(icu_agg, on='hadm_id', how='left')
        icu_df['icu_any']       = icu_df['icu_any'].fillna(0).astype(int)
        icu_df['icu_num_stays'] = icu_df['icu_num_stays'].fillna(0).astype(int)
    else:
        icu_df = base[['hadm_id']].copy()
        for c, v in [('icu_any', 0), ('icu_num_stays', 0),
                     ('icu_total_los_days_dpt', np.nan),
                     ('icu_first_intime', np.nan),
                     ('icu_time_to_first_icu_hours', np.nan)]:
            icu_df[c] = v

    return feats.merge(icu_df, on='hadm_id', how='left')

# ─────────────────────────────────────────────────────────────────────────────
# 8) C4 — DIAGNOSES / PROCEDURES / CHARLSON
# ─────────────────────────────────────────────────────────────────────────────
def build_c4(base: pd.DataFrame) -> pd.DataFrame:
    """
    LEAK-06: REPLACE Charlson from this admission's ICD codes → PRIOR admissions only.

    In MIMIC-IV, diagnoses_icd codes are assigned at discharge billing time and cover
    the entire current admission — including new diagnoses that arose during the stay
    (e.g. hospital-acquired MI coded as I21, new AKI, new sepsis).
    Using them for Charlson gives the model knowledge of conditions that only became
    known at the END of the stay, which is pure future leakage.

    FIX: Load ALL admissions for the cohort subjects, then for each index admission
    compute Charlson only from ICD codes belonging to admissions that were fully
    discharged (dischtime) BEFORE the index admittime.

    dx/proc COUNT features from the current admission (_ws) are kept — they describe
    complexity at the discharge coding point, which is legitimate for discharge-time
    models and is clearly labelled with the _ws suffix.
    """
    log.info("C4 — Diagnoses, procedures, Charlson (prior admissions only) …")

    base_subjects = set(base['subject_id'].unique())

    # Load all admissions for every subject in the cohort (not just index admissions)
    all_adm = pd.read_csv(
        FILE_PATHS['admissions'],
        usecols=['subject_id','hadm_id','dischtime'],
        parse_dates=['dischtime'],
    )
    all_adm = all_adm[all_adm['subject_id'].isin(base_subjects)].copy()

    log.info("  Loading diagnoses_icd (all admissions for cohort subjects) …")
    dx_all = pd.read_csv(FILE_PATHS['diagnoses'],
                         usecols=['hadm_id','icd_code','icd_version'], low_memory=False)
    dx_all['hadm_id']  = dx_all['hadm_id'].astype(int)
    dx_all['icd_code'] = dx_all['icd_code'].fillna('').astype(str).str.strip()

    # Attach subject_id and that admission's dischtime to every ICD row
    dx_all = dx_all.merge(all_adm[['hadm_id','subject_id','dischtime']],
                          on='hadm_id', how='inner')

    # Join with index admission times so we can filter to prior-only rows
    index_times = base[['subject_id','hadm_id','admittime']].copy()
    dx_prior = dx_all.merge(index_times, on='subject_id', suffixes=('', '_idx'))
    # REPLACE: dx codes from this admission → keep only codes from admissions
    #          whose dischtime is strictly before this admission's admittime
    dx_prior = dx_prior[dx_prior['dischtime'] < dx_prior['admittime']].copy()

    # Count features from THIS admission's ICD codes (valid at discharge-time)
    dx_this = dx_all[dx_all['hadm_id'].isin(base['hadm_id'])].copy()
    dx_cnt = (dx_this.groupby('hadm_id')
                     .agg(dx_num_codes_ws=('icd_code', 'size'),
                          dx_num_distinct_codes_ws=('icd_code', 'nunique'))
                     .reset_index())

    dx9_prior  = dx_prior[dx_prior['icd_version'] == 9].copy()
    dx10_prior = dx_prior[dx_prior['icd_version'] == 10].copy()

    charlson_df = base[['hadm_id']].copy()
    for comp in CHARLSON_WEIGHTS.keys():
        pfx9  = CHARLSON_ICD9.get(comp,  [])
        pfx10 = CHARLSON_ICD10.get(comp, [])

        flag9  = _icd_match(dx9_prior['icd_code'],  pfx9)
        flag10 = _icd_match(dx10_prior['icd_code'], pfx10)

        # Map matched rows back to the index admission's hadm_id
        hadm_with = pd.concat([
            dx9_prior.loc[flag9,   ['hadm_id_idx']].rename(columns={'hadm_id_idx':'hadm_id'}),
            dx10_prior.loc[flag10, ['hadm_id_idx']].rename(columns={'hadm_id_idx':'hadm_id'}),
        ])['hadm_id'].unique()

        # REPLACE: dx_charlson_{comp}_ws → dx_charlson_{comp}_prior
        charlson_df[f'dx_charlson_{comp}_prior'] = (
            charlson_df['hadm_id'].isin(hadm_with).astype(int)
        )

    # Hierarchical overrides
    mask_dc = charlson_df['dx_charlson_diabetes_comp_prior'] == 1
    charlson_df.loc[mask_dc, 'dx_charlson_diabetes_prior'] = 0
    mask_ml = charlson_df['dx_charlson_mod_severe_liver_prior'] == 1
    charlson_df.loc[mask_ml, 'dx_charlson_mild_liver_prior'] = 0

    # REPLACE: dx_charlson_index_ws → dx_charlson_index_prior
    charlson_df['dx_charlson_index_prior'] = sum(
        charlson_df[f'dx_charlson_{comp}_prior'] * w
        for comp, w in CHARLSON_WEIGHTS.items()
    )

    log.info("  Loading procedures_icd …")
    pr = pd.read_csv(FILE_PATHS['procedures'], usecols=['hadm_id','icd_code'], low_memory=False)
    pr = pr[pr['hadm_id'].isin(base['hadm_id'])].copy()
    pr['hadm_id'] = pr['hadm_id'].astype(int)
    proc_cnt = (pr.groupby('hadm_id')
                  .agg(proc_num_codes_ws=('icd_code', 'size'),
                       proc_num_distinct_codes_ws=('icd_code', 'nunique'))
                  .reset_index())

    out = (charlson_df
           .merge(dx_cnt,   on='hadm_id', how='left')
           .merge(proc_cnt, on='hadm_id', how='left'))
    out['dx_num_codes_ws']            = out['dx_num_codes_ws'].fillna(0).astype(int)
    out['dx_num_distinct_codes_ws']   = out['dx_num_distinct_codes_ws'].fillna(0).astype(int)
    out['proc_num_codes_ws']          = out['proc_num_codes_ws'].fillna(0).astype(int)
    out['proc_num_distinct_codes_ws'] = out['proc_num_distinct_codes_ws'].fillna(0).astype(int)
    out['dx_charlson_index_prior']    = out['dx_charlson_index_prior'].fillna(0)

    log.info(f"  Charlson (prior) — mean {out['dx_charlson_index_prior'].mean():.2f}, "
             f"median {out['dx_charlson_index_prior'].median():.0f}")
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 9) C5 — LABS
# ─────────────────────────────────────────────────────────────────────────────
def build_c5(base: pd.DataFrame) -> pd.DataFrame:
    """
    LEAK-07: REPLACE _ws (whole-stay) lab columns → REMOVED from model output.
             Whole-stay min/max span admittime to dischtime. For any prediction made
             before discharge these include future lab values.

             REPLACE with: only bw (first 24h) and pdw (last 24h) windows exported.
             - bw window is safe for admission-time models.
             - pdw window is valid only for discharge-time models (suffix _pdw makes this explicit).

             AKI features: aki_creat_max_ws removed.
             REPLACE with: aki_creat_max_bw (max creatinine in the first 24h).
    """
    log.info("C5 — Labs (chunked, bw + pdw windows only) …")

    times = base[['hadm_id','admittime','dischtime']].copy()
    times['hadm_id']   = times['hadm_id'].astype(int)
    times['bw_end']    = times['admittime'] + pd.Timedelta(hours=24)
    times['pdw_start'] = times['dischtime'] - pd.Timedelta(hours=24)

    acc = pd.DataFrame()

    reader = pd.read_csv(
        FILE_PATHS['labevents'],
        usecols=['hadm_id','itemid','charttime','valuenum','flag'],
        parse_dates=['charttime'],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in tqdm(reader, desc="  labevents chunks", unit="chunk"):
        chunk = chunk[
            chunk['hadm_id'].notna() &
            chunk['itemid'].isin(ALL_LAB_ITEMIDS) &
            chunk['valuenum'].notna()
        ].copy()
        if chunk.empty:
            continue
        chunk['hadm_id'] = chunk['hadm_id'].astype(int)
        chunk['analyte'] = chunk['itemid'].map(ITEMID_TO_ANALYTE)
        chunk = chunk.merge(times, on='hadm_id', how='inner')
        chunk = chunk[(chunk['charttime'] >= chunk['admittime']) &
                      (chunk['charttime'] <= chunk['dischtime'])]
        if chunk.empty:
            continue

        bw  = chunk['charttime'] <= chunk['bw_end']
        pdw = chunk['charttime'] >= chunk['pdw_start']

        # REPLACE: ws_df (whole-stay window) removed
        bw_df  = chunk.loc[bw,  ['hadm_id','analyte','charttime','valuenum','flag']].copy()
        bw_df['win'] = 'bw'
        pdw_df = chunk.loc[pdw, ['hadm_id','analyte','charttime','valuenum','flag']].copy()
        pdw_df['win'] = 'pdw'

        dfw = pd.concat([bw_df, pdw_df], ignore_index=True)
        if dfw.empty:
            continue

        fl = dfw['flag'].fillna("").astype(str).str.lower()
        dfw['is_abn'] = fl.str.contains('abnormal|high|low|critical', na=False).astype(int)

        idx_last  = dfw.groupby(['hadm_id','analyte','win'])['charttime'].idxmax()
        idx_first = dfw.groupby(['hadm_id','analyte','win'])['charttime'].idxmin()

        last_vals  = dfw.loc[idx_last,  ['hadm_id','analyte','win','valuenum','charttime']].rename(
            columns={'valuenum':'last_val', 'charttime':'last_time'})
        first_vals = dfw.loc[idx_first, ['hadm_id','analyte','win','valuenum','charttime']].rename(
            columns={'valuenum':'first_val','charttime':'first_time'})

        agg = (dfw.groupby(['hadm_id','analyte','win'])
                  .agg(v_min=('valuenum','min'), v_max=('valuenum','max'),
                       v_sum=('valuenum','sum'), v_cnt=('valuenum','count'),
                       abn_cnt=('is_abn','sum'))
                  .reset_index()
               .merge(last_vals,  on=['hadm_id','analyte','win'], how='left')
               .merge(first_vals, on=['hadm_id','analyte','win'], how='left'))

        if acc.empty:
            acc = agg[['hadm_id','analyte','win','v_min','v_max','v_sum','v_cnt',
                        'abn_cnt','last_time','last_val','first_time','first_val']].copy()
        else:
            key_cols = ['hadm_id','analyte','win']
            acc = acc.set_index(key_cols)
            new = agg.set_index(key_cols)[['v_min','v_max','v_sum','v_cnt',
                                            'abn_cnt','last_time','last_val',
                                            'first_time','first_val']]
            idx = acc.index.union(new.index)
            acc = acc.reindex(idx); new = new.reindex(idx)

            for c in ['v_sum','v_cnt','abn_cnt']:
                acc[c] = acc[c].fillna(0) + new[c].fillna(0)
            acc['v_min'] = np.fmin(acc['v_min'].to_numpy(), new['v_min'].to_numpy())
            acc['v_max'] = np.fmax(acc['v_max'].to_numpy(), new['v_max'].to_numpy())

            a_t = acc['last_time']; n_t = new['last_time']
            take_new = ((a_t.isna() & n_t.notna()) |
                        (n_t.notna() & a_t.notna() & (n_t > a_t)))
            acc.loc[take_new, 'last_time'] = n_t[take_new]
            acc.loc[take_new, 'last_val']  = new.loc[take_new, 'last_val']

            a_f = acc['first_time']; n_f = new['first_time']
            take_new_f = ((a_f.isna() & n_f.notna()) |
                          (n_f.notna() & a_f.notna() & (n_f < a_f)))
            acc.loc[take_new_f, 'first_time'] = n_f[take_new_f]
            acc.loc[take_new_f, 'first_val']  = new.loc[take_new_f, 'first_val']

            acc = acc.reset_index()

    if acc.empty:
        log.info("  No lab rows — returning empty frame.")
        return base[['hadm_id']].copy()

    acc['v_mean'] = acc['v_sum'] / acc['v_cnt'].replace(0, np.nan)

    # Slope + delta for PDW
    pdw_acc = acc[acc['win'] == 'pdw'].copy()
    pdw_acc['delta_last_first'] = pdw_acc['last_val'] - pdw_acc['first_val']
    dt_hours = (pdw_acc['last_time'] - pdw_acc['first_time']).dt.total_seconds() / 3600.0
    pdw_acc['slope'] = np.where(
        (pdw_acc['v_cnt'] >= 2) & (dt_hours > 0),
        pdw_acc['delta_last_first'] / dt_hours,
        np.nan,
    )

    def colname(metric, analyte, win):
        return f'lab_{analyte}_{metric}_{win}'

    wide_parts = []
    # REPLACE: loop was ['bw','ws','pdw'] → now ['bw','pdw'] only (ws removed)
    for win in ['bw', 'pdw']:
        sub = acc[acc['win'] == win].copy()
        if sub.empty:
            continue
        for m in ['last_val','v_min','v_max','v_mean']:
            p = sub.pivot(index='hadm_id', columns='analyte', values=m)
            p.columns = [colname(m.replace('v_','').replace('last_val','last'), a, win)
                         for a in p.columns]
            wide_parts.append(p)
        ab = sub.groupby('hadm_id')['abn_cnt'].sum().rename(f'lab_abnormal_count_{win}')
        wide_parts.append(ab.to_frame())

    for metric, col_suf in [('delta_last_first','delta_last_first_pdw'),
                             ('slope','slope_pdw')]:
        p = pdw_acc.pivot(index='hadm_id', columns='analyte', values=metric)
        p.columns = [f'lab_{a}_{col_suf}' for a in p.columns]
        wide_parts.append(p)

    lab_wide = pd.concat(wide_parts, axis=1).reset_index()
    lab_wide = base[['hadm_id']].merge(lab_wide, on='hadm_id', how='left')

    # REPLACE: aki_creat_baseline_bw  = last bw value → first bw value (true pre-admission baseline)
    # REPLACE: aki_creat_max_ws (whole-stay) → aki_creat_max_bw (first 24h max only)
    lab_wide['aki_creat_baseline_bw'] = lab_wide.get('lab_creatinine_first_val_bw', np.nan)
    lab_wide['aki_creat_max_bw']      = lab_wide.get('lab_creatinine_max_bw', np.nan)
    lab_wide['aki_creat_ratio_max_over_baseline'] = _safe_div(
        lab_wide['aki_creat_max_bw'], lab_wide['aki_creat_baseline_bw'])
    lab_wide['aki_creat_delta_max_minus_baseline'] = (
        lab_wide['aki_creat_max_bw'] - lab_wide['aki_creat_baseline_bw'])

    return lab_wide

# ─────────────────────────────────────────────────────────────────────────────
# 10) C6 — MEDICATIONS  (unchanged — medwin is valid for discharge-time models)
# ─────────────────────────────────────────────────────────────────────────────
def build_c6(base: pd.DataFrame) -> pd.DataFrame:
    log.info("C6 — Medications (chunked, merge+groupby) …")

    times = base[['hadm_id','admittime','dischtime']].copy()
    times['hadm_id']      = times['hadm_id'].astype(int)
    times['medwin_start'] = times['dischtime'] - pd.Timedelta(hours=24)

    presc_agg_rows = []
    presc_reader = pd.read_csv(
        FILE_PATHS['prescriptions'],
        usecols=['hadm_id','drug','route','starttime','stoptime'],
        parse_dates=['starttime','stoptime'],
        chunksize=CHUNK_SIZE, low_memory=False,
    )
    for chunk in tqdm(presc_reader, desc="  prescriptions chunks", unit="chunk"):
        chunk = chunk[chunk['hadm_id'].notna()].copy()
        if chunk.empty:
            continue
        chunk['hadm_id'] = chunk['hadm_id'].astype(int)
        chunk = chunk.merge(times[['hadm_id','admittime','dischtime','medwin_start']],
                            on='hadm_id', how='inner')
        ws = chunk[(chunk['starttime'] >= chunk['admittime']) &
                   (chunk['starttime'] <= chunk['dischtime'])].copy()
        if not ws.empty:
            ws['is_prn'] = (
                ws['drug'].fillna("").str.contains('prn', case=False, na=False) |
                ws['route'].fillna("").str.contains('prn', case=False, na=False)
            )
            presc_agg_rows.append(
                ws.groupby('hadm_id')
                  .agg(med_total_orders_count_ws=('drug','size'),
                       med_prn_count_ws=('is_prn','sum'))
                  .reset_index()
            )

    presc_agg = (pd.concat(presc_agg_rows, ignore_index=True)
                   .groupby('hadm_id', as_index=False).sum()
                 if presc_agg_rows
                 else pd.DataFrame(columns=['hadm_id','med_total_orders_count_ws',
                                            'med_prn_count_ws']))

    medwin_parts = []
    for src_file, usecols_src, drug_col, has_freq in [
        (FILE_PATHS['prescriptions'],
         ['hadm_id','drug','route','starttime','stoptime'], 'drug', False),
        (FILE_PATHS['pharmacy'],
         ['hadm_id','medication','route','frequency','starttime','stoptime'],
         'medication', True),
    ]:
        reader2 = pd.read_csv(src_file, usecols=usecols_src,
                              parse_dates=['starttime','stoptime'],
                              chunksize=CHUNK_SIZE, low_memory=False)
        for chunk in tqdm(reader2, desc=f"  {os.path.basename(src_file)} medwin",
                          unit="chunk"):
            chunk = chunk[chunk['hadm_id'].notna()].copy()
            if chunk.empty:
                continue
            chunk['hadm_id']   = chunk['hadm_id'].astype(int)
            chunk['drug_norm'] = chunk[drug_col].apply(normalize_drug_name)
            chunk['route_l']   = chunk['route'].fillna("").astype(str).str.lower()
            if has_freq:
                chunk['freq_score'] = map_frequency_series(
                    chunk.get('frequency', pd.Series([], dtype=str)))

            chunk = chunk.merge(times[['hadm_id','dischtime','medwin_start']],
                                on='hadm_id', how='inner')
            med = chunk[
                (chunk['starttime'] <= chunk['dischtime']) &
                (chunk['stoptime'].isna() | (chunk['stoptime'] >= chunk['medwin_start']))
            ].copy()
            if med.empty:
                continue

            for cls, sset in HIGH_RISK_CLASSES.items():
                med[f'hr_{cls}'] = med['drug_norm'].isin(sset).astype(int)

            agg_dict = dict(
                med_unique_count_medwin=(
                    'drug_norm',
                    lambda x: int(pd.Series(x).replace("", np.nan).dropna().nunique())),
                med_num_routes_medwin=(
                    'route_l',
                    lambda x: int(pd.Series(x).replace("", np.nan).dropna().nunique())),
                **{f'med_highrisk_{cls}_medwin': (f'hr_{cls}','max')
                   for cls in HIGH_RISK_CLASSES.keys()}
            )
            if has_freq:
                agg_dict['med_frequency_complexity_score_medwin'] = ('freq_score','sum')

            medwin_parts.append(med.groupby('hadm_id').agg(**agg_dict).reset_index())

    if medwin_parts:
        medwin = pd.concat(medwin_parts, ignore_index=True)
        agg_d  = {'med_unique_count_medwin':'max', 'med_num_routes_medwin':'max',
                  'med_frequency_complexity_score_medwin':'sum'}
        for cls in HIGH_RISK_CLASSES.keys():
            agg_d[f'med_highrisk_{cls}_medwin'] = 'max'
        medwin = medwin.groupby('hadm_id', as_index=False).agg(
            {k:v for k,v in agg_d.items() if k in medwin.columns})
        hr_cols = [f'med_highrisk_{cls}_medwin' for cls in HIGH_RISK_CLASSES.keys()]
        medwin['med_highrisk_class_count_medwin'] = medwin[
            [c for c in hr_cols if c in medwin.columns]].sum(axis=1)
    else:
        medwin = base[['hadm_id']].copy()
        for c in (['med_unique_count_medwin','med_num_routes_medwin',
                   'med_frequency_complexity_score_medwin',
                   'med_highrisk_class_count_medwin'] +
                  [f'med_highrisk_{cls}_medwin' for cls in HIGH_RISK_CLASSES.keys()]):
            medwin[c] = np.nan

    out = (base[['hadm_id']]
           .merge(presc_agg, on='hadm_id', how='left')
           .merge(medwin,    on='hadm_id', how='left'))
    out['med_total_orders_count_ws'] = out['med_total_orders_count_ws'].fillna(0).astype(int)
    out['med_prn_count_ws']          = out['med_prn_count_ws'].fillna(0).astype(int)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 11) C7 — OMR  (unchanged — already uses pre-admission data only)
# ─────────────────────────────────────────────────────────────────────────────
def build_c7(base: pd.DataFrame) -> pd.DataFrame:
    log.info("C7 — OMR (body measures, unit-normalised) …")
    if not os.path.exists(FILE_PATHS['omr']):
        df = base[['hadm_id']].copy()
        df['omr_weight_kg_last_pre'] = np.nan
        df['omr_height_cm_last_pre'] = np.nan
        df['omr_bmi_last_pre']       = np.nan
        return df

    omr = pd.read_csv(FILE_PATHS['omr'],
                      usecols=['subject_id','chartdate','result_name','result_value'],
                      parse_dates=['chartdate'])
    omr['result_name_lower'] = omr['result_name'].str.lower()
    omr['result_value_num']  = pd.to_numeric(omr['result_value'], errors='coerce')

    omr_w = omr[omr['result_name_lower'].str.contains('weight', na=False)].copy()
    omr_h = omr[omr['result_name_lower'].str.contains('height', na=False)].copy()
    omr_b = omr[omr['result_name_lower'].str.contains('bmi',    na=False)].copy()

    omr_w['result_value_num'] = np.where(
        omr_w['result_value_num'] > 200,
        omr_w['result_value_num'] * 0.453592, omr_w['result_value_num'])
    omr_h['result_value_num'] = np.where(
        omr_h['result_value_num'] < 100,
        omr_h['result_value_num'] * 2.54, omr_h['result_value_num'])

    def pick_closest_pre(sub_df, t_admit):
        t_start = t_admit - pd.Timedelta(days=365)
        sub = sub_df[(sub_df['chartdate'] < t_admit) & (sub_df['chartdate'] > t_start)]
        sub = sub.dropna(subset=['result_value_num'])
        if sub.empty:
            return np.nan
        return float(sub.loc[(t_admit - sub['chartdate']).abs().idxmin(), 'result_value_num'])

    rows = []
    for _, row in tqdm(base[['hadm_id','subject_id','admittime']].iterrows(),
                       total=len(base), desc="  omr"):
        sid = row['subject_id']; t = row['admittime']
        w = pick_closest_pre(omr_w[omr_w['subject_id'] == sid], t)
        h = pick_closest_pre(omr_h[omr_h['subject_id'] == sid], t)
        b = pick_closest_pre(omr_b[omr_b['subject_id'] == sid], t)
        if (pd.isna(b) or b < 10 or b > 100) and not pd.isna(w) and not pd.isna(h) and h > 0:
            b = round(w / ((h / 100.0) ** 2), 1)
        rows.append({'hadm_id': row['hadm_id'],
                     'omr_weight_kg_last_pre': w,
                     'omr_height_cm_last_pre': h,
                     'omr_bmi_last_pre':       b})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# 12) C8 — ICU CHART VITALS + GCS  (unchanged — already scoped to pdw window)
# ─────────────────────────────────────────────────────────────────────────────
def build_c8(base: pd.DataFrame) -> pd.DataFrame:
    log.info("C8 — ICU chart vitals + GCS (chunked, merge+groupby) …")

    if not os.path.exists(FILE_PATHS['chartevents']):
        empty = base[['hadm_id']].copy()
        for v in ALL_VITALS + ['gcs']:
            for suf in ['last_pdw','min_pdw','max_pdw','mean_pdw','std_pdw','slope_pdw']:
                empty[f'vital_{v}_{suf}'] = np.nan
        empty['vital_measure_count_pdw'] = np.nan
        return empty

    times = base[['hadm_id','dischtime']].copy()
    times['hadm_id']   = times['hadm_id'].astype(int)
    times['pdw_start'] = times['dischtime'] - pd.Timedelta(hours=24)

    ALL_ITEMIDS = ALL_VITAL_ITEMIDS | GCS_ITEMIDS
    acc_rows  = []
    gcs_parts = []

    reader = pd.read_csv(
        FILE_PATHS['chartevents'],
        usecols=['hadm_id','itemid','charttime','valuenum'],
        parse_dates=['charttime'],
        chunksize=CHUNK_SIZE, low_memory=False,
    )

    for chunk in tqdm(reader, desc="  chartevents chunks", unit="chunk"):
        chunk = chunk[
            chunk['hadm_id'].notna() &
            chunk['itemid'].isin(ALL_ITEMIDS) &
            chunk['valuenum'].notna()
        ].copy()
        if chunk.empty:
            continue
        chunk['hadm_id'] = chunk['hadm_id'].astype(int)
        chunk = chunk.merge(times, on='hadm_id', how='inner')
        chunk = chunk[(chunk['charttime'] >= chunk['pdw_start']) &
                      (chunk['charttime'] <= chunk['dischtime'])]
        if chunk.empty:
            continue

        gcs_mask  = chunk['itemid'].isin(GCS_ITEMIDS)
        gcs_chunk = chunk[gcs_mask][['hadm_id','charttime','valuenum']].copy()
        if not gcs_chunk.empty:
            gcs_parts.append(gcs_chunk)

        vit_chunk = chunk[~gcs_mask].copy()
        if vit_chunk.empty:
            continue
        vit_chunk['vital'] = vit_chunk['itemid'].map(ITEMID_TO_VITAL)

        f_mask = (vit_chunk['vital'] == 'temp') & (vit_chunk['itemid'] == 223761)
        vit_chunk.loc[f_mask, 'valuenum'] = (
            (vit_chunk.loc[f_mask, 'valuenum'] - 32.0) * 5.0 / 9.0)

        idx_last  = vit_chunk.groupby(['hadm_id','vital'])['charttime'].idxmax()
        idx_first = vit_chunk.groupby(['hadm_id','vital'])['charttime'].idxmin()
        last_v  = vit_chunk.loc[idx_last,  ['hadm_id','vital','valuenum','charttime']].rename(
            columns={'valuenum':'last','charttime':'last_time'})
        first_v = vit_chunk.loc[idx_first, ['hadm_id','vital','valuenum','charttime']].rename(
            columns={'valuenum':'first_val','charttime':'first_time'})

        agg = (vit_chunk.groupby(['hadm_id','vital'])
                        .agg(v_min=('valuenum','min'), v_max=('valuenum','max'),
                             v_mean=('valuenum','mean'), v_std=('valuenum','std'),
                             v_cnt=('valuenum','count'))
                        .reset_index()
               ).merge(last_v,  on=['hadm_id','vital'], how='left') \
                .merge(first_v, on=['hadm_id','vital'], how='left')
        acc_rows.append(agg)

    if not acc_rows:
        return base[['hadm_id']].copy()

    vit = pd.concat(acc_rows, ignore_index=True)
    vit = vit.groupby(['hadm_id','vital'], as_index=False).agg(
        v_min=('v_min','min'), v_max=('v_max','max'),
        v_mean=('v_mean','mean'), v_std=('v_std','mean'),
        v_cnt=('v_cnt','sum'),
        last=('last','last'), last_time=('last_time','last'),
        first_val=('first_val','first'), first_time=('first_time','first'),
    )
    dt_h = (vit['last_time'] - vit['first_time']).dt.total_seconds() / 3600.0
    vit['slope'] = np.where(
        (vit['v_cnt'] >= 2) & (dt_h > 0),
        (vit['last'] - vit['first_val']) / dt_h, np.nan)

    parts = []
    for metric, suf in [('last','last_pdw'),('v_min','min_pdw'),('v_max','max_pdw'),
                        ('v_mean','mean_pdw'),('v_std','std_pdw'),('slope','slope_pdw')]:
        p = vit.pivot(index='hadm_id', columns='vital', values=metric)
        p.columns = [f'vital_{v}_{suf}' for v in p.columns]
        parts.append(p)

    cnt = vit.groupby('hadm_id')['v_cnt'].sum().rename('vital_measure_count_pdw').to_frame()
    parts.append(cnt)

    if gcs_parts:
        gcs = pd.concat(gcs_parts, ignore_index=True)
        gcs_total = (gcs.groupby(['hadm_id','charttime'])['valuenum']
                        .sum().reset_index(name='gcs_total'))
        gcs_agg = gcs_total.groupby('hadm_id').agg(
            gcs_last=('gcs_total','last'),   gcs_min=('gcs_total','min'),
            gcs_max=('gcs_total','max'),     gcs_std=('gcs_total','std'),
            gcs_cnt=('gcs_total','count'),
            gcs_last_time=('charttime','last'),
            gcs_first_val=('gcs_total','first'),
            gcs_first_time=('charttime','first'),
        ).reset_index()
        dt_g = (gcs_agg['gcs_last_time'] -
                gcs_agg['gcs_first_time']).dt.total_seconds() / 3600.0
        gcs_agg['vital_gcs_slope_pdw'] = np.where(
            (gcs_agg['gcs_cnt'] >= 2) & (dt_g > 0),
            (gcs_agg['gcs_last'] - gcs_agg['gcs_first_val']) / dt_g, np.nan)
        gcs_agg = gcs_agg.rename(columns={
            'gcs_last':'vital_gcs_last_pdw', 'gcs_min':'vital_gcs_min_pdw',
            'gcs_max':'vital_gcs_max_pdw',   'gcs_std':'vital_gcs_std_pdw',
        })[['hadm_id','vital_gcs_last_pdw','vital_gcs_min_pdw',
             'vital_gcs_max_pdw','vital_gcs_std_pdw','vital_gcs_slope_pdw']]
        parts.append(gcs_agg.set_index('hadm_id'))

    out = pd.concat(parts, axis=1).reset_index()
    out = base[['hadm_id']].merge(out, on='hadm_id', how='left')
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 13) D — DISCHARGE NOTE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
_RX_FOLLOWUP    = re.compile(r'follow.?up|appointment|pcp|clinic', re.I)
_RX_HOME_HEALTH = re.compile(r'vna|home nursing|home pt|visiting nurse', re.I)
_RX_LIVES_ALONE = re.compile(r'lives alone', re.I)
_RX_CAREGIVER   = re.compile(r'caregiver|family|wife|husband|son|daughter.{0,20}assist', re.I)
_RX_FUNC_ASSIST = re.compile(r'needs assistance|walker|wheelchair|adls|pt recommends', re.I)
_RX_COGN_IMP    = re.compile(r'dementia|confused|delirium|poor historian|not oriented', re.I)

# REPLACE: _RX_STABLE, _RX_RESOLVED, _RX_WORSENING removed (LEAK-08)
# note_stable_at_discharge → "hemodynamically stable", "on room air" = outcome proxy
# note_resolution_score    → counts of "resolved/improving" vs "worsening" = outcome proxy


def _extract_note_features(text: str) -> dict:
    """
    LEAK-08: REPLACE note_stable_at_discharge → REMOVED
             REPLACE note_resolution_score    → REMOVED

             Both features fire on clinical language that summarises the outcome of the
             admission (e.g. "patient is stable", "infection resolved", "symptoms worsening").
             They are leaky proxies for the target regardless of whether it is mortality,
             readmission, or discharge disposition.

             KEPT: social and logistic features that describe the patient's situation
             and post-discharge plan — not the clinical outcome of this stay.
    """
    empty = {
        'note_has_followup_instructions': 0,
        'note_has_home_health':           0,
        'note_lives_alone':               0,
        'note_has_caregiver':             0,
        'note_function_needs_assistance': 0,
        'note_cognitive_impairment':      0,
        'note_days_to_followup_est':      np.nan,
    }
    if pd.isna(text) or not isinstance(text, str):
        return empty

    _RX_FU_DAYS = re.compile(r'follow.?up\s+in\s+(\d+)\s*(day|week)', re.I)

    d = {
        'note_has_followup_instructions': int(bool(_RX_FOLLOWUP.search(text))),
        'note_has_home_health':           int(bool(_RX_HOME_HEALTH.search(text))),
        'note_lives_alone':               int(bool(_RX_LIVES_ALONE.search(text))),
        'note_has_caregiver':             int(bool(_RX_CAREGIVER.search(text))),
        'note_function_needs_assistance': int(bool(_RX_FUNC_ASSIST.search(text))),
        'note_cognitive_impairment':      int(bool(_RX_COGN_IMP.search(text))),
    }
    m = _RX_FU_DAYS.search(text)
    d['note_days_to_followup_est'] = (
        float(int(m.group(1)) * (7 if 'week' in m.group(2).lower() else 1))
        if m else np.nan)
    return d


def build_d(base: pd.DataFrame) -> pd.DataFrame:
    log.info("D — Discharge note features …")
    if not os.path.exists(FILE_PATHS['discharge']):
        empty = base[['hadm_id']].copy()
        for c in ['note_has_followup_instructions','note_has_home_health',
                  'note_lives_alone','note_has_caregiver',
                  'note_function_needs_assistance','note_cognitive_impairment',
                  'note_days_to_followup_est']:
            empty[c] = np.nan
        return empty

    note_parts = []
    base_hadm  = set(base['hadm_id'].astype(int).unique())
    reader     = pd.read_csv(FILE_PATHS['discharge'], usecols=['hadm_id','text'],
                             chunksize=CHUNK_SIZE, low_memory=False)
    for chunk in tqdm(reader, desc="  notes chunks", unit="chunk"):
        chunk = chunk[chunk['hadm_id'].notna()].copy()
        chunk['hadm_id'] = chunk['hadm_id'].astype(int)
        chunk = chunk[chunk['hadm_id'].isin(base_hadm)]
        if not chunk.empty:
            note_parts.append(chunk)

    if not note_parts:
        return base[['hadm_id']].copy()

    notes   = (pd.concat(note_parts, ignore_index=True)
                 .drop_duplicates(subset=['hadm_id'], keep='last'))
    feat_df = pd.DataFrame(list(notes['text'].apply(_extract_note_features)))
    out     = pd.concat([notes[['hadm_id']].reset_index(drop=True),
                         feat_df.reset_index(drop=True)], axis=1)
    return base[['hadm_id']].merge(out, on='hadm_id', how='left')

# ─────────────────────────────────────────────────────────────────────────────
# 14) MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info(" MIMIC-IV Feature Extraction Pipeline v1.3 (leakage-fixed)")
    log.info("=" * 60)

    for key, path in FILE_PATHS.items():
        status = "OK" if os.path.exists(path) else "MISSING"
        log.info(f"  {status}  {key}: {path}")

    base = load_base_tables()

    c1 = build_c1(base)
    c2 = build_c2(base)
    c3 = build_c3(base)
    c4 = build_c4(base)
    c5 = build_c5(base)
    c6 = build_c6(base)
    c7 = build_c7(base)
    c8 = build_c8(base)
    d  = build_d(base)

    log.info("Merging all feature blocks …")
    final = c1.copy()
    for name, block in [
        ('C2-util',   c2),
        ('C3-trans',  c3),
        ('C4-dx',     c4),
        ('C5-labs',   c5),
        ('C6-meds',   c6),
        ('C7-omr',    c7),
        ('C8-vitals', c8),
        ('D-notes',   d),
    ]:
        before   = set(final.columns)
        final    = final.merge(block, on='hadm_id', how='left')
        new_cols = set(final.columns) - before - {'hadm_id'}
        log.info(f"  {name}: +{len(new_cols)} cols (total={len(final.columns)})")

    final = final.drop_duplicates(subset=['hadm_id'])
    log.info(f"Final shape: {final.shape[0]:,} rows x {final.shape[1]:,} cols")

    miss = final.isna().mean().sort_values(ascending=False)
    high = miss[miss > 0.5]
    if not high.empty:
        log.info("Columns with >50% missing:")
        for col, pct in high.items():
            log.info(f"  {col}: {pct*100:.1f}%")

    log.info(f"Saving to {OUTPUT_FILE} …")
    final.to_csv(OUTPUT_FILE, index=False, float_format='%.6g')
    log.info("Done.")
    log.info(f"  Output: {OUTPUT_FILE}")
    log.info(f"  Rows:   {len(final):,}")
    log.info(f"  Cols:   {len(final.columns):,}")
    return final


if __name__ == '__main__':
    _ = main()