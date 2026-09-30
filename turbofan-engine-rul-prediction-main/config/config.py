"""
Configuration file for C-MAPSS Turbofan Engine Remaining Useful Life (RUL) prediction.
Central source of truth for all file paths, column names, sensor metadata,
and model hyperparameters.
"""

from pathlib import Path

# Project root directory (calculated dynamically from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

# Create directories if they do not exist
for dir_path in [PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, DOCS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data column specifications (26 columns total)
INDEX_COLS = ["unit_number", "time_cycles"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Descriptive mapping for 21 sensors based on NASA PHM '08 reference specification
SENSOR_INFO = {
    "s_1": {"desc": "Fan inlet temperature", "unit": "°R"},
    "s_2": {"desc": "LPC outlet temperature", "unit": "°R"},
    "s_3": {"desc": "HPC outlet temperature", "unit": "°R"},
    "s_4": {"desc": "LPT outlet temperature", "unit": "°R"},
    "s_5": {"desc": "Fan inlet pressure", "unit": "psia"},
    "s_6": {"desc": "Bypass duct pressure", "unit": "psia"},
    "s_7": {"desc": "HPC outlet pressure", "unit": "psia"},
    "s_8": {"desc": "Physical fan speed", "unit": "rpm"},
    "s_9": {"desc": "Physical core speed", "unit": "rpm"},
    "s_10": {"desc": "Engine pressure ratio", "unit": "--"},
    "s_11": {"desc": "HPC outlet static pressure", "unit": "psia"},
    "s_12": {"desc": "Ratio of fuel flow to Ps30", "unit": "pps/psia"},
    "s_13": {"desc": "Corrected fan speed", "unit": "rpm"},
    "s_14": {"desc": "Corrected core speed", "unit": "rpm"},
    "s_15": {"desc": "Bypass ratio", "unit": "--"},
    "s_16": {"desc": "Burner fuel-air ratio", "unit": "--"},
    "s_17": {"desc": "Bleed enthalpy", "unit": "--"},
    "s_18": {"desc": "Demanded fan speed", "unit": "rpm"},
    "s_19": {"desc": "Demanded corrected fan speed", "unit": "rpm"},
    "s_20": {"desc": "HPT coolant bleed", "unit": "lbm/s"},
    "s_21": {"desc": "LPT coolant bleed", "unit": "lbm/s"},
}

# Invariant (zero or near-zero variance) sensors in FD001 (Sea Level, single operating condition)
# Dropping these 7 sensors leaves the canonical 14 degradation sensors used in literature
FD001_DROP_SENSORS = ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]
FD001_DROP_SETTINGS = ["setting_1", "setting_2", "setting_3"]  # Settings have near-zero variation in FD001

# Canonical 14 informative sensors kept for modeling in FD001
FD001_ACTIVE_SENSORS = [s for s in SENSOR_COLS if s not in FD001_DROP_SENSORS]

# Modeling Hyperparameters & Constants
RUL_CAP = 125          # Piecewise linear RUL threshold (cycles)
WINDOW_SIZE = 30       # Sliding sequence window length (cycles)
RANDOM_SEED = 42       # Seed for reproducibility
VAL_SIZE = 0.2         # Validation fraction (split strictly by engine unit_number)
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 0.001

# Subsets metadata
DATASETS = {
    "FD001": {
        "train": RAW_DATA_DIR / "train_FD001.txt",
        "test": RAW_DATA_DIR / "test_FD001.txt",
        "rul": RAW_DATA_DIR / "RUL_FD001.txt",
        "conditions": 1,
        "faults": "HPC Degradation",
        "train_units": 100,
        "test_units": 100
    },
    "FD002": {
        "train": RAW_DATA_DIR / "train_FD002.txt",
        "test": RAW_DATA_DIR / "test_FD002.txt",
        "rul": RAW_DATA_DIR / "RUL_FD002.txt",
        "conditions": 6,
        "faults": "HPC Degradation",
        "train_units": 260,
        "test_units": 259
    },
    "FD003": {
        "train": RAW_DATA_DIR / "train_FD003.txt",
        "test": RAW_DATA_DIR / "test_FD003.txt",
        "rul": RAW_DATA_DIR / "RUL_FD003.txt",
        "conditions": 1,
        "faults": "HPC + Fan Degradation",
        "train_units": 100,
        "test_units": 100
    },
    "FD004": {
        "train": RAW_DATA_DIR / "train_FD004.txt",
        "test": RAW_DATA_DIR / "test_FD004.txt",
        "rul": RAW_DATA_DIR / "RUL_FD004.txt",
        "conditions": 6,
        "faults": "HPC + Fan Degradation",
        "train_units": 248,
        "test_units": 248
    },
}
