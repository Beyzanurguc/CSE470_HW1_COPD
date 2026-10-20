"""Central configuration: every parameter used by the pipeline lives here.

Changing a value here changes it for inspect_data.py, train.py and app.py alike,
so the paper/README can quote one set of numbers.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"            # put the ICBHI / Kaggle dataset here (any sub-folder)
META_DIR = ROOT / "metadata"        # official split + diagnosis files shipped with the code
OUT_DIR = ROOT / "outputs"          # metrics, tables
FIG_DIR = OUT_DIR / "figures"       # all figures
CACHE_DIR = OUT_DIR / "cache"       # cached feature matrices (safe to delete)
MODEL_DIR = ROOT / "models"         # trained models used by the demo app

# ---------------------------------------------------------------- task
POSITIVE_CLASS = "COPD"             # label 1
NEGATIVE_CLASS = "non-COPD"         # label 0 (Healthy + all other diagnoses)
CLASS_NAMES = [NEGATIVE_CLASS, POSITIVE_CLASS]
RANDOM_SEED = 42

# ---------------------------------------------------------------- preprocessing
TARGET_SR = 4000        # Hz. Lowest native rate in ICBHI is 4 kHz -> common bandwidth 0-2 kHz for every device
BANDPASS_HZ = (100.0, 1800.0)   # removes heart sounds / DC / handling noise (<100 Hz), stays below Nyquist (2 kHz)
FILTER_ORDER = 4                # Butterworth order, applied forward-backward (zero phase)
DURATION_S = 20.0               # same fixed length as the paper (median ICBHI recording ~19.9 s)

# ---------------------------------------------------------------- features
N_FFT = 512             # 128 ms analysis window at 4 kHz
HOP_LENGTH = 128        # 32 ms hop -> 626 frames for 20 s
N_MELS = 64
N_MFCC = 40             # paper uses n = 40 for every feature
FMIN, FMAX = 50.0, 2000.0
FEATURE_VERSION = 2       # bump when features.extract() changes (invalidates the feature cache)

# ---------------------------------------------------------------- augmentation (training set only)
AUGMENT_CLASS = 0               # the paper augments the minority (non-COPD) class
LOUDNESS_RANGE = (0.6, 1.4)     # gain factor (paper: 0.6-1.4)
SPEED_RANGE = (0.8, 1.25)       # paper: 0.5-2.0x; narrowed so breathing stays physiologically plausible
MASK_FRACTION = 0.10            # length of the zeroed time span
MAX_SHIFT_S = 5.0               # circular time shift

# ---------------------------------------------------------------- evaluation
CV_FOLDS = 4                    # patient-grouped folds used for hyper-parameter search on the training set
DEVICE_MATCHED = ["Meditron", "LittC2SE"]   # the only devices that contain both classes
DEVICE_MATCHED_FOLDS = 5

for _d in (OUT_DIR, FIG_DIR, CACHE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)
