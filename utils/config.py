"""Application-wide configuration and default hyperparameters."""
import os

# Absolute paths anchored to this file's location
_UTILS_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(_UTILS_DIR)                       # workspace root
DATA_DIR     = os.path.join(ROOT_DIR, "ImageandvideoI3D")        # inner data folder

# MS-ASL dataset JSON files
MSASL_DIR   = os.path.join(DATA_DIR, "MSASL")
CLASSES_JSON = os.path.join(MSASL_DIR, "MSASL_classes.json")
TRAIN_JSON   = os.path.join(MSASL_DIR, "MSASL_train.json")
VAL_JSON     = os.path.join(MSASL_DIR, "MSASL_val.json")
TEST_JSON    = os.path.join(MSASL_DIR, "MSASL_test.json")

# Pipeline output directories
ASL_VIDEOS_DIR = os.path.join(DATA_DIR, "ASL20")
ASL_FRAMES_DIR = os.path.join(DATA_DIR, "ASL20_frames")

# Model artefact
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "trained_model.pth")

# Error log (video download failures)
ERROR_LOG_PATH = os.path.join(DATA_DIR, "error_log.txt")

# ── Training hyperparameters ──────────────────────────────────────────────────
NUM_CLASSES              = 100
BATCH_SIZE               = 4
MAX_FRAMES               = 64
NUM_EPOCHS               = 20
LEARNING_RATE            = 5e-5
DROPOUT_RATE             = 0.5
EARLY_STOPPING_PATIENCE  = 5

# ── Video / frame settings ────────────────────────────────────────────────────
FRAME_SIZE      = (224, 224)   # (H, W) fed to the model
SHORT_SIDE_SIZE = 256          # resize shorter side to this before cropping

# Normalisation constants matching R3D-18 Kinetics-400 pre-training
MEAN = [0.43216, 0.394666, 0.37645]
STD  = [0.22803, 0.22145,  0.216989]
