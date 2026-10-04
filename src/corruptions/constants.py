"""Constants shared by training code and the web backend (no torch dependency).

All values are taken literally from the assignment PDF (p.3).
"""

IMAGE_SIZE = 128
SEED = 42

# Class index order is fixed by the assignment: 0 clean, 1 salt, 2 blur, 3 occlusion.
CLASS_NAMES = ["clean", "salt_pepper", "blur", "occlusion"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
CLASS_DISPLAY = {
    "clean": "Clean",
    "salt_pepper": "Salt-and-pepper",
    "blur": "Gaussian blur",
    "occlusion": "Rectangular occlusion",
}

# ---- Training ranges (sampled at runtime, PDF p.3) ----
TRAIN_SALT_PROB_RANGE = (0.02, 0.15)
TRAIN_BLUR_KERNELS = (3, 5, 7)
TRAIN_BLUR_SIGMA_RANGE = (0.5, 2.5)
TRAIN_OCC_NUM_RECTS = (1, 2, 3)
TRAIN_OCC_COVERAGE_RANGE = (0.10, 0.35)

# Accepted deviation between requested and achieved union coverage.
OCC_COVERAGE_TOLERANCE = 0.015

# ---- Fixed final-test severity levels (PDF p.3) ----
SEVERITY_LEVELS = ["low", "medium", "high"]
TEST_LEVELS = {
    "salt_pepper": {
        "low": {"prob": 0.03},
        "medium": {"prob": 0.08},
        "high": {"prob": 0.15},
    },
    "blur": {
        "low": {"kernel": 3, "sigma": 0.7},
        "medium": {"kernel": 5, "sigma": 1.5},
        "high": {"kernel": 7, "sigma": 2.5},
    },
    "occlusion": {
        "low": {"coverage": 0.10, "num_rects": 1},
        "medium": {"coverage": 0.20, "num_rects": 2},
        "high": {"coverage": 0.35, "num_rects": 3},
    },
}

# FS2K styles (Task 4). The UI names them Style 1/2/3; the model uses indices 0/1/2.
NUM_STYLES = 3
STYLE_NAMES = ["Style 1", "Style 2", "Style 3"]
