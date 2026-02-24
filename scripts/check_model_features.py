import os
import sys

import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ml_engine import FEATURE_COLS
from config import MODEL_PATH_5D, MODEL_PATH_10D, MODEL_PATH_30D

print(f"Current System FEATURE_COLS ({len(FEATURE_COLS)}):")
print(FEATURE_COLS)
print("-" * 50)

for name, path in [("5D", MODEL_PATH_5D), ("10D", MODEL_PATH_10D), ("30D", MODEL_PATH_30D)]:
    print(f"Checking {name} model at {path}...")
    try:
        model = joblib.load(path)
        # Random Forest / GBM store feature names in different places depending on version/type
        feat_names = None
        if hasattr(model, 'feature_names_in_'):
            feat_names = list(model.feature_names_in_)
        elif hasattr(model, 'estimators_'):
            # It's an ensemble, check the first estimator
            est = model.estimators_[0]
            if hasattr(est, 'feature_names_in_'):
                feat_names = list(est.feature_names_in_)
        
        if feat_names:
            print(f"  Model Features ({len(feat_names)}):")
            print(feat_names)
            # Check for differences
            missing = set(feat_names) - set(FEATURE_COLS)
            extra = set(FEATURE_COLS) - set(feat_names)
            if missing: print(f"  !!! MISSING from system: {missing}")
            if extra: print(f"  !!! EXTRA in system: {extra}")
            if feat_names != FEATURE_COLS:
                print("  !!! ORDER MISMATCH or COUNT MISMATCH")
        else:
            print("  Could not find feature names in model object.")
            
    except Exception as e:
        print(f"  Error loading {name}: {e}")
    print("-" * 50)
