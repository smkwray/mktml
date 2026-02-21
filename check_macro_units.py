import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from macro_loader import get_macro_features

features = get_macro_features()
print("Macro Feature Samples:")
for k in ['macro_VIXCLS', 'macro_T10Y2Y', 'macro_WALCL', 'macro_WALCL_chg1m']:
    if k in features:
        print(f"  {k}: {features[k]:.4f}")
    else:
        print(f"  {k}: MISSING")
