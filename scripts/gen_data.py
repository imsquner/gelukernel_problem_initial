import numpy as np
import os
import sys

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None

sys.path.insert(0, os.path.dirname(__file__))
from Gelu import impl

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)


# --- Case 0 ---
np.random.seed(42 + 0)
os.makedirs("input/case0", exist_ok=True)
input_x = np.random.uniform(low=-1.0, high=1.0, size=(32)).astype(np.float16)
input_x.tofile("input/case0/input_x.bin")



os.makedirs("output/golden_case0", exist_ok=True)
golden = impl(input_x)
if golden is not None:
    golden.tofile("output/golden_case0/golden_output.bin")

print(f"Generated test data and golden output for 1 cases.")
