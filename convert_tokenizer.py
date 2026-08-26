import os
import sys
import json
import pickle
import types

import tensorflow as tf


# ============================================================
# Compatibility patch for old Keras tokenizer pickle
# ============================================================

try:
    from tensorflow.keras.preprocessing.text import Tokenizer
except ImportError as e:
    print("Could not import TensorFlow Keras Tokenizer.")
    print(e)
    sys.exit(1)


# Create fake old Keras module path
keras_module = types.ModuleType("keras")
preprocessing_module = types.ModuleType("keras.preprocessing")
text_module = types.ModuleType("keras.preprocessing.text")

text_module.Tokenizer = Tokenizer

preprocessing_module.text = text_module
keras_module.preprocessing = preprocessing_module

sys.modules["keras"] = keras_module
sys.modules["keras.preprocessing"] = preprocessing_module
sys.modules["keras.preprocessing.text"] = text_module


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pickle_path = os.path.join(
    BASE_DIR,
    "tokenizer.pickle"
)

json_path = os.path.join(
    BASE_DIR,
    "tokenizer.json"
)


# ============================================================
# Load old tokenizer
# ============================================================

print("Loading old tokenizer...")

with open(pickle_path, "rb") as f:
    tokenizer = pickle.load(f)

print("Old tokenizer loaded successfully.")
print("Tokenizer type:", type(tokenizer))


# ============================================================
# Convert to JSON
# ============================================================

tokenizer_json = tokenizer.to_json()

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(tokenizer_json)


print()
print("==============================================")
print("Tokenizer conversion successful!")
print("==============================================")
print("Created:")
print(json_path)