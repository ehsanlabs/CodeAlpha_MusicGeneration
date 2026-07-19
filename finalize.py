"""
Recover from an interrupted training run:
- loads checkpoint.keras (already trained)
- saves it as the final model file
- rebuilds and saves vocab.pkl (deterministic, so it matches training exactly)
"""

import pickle
from tensorflow.keras.models import load_model

import config

# 1. Load the checkpoint and save it as the final model
model = load_model(config.CHECKPOINT_FILE)
model.save(config.MODEL_FILE)
print(f"✅ Saved final model to {config.MODEL_FILE}")

# 2. Rebuild vocab (same logic as train.py — sorted(set(notes)) is deterministic)
with open(config.NOTES_FILE, "rb") as f:
    notes = pickle.load(f)

vocab = sorted(set(notes))
note_to_int = {n: i for i, n in enumerate(vocab)}

with open(config.MODEL_DIR / "vocab.pkl", "wb") as f:
    pickle.dump({"vocab": vocab, "note_to_int": note_to_int}, f)
print(f"✅ Saved vocab.pkl ({len(vocab)} tokens)")