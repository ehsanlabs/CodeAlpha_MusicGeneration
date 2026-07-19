"""
CodeAlpha Internship - Task 3: Music Generation with AI
Central configuration for the whole pipeline.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent

# ----------------------------- Data -----------------------------
# Root folder that contains MIDI files (searched recursively)
MIDI_DIR = BASE_DIR / "data" / "midi"

# Skip the "chords" accompaniment-only folder of the Nottingham dataset
# (the main MIDI files already contain melody + chords)
EXCLUDE_DIRS = {"chords"}

# Limit how many MIDI files to parse (None = all). Parsing is the slowest
# step, and ~300 folk tunes are plenty for a good model on CPU.
MAX_FILES = 300

# Where preprocessed data is stored
NOTES_FILE = BASE_DIR / "data" / "notes.pkl"

# ----------------------------- Model -----------------------------
SEQUENCE_LENGTH = 40          # input notes used to predict the next note
EMBEDDING_DIM = 96
LSTM_UNITS = 256
DROPOUT = 0.3

MODEL_DIR = BASE_DIR / "model"
MODEL_FILE = MODEL_DIR / "music_lstm.keras"
CHECKPOINT_FILE = MODEL_DIR / "checkpoint.keras"

# ----------------------------- Training -----------------------------
EPOCHS = 30
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 4

# ----------------------------- Generation -----------------------------
OUTPUT_DIR = BASE_DIR / "output"
GENERATE_NOTES = 200          # how many notes/chords to generate
TEMPERATURE = 1.0             # >1.0 = more random/creative, <1.0 = safer
NOTE_STEP = 0.5               # time (in quarter lengths) between notes
