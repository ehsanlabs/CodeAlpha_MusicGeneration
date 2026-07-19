"""
CodeAlpha Internship - Task 3: Music Generation with AI
Step 2 — Model Training

Builds and trains an LSTM network that learns musical patterns from the
preprocessed note sequences and predicts the next note/chord.

Architecture:
    Embedding -> LSTM(256, return_sequences) -> Dropout
              -> LSTM(256) -> Dropout -> Dense -> Softmax

Run with:
    python train.py
"""

import pickle

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input
from tensorflow.keras.models import Sequential

import config


def load_notes():
    if not config.NOTES_FILE.exists():
        raise SystemExit("❌ notes.pkl not found. Run `python preprocess.py` first.")
    with open(config.NOTES_FILE, "rb") as f:
        return pickle.load(f)


def build_sequences(notes):
    """Convert the flat token list into (input sequence -> next token) pairs."""
    vocab = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(vocab)}

    seq_len = config.SEQUENCE_LENGTH
    inputs, targets = [], []
    for i in range(len(notes) - seq_len):
        seq_in = notes[i : i + seq_len]
        seq_out = notes[i + seq_len]
        inputs.append([note_to_int[n] for n in seq_in])
        targets.append(note_to_int[seq_out])

    X = np.array(inputs, dtype=np.int32)
    y = np.array(targets, dtype=np.int32)
    return X, y, vocab, note_to_int


def build_model(vocab_size):
    model = Sequential(
        [
            Input(shape=(config.SEQUENCE_LENGTH,)),
            Embedding(vocab_size, config.EMBEDDING_DIM),
            LSTM(config.LSTM_UNITS, return_sequences=True),
            Dropout(config.DROPOUT),
            LSTM(config.LSTM_UNITS),
            Dropout(config.DROPOUT),
            Dense(vocab_size, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def main():
    notes = load_notes()
    X, y, vocab, note_to_int = build_sequences(notes)

    print(f"🎼 Training data: {X.shape[0]:,} sequences | vocab size: {len(vocab)}")

    model = build_model(len(vocab))
    model.summary()

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Save the checkpoint that generalizes best (lowest validation loss),
        # not the one that memorizes the training data most.
        ModelCheckpoint(
            str(config.CHECKPOINT_FILE),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    model.fit(
        X,
        y,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=callbacks,
    )

    model.save(config.MODEL_FILE)

    # Save the vocabulary mapping alongside the model for generation
    with open(config.MODEL_DIR / "vocab.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "note_to_int": note_to_int}, f)

    print(f"\n✅ Training complete! Model saved to {config.MODEL_FILE}")


if __name__ == "__main__":
    main()
