"""
CodeAlpha Internship - Task 3: Music Generation with AI
Step 3 — Music Generation

Loads the trained LSTM model, generates a brand-new sequence of notes/chords
using temperature sampling, and saves it as a playable MIDI file.

Run with:
    python generate.py
    python generate.py --notes 300 --temperature 1.2
"""

import argparse
import pickle
import random
from datetime import datetime

import numpy as np
from music21 import chord, instrument, note, stream
from tensorflow.keras.models import load_model

import config


def load_artifacts():
    if not config.MODEL_FILE.exists():
        raise SystemExit("❌ Trained model not found. Run `python train.py` first.")
    model = load_model(config.MODEL_FILE)
    with open(config.MODEL_DIR / "vocab.pkl", "rb") as f:
        data = pickle.load(f)
    with open(config.NOTES_FILE, "rb") as f:
        notes = pickle.load(f)
    return model, data["vocab"], data["note_to_int"], notes


def sample_with_temperature(probabilities, temperature):
    """Sample the next token index using temperature scaling."""
    if temperature <= 0:
        return int(np.argmax(probabilities))
    logits = np.log(probabilities + 1e-9) / temperature
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    return int(np.random.choice(len(probs), p=probs))


def generate_tokens(model, vocab, note_to_int, notes, n_notes, temperature):
    """Generate n_notes new tokens starting from a random seed sequence."""
    int_to_note = {i: n for n, i in note_to_int.items()}
    seq_len = config.SEQUENCE_LENGTH

    # Random seed sequence from the real training data
    start = random.randint(0, len(notes) - seq_len - 1)
    pattern = [note_to_int[n] for n in notes[start : start + seq_len]]

    generated = []
    for _ in range(n_notes):
        x = np.array(pattern[-seq_len:], dtype=np.int32).reshape(1, seq_len)
        probabilities = model.predict(x, verbose=0)[0]
        index = sample_with_temperature(probabilities, temperature)
        generated.append(int_to_note[index])
        pattern.append(index)

    return generated


def tokens_to_midi(tokens, output_path):
    """Convert generated tokens into a MIDI file using music21."""
    offset = 0.0
    elements = []

    for token in tokens:
        if ("." in token) or token.isdigit():
            # Chord token, e.g. "4.7.11"
            pitches = [int(p) for p in token.split(".")]
            chord_notes = [note.Note(p) for p in pitches]
            for n in chord_notes:
                n.storedInstrument = instrument.Piano()
            new_element = chord.Chord(chord_notes)
        else:
            # Single note token, e.g. "E4"
            new_element = note.Note(token)
            new_element.storedInstrument = instrument.Piano()

        new_element.offset = offset
        elements.append(new_element)
        offset += config.NOTE_STEP

    midi_stream = stream.Stream(elements)
    midi_stream.write("midi", fp=str(output_path))


def main():
    parser = argparse.ArgumentParser(description="Generate music with the trained LSTM")
    parser.add_argument("--notes", type=int, default=config.GENERATE_NOTES,
                        help="number of notes/chords to generate")
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE,
                        help=">1.0 = more creative, <1.0 = safer")
    parser.add_argument("--count", type=int, default=1,
                        help="how many pieces to generate")
    args = parser.parse_args()

    model, vocab, note_to_int, notes = load_artifacts()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        print(f"🎹 Generating piece {i + 1}/{args.count} "
              f"({args.notes} notes, temperature={args.temperature})...")
        tokens = generate_tokens(
            model, vocab, note_to_int, notes, args.notes, args.temperature
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = config.OUTPUT_DIR / f"generated_{timestamp}_{i + 1}.mid"
        tokens_to_midi(tokens, output_path)
        print(f"✅ Saved: {output_path}")

    print("\n🎧 Open the .mid files in any media player, MuseScore, or an "
          "online MIDI player to listen!")


if __name__ == "__main__":
    main()
