"""
CodeAlpha Internship - Task 3: Music Generation with AI
Step 1 — Preprocessing

Parses MIDI files with music21 and converts them into a flat sequence of
note/chord tokens suitable for LSTM training.

- Notes are stored as pitch strings, e.g. "E4"
- Chords are stored as dot-joined pitch classes, e.g. "4.7.11"
- The token list is saved to data/notes.pkl

Run with:
    python preprocess.py
"""

import pickle

from music21 import chord, converter, instrument, note
from tqdm import tqdm

import config


def find_midi_files():
    """Recursively find MIDI files, skipping excluded folders."""
    files = [
        p
        for p in sorted(config.MIDI_DIR.rglob("*.mid"))
        if not (config.EXCLUDE_DIRS & set(part.lower() for part in p.parts))
    ]
    files += [
        p
        for p in sorted(config.MIDI_DIR.rglob("*.midi"))
        if not (config.EXCLUDE_DIRS & set(part.lower() for part in p.parts))
    ]
    if config.MAX_FILES:
        files = files[: config.MAX_FILES]
    return files


def parse_midi_file(path):
    """Extract a list of note/chord tokens from one MIDI file."""
    tokens = []
    midi = converter.parse(path)

    # If the file has multiple instrument parts, use the first (melody) part;
    # otherwise take the flat notes directly.
    try:
        parts = instrument.partitionByInstrument(midi)
        elements = parts.parts[0].recurse() if parts else midi.flatten().notes
    except Exception:
        elements = midi.flatten().notes

    for element in elements:
        if isinstance(element, note.Note):
            tokens.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            tokens.append(".".join(str(n) for n in element.normalOrder))
    return tokens


def main():
    midi_files = find_midi_files()
    if not midi_files:
        raise SystemExit(
            f"❌ No MIDI files found in {config.MIDI_DIR}. "
            "Please put your dataset there first."
        )

    print(f"🎵 Found {len(midi_files)} MIDI files. Parsing with music21...")

    notes = []
    failed = 0
    for path in tqdm(midi_files, unit="file"):
        try:
            notes.extend(parse_midi_file(path))
        except Exception:
            failed += 1

    if not notes:
        raise SystemExit("❌ Could not extract any notes from the dataset.")

    config.NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.NOTES_FILE, "wb") as f:
        pickle.dump(notes, f)

    vocab = sorted(set(notes))
    print("\n✅ Preprocessing complete!")
    print(f"   Total tokens (notes/chords): {len(notes):,}")
    print(f"   Vocabulary size:             {len(vocab):,}")
    print(f"   Failed files:                {failed}")
    print(f"   Saved to: {config.NOTES_FILE}")


if __name__ == "__main__":
    main()
