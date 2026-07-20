"""
CodeAlpha Internship - Task 3: Music Generation (Crystal Blue UI)
-----------------------------------------------------------------
Web interface for the trained LSTM music model:
- Generate new music with one click (notes count + creativity controls)
- Play generated MIDI directly in the browser (piano-roll visualizer)
- Download .mid files

Run with:
    python web_app.py
Then open:  http://localhost:5002
"""

from datetime import datetime

from flask import Flask, jsonify, render_template, request, send_from_directory

import config
from generate import generate_tokens, load_artifacts, tokens_to_midi

app = Flask(__name__)

print("🎵 Loading trained LSTM model + vocabulary...")
model, vocab, note_to_int, notes = load_artifacts()
print(f"   Model ready | vocab: {len(vocab)} tokens | corpus: {len(notes):,} notes")


def list_generated():
    """Return generated MIDI files, newest first."""
    if not config.OUTPUT_DIR.exists():
        return []
    files = sorted(config.OUTPUT_DIR.glob("*.mid"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return [f.name for f in files[:12]]


@app.route("/")
def index():
    return render_template(
        "index.html",
        vocab_size=len(vocab),
        corpus_size=f"{len(notes):,}",
        seq_len=config.SEQUENCE_LENGTH,
        files=list_generated(),
    )


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True)
    n_notes = max(50, min(400, int(data.get("notes", 200))))
    temperature = max(0.5, min(1.6, float(data.get("temperature", 1.0))))

    try:
        tokens = generate_tokens(model, vocab, note_to_int, notes, n_notes, temperature)
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mid"
        tokens_to_midi(tokens, config.OUTPUT_DIR / filename)
        return jsonify({
            "file": filename,
            "notes": n_notes,
            "temperature": temperature,
        })
    except Exception as exc:
        return jsonify({"error": f"Generation failed: {exc}"}), 500


@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(config.OUTPUT_DIR, filename)


@app.route("/download/<path:filename>")
def download_output(filename):
    return send_from_directory(config.OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5002))
    print(f"\n🎹 AI Music Studio — open http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
