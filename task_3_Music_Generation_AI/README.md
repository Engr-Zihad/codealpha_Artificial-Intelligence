# Task 3: Music Generation with AI

Trains an **LSTM** (a type of RNN) on a folder of MIDI files and generates
brand-new music sequences, which are converted back into a playable `.mid`
file. Uses `music21` for MIDI parsing/writing and TensorFlow/Keras for the
deep learning model.

## Features
- Parses all `.mid`/`.midi` files in a folder and extracts notes & chords
  as a sequence of tokens
- Builds fixed-length training sequences (default: 50 notes of context)
- Stacked LSTM model that learns to predict the next note in a sequence
- Generation step that seeds from real training data and repeatedly
  predicts the next note to build a new, original sequence
- Converts the generated sequence back into a MIDI file you can play in any
  media player or DAW

## Setup

```bash
cd Task_3_Music_Generation
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Getting MIDI training data
You'll need a folder of `.mid` files to train on (classical, jazz, game
music, etc.). A popular free source is the [Classical Piano MIDI
Page](http://www.piano-midi.de/) or any royalty-free MIDI dataset. Put the
files in a folder, e.g. `./midi_songs/`.

## Run

**Train a new model and generate music:**
```bash
python app.py --midi_dir ./midi_songs --epochs 50 --generate_notes 200 --output generated.mid
```
This will:
1. Parse all MIDI files in `./midi_songs/`
2. Train the LSTM (this can take a while — reduce `--epochs` for a quick test)
3. Save the trained model to `model.keras` and the note vocabulary to
   `notes_data.pkl`
4. Generate 200 new notes and save them to `generated.mid`

**Reuse a previously trained model (skip training):**
```bash
python app.py --load_model model.keras --generate_notes 300 --output generated2.mid
```

## Notes & Tips
- Training a good-sounding model typically needs **many epochs** (100+) and
  a reasonably sized MIDI dataset (dozens of songs). 50 epochs on a small
  dataset is a good smoke test but won't sound very musical yet.
- `--sequence_length` controls how much musical context the model considers
  before predicting the next note — try 50–100.
- The generated MIDI uses a fixed note spacing (`offset += 0.5`) for
  simplicity; for more rhythmic variety you could also predict note
  duration/offset alongside pitch (a common extension to this project).
- GPU training (via TensorFlow) is strongly recommended if you have access
  to one — LSTM training on CPU can be slow.
