"""
Task 3: Music Generation with AI
---------------------------------
Trains an LSTM (RNN) on a folder of MIDI files and generates new music
sequences, which are then converted back into a playable/saveable MIDI file.

Pipeline:
  1. Load MIDI files with `music21` and extract notes/chords as a flat
     sequence of string tokens (e.g. "C4", "E4.G4" for a chord).
  2. Encode tokens to integers, build fixed-length input/output training
     sequences.
  3. Train a stacked LSTM model (Keras/TensorFlow) to predict the next note
     given a sequence of previous notes.
  4. Use the trained model to generate a brand-new sequence of notes by
     repeatedly sampling from its predictions.
  5. Convert the generated sequence back into a `music21` stream and write
     it out as a `.mid` file.

Usage:
    # Train a model on a folder of .mid files and generate new music
    python app.py --midi_dir ./midi_songs --epochs 50 --generate_notes 200 --output generated.mid

    # Skip training and generate using a previously saved model
    python app.py --load_model model.keras --generate_notes 200 --output generated.mid
"""

import argparse
import glob
import pickle
import numpy as np

from music21 import converter, instrument, note, chord, stream


SEQUENCE_LENGTH = 50  # how many previous notes the model looks at


# ---------------------------------------------------------------------------
# 1. Data loading & preprocessing
# ---------------------------------------------------------------------------
def get_notes_from_midi(midi_dir: str):
    """Parse every .mid/.midi file in midi_dir and return a flat list of
    note/chord tokens (as strings), across all songs, in order."""
    notes = []
    midi_files = glob.glob(f"{midi_dir}/*.mid") + glob.glob(f"{midi_dir}/*.midi")
    if not midi_files:
        raise FileNotFoundError(f"No .mid/.midi files found in '{midi_dir}'.")

    for file in midi_files:
        print(f"Parsing {file} ...")
        midi = converter.parse(file)
        try:
            parts = instrument.partitionByInstrument(midi)
            elements = parts.parts[0].recurse() if parts else midi.flat.notes
        except Exception:
            elements = midi.flat.notes

        for element in elements:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))

    return notes


def prepare_sequences(notes, sequence_length=SEQUENCE_LENGTH):
    """Turn a flat list of note tokens into (X, y) training sequences plus
    the vocabulary mappings needed to encode/decode tokens."""
    pitch_names = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(pitch_names)}
    int_to_note = {i: n for n, i in note_to_int.items()}
    n_vocab = len(pitch_names)

    network_input = []
    network_output = []
    for i in range(0, len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    X = np.reshape(network_input, (n_patterns, sequence_length, 1))
    X = X / float(n_vocab)  # normalize

    from tensorflow.keras.utils import to_categorical
    y = to_categorical(network_output, num_classes=n_vocab)

    return X, y, note_to_int, int_to_note, n_vocab


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
def build_model(sequence_length, n_vocab):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

    model = Sequential([
        Input(shape=(sequence_length, 1)),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(n_vocab, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def train_model(model, X, y, epochs=50, batch_size=64):
    model.fit(X, y, epochs=epochs, batch_size=batch_size)
    return model


# ---------------------------------------------------------------------------
# 3. Generation
# ---------------------------------------------------------------------------
def generate_notes(model, network_input_raw, int_to_note, n_vocab, num_notes=200):
    """Seed with a random pattern from the training data, then repeatedly
    predict the next note and feed it back in."""
    start = np.random.randint(0, len(network_input_raw) - 1)
    pattern = list(network_input_raw[start])

    prediction_output = []
    for _ in range(num_notes):
        input_seq = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(input_seq, verbose=0)
        idx = np.argmax(prediction)
        prediction_output.append(int_to_note[idx])
        pattern.append(idx)
        pattern = pattern[1:]

    return prediction_output


def notes_to_midi(prediction_output, output_path="generated.mid"):
    """Convert a list of note/chord token strings back into a MIDI file."""
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if "." in pattern:  # a chord (stored as normalOrder ints)
            chord_notes = [note.Note(int(n)) for n in pattern.split(".")]
            for n in chord_notes:
                n.storedInstrument = instrument.Piano()
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:  # a single note
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5  # fixed spacing between notes; tweak for rhythm variety

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=output_path)
    print(f"Saved generated music to {output_path}")


# ---------------------------------------------------------------------------
# 4. CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train an LSTM on MIDI files and generate new music")
    parser.add_argument("--midi_dir", default=None, help="Folder of .mid/.midi files to train on")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size (default: 64)")
    parser.add_argument("--sequence_length", type=int, default=SEQUENCE_LENGTH,
                         help="Length of note sequence the model looks at (default: 50)")
    parser.add_argument("--generate_notes", type=int, default=200,
                         help="How many notes to generate (default: 200)")
    parser.add_argument("--output", default="generated.mid", help="Output MIDI file path")
    parser.add_argument("--save_model", default="model.keras", help="Where to save the trained model")
    parser.add_argument("--load_model", default=None, help="Path to a previously saved model to skip training")
    args = parser.parse_args()

    if args.load_model:
        from tensorflow.keras.models import load_model
        print(f"Loading model from {args.load_model} ...")
        model = load_model(args.load_model)
        with open("notes_data.pkl", "rb") as f:
            notes, note_to_int, int_to_note, n_vocab, network_input_raw = pickle.load(f)
    else:
        if not args.midi_dir:
            raise ValueError("Provide --midi_dir to train a new model, or --load_model to reuse one.")

        notes = get_notes_from_midi(args.midi_dir)
        X, y, note_to_int, int_to_note, n_vocab = prepare_sequences(notes, args.sequence_length)

        # Keep the raw (un-normalized) integer sequences around for seeding generation later
        network_input_raw = (X * n_vocab).astype(int).reshape(X.shape[0], X.shape[1])

        model = build_model(args.sequence_length, n_vocab)
        print(f"Training on {len(notes)} notes / {n_vocab} unique tokens ...")
        train_model(model, X, y, epochs=args.epochs, batch_size=args.batch_size)

        model.save(args.save_model)
        with open("notes_data.pkl", "wb") as f:
            pickle.dump((notes, note_to_int, int_to_note, n_vocab, network_input_raw), f)
        print(f"Model saved to {args.save_model}")

    print(f"Generating {args.generate_notes} notes ...")
    generated = generate_notes(model, network_input_raw, int_to_note, n_vocab, args.generate_notes)
    notes_to_midi(generated, args.output)


if __name__ == "__main__":
    main()
