# Project Bundle: 4 Mini AI/ML Applications

```
├── Task_1_Language_Translation/
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
├── Task_2_FAQ_Chatbot/
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
├── Task_3_Music_Generation/
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
├── Task_4_Object_Detection/
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
└── README.md   (this file)
```

Each task lives in its own clearly named, numbered folder so it's easy to
tell them apart. Every folder is self-contained: `app.py` (the program),
`requirements.txt` (dependencies to install), and `README.md` (setup + run
instructions specific to that task).

## Task 1 — Language Translation Tool
Streamlit UI where a user enters text, picks a source and target language,
and gets an instant translation (free Google Translate backend via
`deep-translator`, no API key needed). Includes an optional text-to-speech
playback button.
→ See `Task_1_Language_Translation/README.md`

## Task 2 — Chatbot for FAQs
A retrieval-based FAQ chatbot: NLTK preprocesses text (tokenize, remove
stopwords, lemmatize), TF-IDF + cosine similarity finds the closest matching
FAQ, and the answer is shown in a Streamlit chat UI (or a CLI mode).
→ See `Task_2_FAQ_Chatbot/README.md`

## Task 3 — Music Generation with AI
Trains an LSTM (RNN) on a folder of MIDI files, using `music21` to parse
notes/chords into training sequences, then generates a brand-new sequence
of notes and converts it back into a playable `.mid` file.
→ See `Task_3_Music_Generation/README.md`

## Task 4 — Object Detection and Tracking
Real-time detection and tracking on a webcam or video file using OpenCV +
a pre-trained YOLOv8 model, with built-in ByteTrack tracking so each object
keeps a consistent ID across frames.
→ See `Task_4_Object_Detection/README.md`

## Quick Start (any task)
```bash
cd Task_X_.../
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Tasks 1 & 2 (Streamlit apps)
streamlit run app.py

# Task 2 command-line mode
python app.py --cli

# Task 3 (train + generate music)
python app.py --midi_dir ./midi_songs --epochs 50 --output generated.mid

# Task 4 (webcam detection + tracking)
python app.py
```
