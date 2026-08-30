# Task 1: Language Translation Tool

A Streamlit web app that translates text between languages using the free
Google Translate backend (via the `deep-translator` package — no API key
needed). Includes optional text-to-speech playback of the translated text.

## Features
- Text input box for the source text
- Dropdowns to pick source language (or auto-detect) and target language
- Translate button that calls the translation service
- Translated text shown clearly on screen
- Copy-friendly code block for the result
- Optional 🔊 "Play translation" button (text-to-speech via gTTS)

## Setup

```bash
cd Task_1_Language_Translation
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

This opens the app in your browser (usually at http://localhost:8501).

## Notes
- `deep-translator`'s `GoogleTranslator` uses Google Translate's free web
  endpoint, so no API key or billing account is required. If you'd rather use
  the official paid **Google Cloud Translation API** or **Microsoft
  Translator API**, swap out the `translate_text()` function in `app.py`:
  - Google Cloud: `google-cloud-translate` package, requires a service
    account key and billing enabled.
  - Microsoft: `requests` call to the Azure Translator endpoint with a
    subscription key.
- Internet access is required at runtime since translation calls an external
  service.
