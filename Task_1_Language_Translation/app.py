"""
Task 1: Language Translation Tool
----------------------------------
A simple Streamlit web app that lets a user type text, choose a source and
target language, and get a translation back using the free Google Translate
backend (via the `deep-translator` library, no API key required).

Run with:
    streamlit run app.py
"""

import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS
import io

# ---------------------------------------------------------------------------
# Language list supported by deep-translator's GoogleTranslator
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_supported_languages():
    """Returns a dict of {language_name: language_code}."""
    return GoogleTranslator().get_supported_languages(as_dict=True)


def translate_text(text: str, source: str, target: str) -> str:
    """Translate `text` from `source` language code to `target` language code."""
    if not text.strip():
        return ""
    translator = GoogleTranslator(source=source, target=target)
    return translator.translate(text)


def text_to_speech_bytes(text: str, lang_code: str) -> bytes:
    """Convert text to speech and return the mp3 bytes (for playback/download)."""
    tts = gTTS(text=text, lang=lang_code)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def main():
    st.set_page_config(page_title="Language Translator", page_icon="🌐")
    st.title("🌐 Language Translation Tool")
    st.write("Type your text, pick source & target languages, and translate instantly.")

    languages = get_supported_languages()
    lang_names = sorted(languages.keys())

    col1, col2 = st.columns(2)
    with col1:
        source_lang_name = st.selectbox(
            "Source language", ["auto"] + lang_names, index=0
        )
    with col2:
        default_target_index = lang_names.index("english") if "english" in lang_names else 0
        target_lang_name = st.selectbox(
            "Target language", lang_names, index=default_target_index
        )

    input_text = st.text_area("Enter text to translate", height=150)

    if st.button("Translate", type="primary"):
        if not input_text.strip():
            st.warning("Please enter some text first.")
        else:
            source_code = "auto" if source_lang_name == "auto" else languages[source_lang_name]
            target_code = languages[target_lang_name]
            try:
                with st.spinner("Translating..."):
                    result = translate_text(input_text, source_code, target_code)
                st.session_state["translated_text"] = result
                st.session_state["target_code"] = target_code
            except Exception as e:
                st.error(f"Translation failed: {e}")

    if "translated_text" in st.session_state and st.session_state["translated_text"]:
        st.subheader("Translated Text")
        st.text_area(
            "Result", value=st.session_state["translated_text"], height=150, key="result_box"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            # A simple "copy" affordance using Streamlit's built-in code block copy icon
            st.caption("Click the copy icon below to copy the translation:")
            st.code(st.session_state["translated_text"], language=None)

        with col_b:
            if st.button("🔊 Play translation (Text-to-Speech)"):
                try:
                    audio_bytes = text_to_speech_bytes(
                        st.session_state["translated_text"], st.session_state["target_code"]
                    )
                    st.audio(audio_bytes, format="audio/mp3")
                except Exception as e:
                    st.error(f"Text-to-speech failed for this language: {e}")


if __name__ == "__main__":
    main()
