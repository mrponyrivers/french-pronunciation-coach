from __future__ import annotations

import io
import os
import re
import unicodedata
from difflib import SequenceMatcher
from tempfile import NamedTemporaryFile

import streamlit as st
from openai import OpenAI


# =========================================
# Page config
# =========================================
st.set_page_config(
    page_title="French Pronunciation Coach",
    page_icon="🗣️",
    layout="wide",
)


# =========================================
# Styles
# =========================================
st.markdown(
    """
    <style>
    .hero-card {
        padding: 1.2rem 1.2rem 0.8rem 1.2rem;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 16px;
        background: rgba(250,250,250,0.03);
        margin-bottom: 1rem;
    }
    .score-good {
        padding: 0.6rem 0.8rem;
        border-radius: 12px;
        background: rgba(0, 180, 90, 0.12);
        border: 1px solid rgba(0, 180, 90, 0.28);
        font-weight: 600;
    }
    .score-close {
        padding: 0.6rem 0.8rem;
        border-radius: 12px;
        background: rgba(255, 180, 0, 0.12);
        border: 1px solid rgba(255, 180, 0, 0.28);
        font-weight: 600;
    }
    .score-try {
        padding: 0.6rem 0.8rem;
        border-radius: 12px;
        background: rgba(255, 70, 70, 0.12);
        border: 1px solid rgba(255, 70, 70, 0.28);
        font-weight: 600;
    }
    .history-card {
        padding: 0.9rem 1rem;
        border: 1px solid rgba(128,128,128,0.20);
        border-radius: 14px;
        margin-bottom: 0.8rem;
        background: rgba(250,250,250,0.02);
    }
    .small-muted {
        color: #888;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================
# Phrase packs
# =========================================
PHRASE_PACKS = {
    "Greetings": [
        {"text": "Bonjour, je m'appelle Pony.", "difficulty": "Easy"},
        {"text": "Comment ça va aujourd'hui ?", "difficulty": "Easy"},
        {"text": "Bonsoir, merci beaucoup.", "difficulty": "Easy"},
    ],
    "Travel": [
        {"text": "Où est la gare, s'il vous plaît ?", "difficulty": "Medium"},
        {"text": "Je voudrais un café, s'il vous plaît.", "difficulty": "Medium"},
        {"text": "Nous avons visité un musée à Paris.", "difficulty": "Medium"},
    ],
    "Homework Practice": [
        {"text": "Je vais au marché ce matin.", "difficulty": "Easy"},
        {"text": "J'aime apprendre le français tous les jours.", "difficulty": "Medium"},
        {"text": "Nous parlons français en classe.", "difficulty": "Easy"},
    ],
}


# =========================================
# Helpers
# =========================================
def get_api_key() -> str | None:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    return os.getenv("OPENAI_API_KEY")


def get_client() -> OpenAI:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Put it in .streamlit/secrets.toml")
    return OpenAI(api_key=api_key)


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    text = strip_accents(text.lower())
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9'\s-]", " ", text)
    text = text.replace("-", " ")
    text = text.replace("'", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split()


def compare_phrases(target: str, heard: str) -> dict:
    target_norm = normalize_text(target)
    heard_norm = normalize_text(heard)

    ratio = SequenceMatcher(None, target_norm, heard_norm).ratio()
    score = int(round(ratio * 100))

    target_tokens = tokenize(target)
    heard_tokens = tokenize(heard)

    matched_words = [word for word in target_tokens if word in heard_tokens]
    missing_words = [word for word in target_tokens if word not in heard_tokens]
    extra_words = [word for word in heard_tokens if word not in target_tokens]

    if score >= 92:
        verdict = "Great"
        message = "Very close. Your phrase matched the target well."
    elif score >= 75:
        verdict = "Close"
        message = "You were close. A few words may need another try."
    else:
        verdict = "Try again"
        message = "The app heard a noticeable difference from the target phrase."

    return {
        "score": score,
        "verdict": verdict,
        "message": message,
        "matched_words": matched_words,
        "missing_words": missing_words,
        "extra_words": extra_words,
    }


def guess_audio_filename(uploaded_file) -> str:
    original_name = getattr(uploaded_file, "name", "") or ""
    lower_name = original_name.lower()

    if lower_name.endswith(".mp3"):
        return "audio.mp3"
    if lower_name.endswith(".wav"):
        return "audio.wav"
    if lower_name.endswith(".m4a"):
        return "audio.m4a"
    if lower_name.endswith(".ogg"):
        return "audio.ogg"
    if lower_name.endswith(".webm"):
        return "audio.webm"

    return "audio.wav"


def transcribe_audio(client: OpenAI, uploaded_file) -> str:
    audio_bytes = uploaded_file.getvalue()
    file_obj = io.BytesIO(audio_bytes)
    file_obj.name = guess_audio_filename(uploaded_file)

    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=file_obj,
        language="fr",
        prompt="Transcribe the spoken French exactly as heard. The speaker is practicing pronunciation.",
    )

    return transcript.text.strip()


def synthesize_french_audio(client: OpenAI, text: str) -> bytes:
    temp_path = None

    try:
        with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            temp_path = tmp.name

        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="marin",
            input=text,
            response_format="mp3",
            instructions="Speak clearly in standard French, like a patient language teacher.",
        ) as response:
            response.stream_to_file(temp_path)

        with open(temp_path, "rb") as f:
            return f.read()

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def render_word_list(title: str, words: list[str], empty_text: str) -> None:
    st.markdown(f"**{title}**")
    if words:
        st.write(", ".join(words))
    else:
        st.write(empty_text)


def get_score_class(score: int) -> str:
    if score >= 92:
        return "score-good"
    if score >= 75:
        return "score-close"
    return "score-try"


def get_score_label(score: int) -> str:
    if score >= 92:
        return "Excellent pronunciation match"
    if score >= 75:
        return "Good attempt — close match"
    return "Needs another try"


def best_score(history: list[dict]) -> int:
    if not history:
        return 0
    return max(item["score"] for item in history)


# =========================================
# Session state
# =========================================
if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "target_audio_bytes" not in st.session_state:
    st.session_state.target_audio_bytes = None

if "target_audio_phrase" not in st.session_state:
    st.session_state.target_audio_phrase = ""

if "attempt_history" not in st.session_state:
    st.session_state.attempt_history = []

if "retry_count" not in st.session_state:
    st.session_state.retry_count = 0


# =========================================
# Sidebar
# =========================================
with st.sidebar:
    st.header("Practice Summary")
    st.metric("Attempts this session", st.session_state.retry_count)
    st.metric("Best score", best_score(st.session_state.attempt_history))
    st.write("Use short phrases first, then move to longer sentences.")
    if st.button("🧹 Reset session", use_container_width=True):
        st.session_state.analysis = None
        st.session_state.target_audio_bytes = None
        st.session_state.target_audio_phrase = ""
        st.session_state.attempt_history = []
        st.session_state.retry_count = 0
        st.rerun()


# =========================================
# Header
# =========================================
st.title("🗣️ French Pronunciation Coach")
st.markdown(
    """
    <div class="hero-card">
        <h4 style="margin-top:0;">Speak, compare, improve</h4>
        <p style="margin-bottom:0.3rem;">
            Practice a French phrase, record yourself, compare what the app heard,
            and replay the correct pronunciation.
        </p>
        <p class="small-muted" style="margin-bottom:0;">
            This version is designed as a pronunciation coach for homework phrases,
            travel phrases, and beginner speaking practice.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

col1, col2 = st.columns([1, 1])


# =========================================
# Left column: phrase selection
# =========================================
with col1:
    st.subheader("1. Choose your phrase")

    phrase_mode = st.radio(
        "Practice mode",
        ["Sample phrase", "Custom phrase"],
        horizontal=True,
    )

    selected_difficulty = ""

    if phrase_mode == "Sample phrase":
        category = st.selectbox("Phrase category", list(PHRASE_PACKS.keys()))
        selected_phrase_obj = st.selectbox(
            "Pick a French phrase",
            PHRASE_PACKS[category],
            format_func=lambda x: x["text"],
        )
        target_phrase = selected_phrase_obj["text"]
        selected_difficulty = selected_phrase_obj["difficulty"]
    else:
        target_phrase = st.text_input(
            "Type your French phrase",
            placeholder="Example: Je vais au marché ce matin.",
        )
        selected_difficulty = "Custom"

    st.markdown("**Target phrase**")
    st.write(target_phrase if target_phrase else "No phrase entered yet.")

    st.markdown("**Difficulty**")
    st.write(selected_difficulty)

    if st.button("🔊 Generate correct pronunciation", use_container_width=True):
        if not target_phrase.strip():
            st.error("Please enter a target phrase first.")
        else:
            try:
                client = get_client()
                with st.spinner("Generating target pronunciation..."):
                    audio_bytes = synthesize_french_audio(client, target_phrase.strip())
                st.session_state.target_audio_bytes = audio_bytes
                st.session_state.target_audio_phrase = target_phrase.strip()
                st.success("Target audio ready.")
            except Exception as e:
                st.error(f"Could not generate target audio: {e}")

    if (
        st.session_state.target_audio_bytes
        and st.session_state.target_audio_phrase == target_phrase.strip()
    ):
        st.audio(st.session_state.target_audio_bytes, format="audio/mp3")


# =========================================
# Right column: recording
# =========================================
with col2:
    st.subheader("2. Record yourself")

    mic_audio = st.audio_input("Use your microphone")

    st.markdown("**Or upload audio**")
    uploaded_audio = st.file_uploader(
        "Upload a French recording",
        type=["wav", "mp3", "m4a", "ogg", "webm"],
        accept_multiple_files=False,
    )

    selected_audio = None
    selected_source = None

    if mic_audio is not None:
        selected_audio = mic_audio
        selected_source = "Microphone"
        st.audio(mic_audio.getvalue(), format=getattr(mic_audio, "type", "audio/wav"))
    elif uploaded_audio is not None:
        selected_audio = uploaded_audio
        selected_source = "Upload"
        st.audio(
            uploaded_audio.getvalue(),
            format=getattr(uploaded_audio, "type", "audio/wav"),
        )

    if st.button("✅ Analyze pronunciation", type="primary", use_container_width=True):
        if not target_phrase.strip():
            st.error("Please choose or type a target phrase first.")
        elif selected_audio is None:
            st.error("Please record audio or upload a file first.")
        else:
            try:
                client = get_client()
                with st.spinner("Listening and comparing your pronunciation..."):
                    heard_text = transcribe_audio(client, selected_audio)
                    analysis = compare_phrases(target_phrase.strip(), heard_text)

                st.session_state.retry_count += 1

                saved_result = {
                    "target_phrase": target_phrase.strip(),
                    "heard_text": heard_text,
                    "source": selected_source,
                    "attempt_number": st.session_state.retry_count,
                    "difficulty": selected_difficulty,
                    **analysis,
                }

                st.session_state.analysis = saved_result
                st.session_state.attempt_history.insert(0, saved_result)

                if len(st.session_state.attempt_history) > 5:
                    st.session_state.attempt_history = st.session_state.attempt_history[:5]

            except Exception as e:
                st.error(f"Could not analyze audio: {e}")


# =========================================
# Results
# =========================================
st.divider()
st.subheader("3. Results")

analysis = st.session_state.analysis

if analysis is None:
    st.info("Record yourself and click Analyze pronunciation to see results.")
else:
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Score", f"{analysis['score']}/100")
    metric_col2.metric("Result", analysis["verdict"])
    metric_col3.metric("Source", analysis["source"])
    metric_col4.metric("Attempt", analysis["attempt_number"])

    st.markdown(
        f"""
        <div class="{get_score_class(analysis['score'])}">
            {get_score_label(analysis['score'])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Target phrase**")
    st.write(analysis["target_phrase"])

    st.markdown("**What the app heard**")
    st.write(analysis["heard_text"] if analysis["heard_text"] else "No speech detected.")

    if analysis["verdict"] == "Great":
        st.success(analysis["message"])
    elif analysis["verdict"] == "Close":
        st.warning(analysis["message"])
    else:
        st.error(analysis["message"])

    feedback_col1, feedback_col2 = st.columns(2)

    with feedback_col1:
        render_word_list(
            "Matched words",
            analysis["matched_words"],
            "No strong word matches were detected.",
        )

        render_word_list(
            "Missing target words",
            analysis["missing_words"],
            "None.",
        )

    with feedback_col2:
        render_word_list(
            "Extra / unclear words heard",
            analysis["extra_words"],
            "None.",
        )

        st.markdown("**Tip**")
        if analysis["missing_words"]:
            st.write(
                "Try saying the phrase more slowly and focus on these words: "
                + ", ".join(analysis["missing_words"])
            )
        else:
            st.write("Try repeating the phrase once more for an even cleaner result.")


# =========================================
# Recent attempts
# =========================================
st.divider()
st.subheader("4. Recent attempts")

if not st.session_state.attempt_history:
    st.caption("No attempts yet.")
else:
    for item in st.session_state.attempt_history:
        st.markdown(
            f"""
            <div class="history-card">
                <div style="display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap;">
                    <div><strong>Attempt {item['attempt_number']}</strong></div>
                    <div><strong>{item['score']}/100</strong></div>
                    <div>{item['verdict']}</div>
                    <div>{item['source']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(f"**Target:** {item['target_phrase']}")
        st.write(f"**Heard:** {item['heard_text']}")
        st.caption(f"Difficulty: {item.get('difficulty', 'Unknown')}")
        st.divider()