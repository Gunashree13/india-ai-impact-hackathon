import os
from pyexpat import features
import uuid
import base64
from fastapi import FastAPI, Header, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import librosa
from pydub import AudioSegment

# ML inference bridge
# from ml.inference.predict import predict

from predict import predict

# -------------------- CONFIGURATION --------------------

app = FastAPI(title="AI Voice Detection API")

# API Key (use ENV in production)
API_KEY = os.getenv("API_KEY", "hackathon-secret")

SUPPORTED_LANGUAGES = [
    "Tamil",
    "English",
    "Hindi",
    "Malayalam",
    "Telugu"
]


# -------------------- REQUEST MODEL --------------------

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


# -------------------- HELPER FUNCTIONS --------------------

# -------------------- PASTE INTO main.py --------------------
# Replace the old generate_explanation function AND update the predict call

def generate_explanation(classification: str, confidence: float, language: str, features: dict = None) -> str:
    """
    Generate explanation based on actual acoustic features detected.
    """
    if features is None:
        features = {}

    reasons = []

    if classification == "AI_GENERATED":
        # Check actual feature values
        jitter = features.get('jitter', -1)
        shimmer = features.get('shimmer', -1)
        pitch_cv = features.get('pitch_cv', -1)
        silence_ratio = features.get('silence_ratio', -1)
        rms_cv = features.get('rms_cv', -1)
        hf_smooth = features.get('hf_smooth', -1)
        hnr = features.get('hnr', -1)

        if 0 <= jitter < 0.02:
            reasons.append(f"unusually low pitch micro-variations (jitter={jitter:.4f}) suggesting synthetic vocal generation")
        if 0 <= shimmer < 0.2:
            reasons.append("abnormally consistent amplitude patterns not typical of natural speech")
        if 0 <= pitch_cv < 0.15:
            reasons.append("limited pitch variation indicating machine-generated monotone characteristics")
        if 0 <= rms_cv < 0.4:
            reasons.append("uniform energy distribution lacking natural human speech dynamics")
        if silence_ratio >= 0 and silence_ratio < 0.05:
            reasons.append("absence of natural breathing pauses between speech segments")
        if hf_smooth >= 0 and hf_smooth < 0.001:
            reasons.append("smooth high-frequency spectrum consistent with neural vocoder artifacts")
        if hnr > 15:
            reasons.append("abnormally high harmonic-to-noise ratio indicating synthesized audio clarity")

        if not reasons:
            reasons.append("combination of spectral and temporal patterns consistent with AI-generated speech")

        if confidence >= 0.85:
            prefix = f"High-confidence AI detection in the {language} sample"
        elif confidence >= 0.7:
            prefix = f"Moderate indicators of synthetic generation in the {language} sample"
        else:
            prefix = f"Subtle synthetic patterns detected in the {language} sample"

        # Pick top 2 reasons max
        selected = reasons[:2]
        return f"{prefix}: {'; '.join(selected)}."

    else:
        pitch_cv = features.get('pitch_cv', -1)
        jitter = features.get('jitter', -1)
        rms_cv = features.get('rms_cv', -1)
        silence_ratio = features.get('silence_ratio', -1)

        if jitter > 0.02:
            reasons.append("natural pitch micro-variations (jitter) consistent with human vocal cord vibration")
        if pitch_cv > 0.15:
            reasons.append("healthy pitch variation reflecting natural prosody and emotional expression")
        if rms_cv > 0.5:
            reasons.append("dynamic energy patterns showing natural speech rhythm and emphasis")
        if silence_ratio > 0.05:
            reasons.append("natural breathing pauses and organic speech timing detected")

        if not reasons:
            reasons.append("overall acoustic signature consistent with natural human speech production")

        if confidence >= 0.85:
            prefix = f"Strong indicators of natural human speech in the {language} sample"
        elif confidence >= 0.7:
            prefix = f"Speech patterns in the {language} sample align with human vocal characteristics"
        else:
            prefix = f"The {language} sample shows characteristics generally associated with human speech"

        selected = reasons[:2]
        return f"{prefix}: {'; '.join(selected)}."

# -------------------- HEALTH CHECK --------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "AI Voice Detection API is running"
    }


# -------------------- MAIN API --------------------

@app.post("/api/voice-detection")
def detect_voice(
    request: VoiceRequest = Body(...),
    x_api_key: str = Header(...)
):
    # 1️⃣ API KEY VALIDATION
    if x_api_key != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Invalid API key"}
        )

    # 2️⃣ LANGUAGE VALIDATION
    if request.language not in SUPPORTED_LANGUAGES:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Unsupported language. Allowed values: {SUPPORTED_LANGUAGES}"}
        )

    # 3️⃣ AUDIO FORMAT VALIDATION
    if request.audioFormat.lower() != "mp3":
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Only mp3 audio format is supported"}
        )

    # Temporary file names
    temp_mp3 = f"temp_{uuid.uuid4()}.mp3"
    original_temp_mp3 = temp_mp3

    try:
        # 4️⃣ BASE64 DECODE
        try:
            audio_bytes = base64.b64decode(
                request.audioBase64,
                validate=True
            )
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Invalid Base64 audio string"}
            )

        # Reject empty or fake audio
        if len(audio_bytes) < 1000:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Audio data is too small or empty"}
            )

        # 5️⃣ SAVE MP3 FILE
        with open(temp_mp3, "wb") as f:
            f.write(audio_bytes)

        # 5.5️⃣ CHECK AND TRIM AUDIO DURATION (max 60 seconds)
        y, sr = librosa.load(temp_mp3, sr=None)
        duration = len(y) / sr
        if duration > 30:
            # Trim to first 60 seconds
            audio = AudioSegment.from_file(temp_mp3, format="mp3")
            trimmed_audio = audio[:30000] # 60 seconds in milliseconds
            trimmed_mp3 = temp_mp3.replace(".mp3", "_trimmed.mp3")
            trimmed_audio.export(trimmed_mp3, format="mp3")
            temp_mp3 = trimmed_mp3  # Use trimmed file

        # 6️⃣ ML INFERENCE (Member-1 implementation)
        result = predict(temp_mp3)

        classification = result.get("classification")
        confidence = result.get("confidenceScore")

        if classification not in ["AI_GENERATED", "HUMAN"]:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Invalid classification returned by ML model"}
            )

        features = result.get("features", {})
        explanation = generate_explanation(classification, confidence, request.language, features)

        # 8️⃣ SUCCESS RESPONSE (STRICT FORMAT)
        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": explanation
        }

    except Exception as e:
        # Catch-all for unexpected failures
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Processing error: {str(e)}"}
        )

    finally:
        # 9️⃣ CLEANUP TEMP FILES
        for path in [original_temp_mp3, temp_mp3]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
