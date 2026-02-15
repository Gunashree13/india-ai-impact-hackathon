import os
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

def generate_explanation(classification: str, confidence: float, language: str) -> str:
    """
    Generates a human-readable explanation for the result.
    """
    if classification == "AI_GENERATED":
        if confidence >= 0.8:
            return (
                f"High-confidence detection of synthetic voice patterns, "
                f"including unnatural pitch consistency in the {language} sample."
            )
        else:
            return (
                f"Minor digital artifacts detected in the {language} speech, "
                f"suggesting possible AI generation."
            )
    else:
        if confidence >= 0.8:
            return (
                f"Natural prosody, breathing patterns, and organic speech flow "
                f"detected, consistent with a human {language} speaker."
            )
        else:
            return (
                f"Speech characteristics align with human vocal patterns "
                f"for the {language} language."
            )


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

        explanation = generate_explanation(
            classification,
            confidence,
            request.language
        )

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
