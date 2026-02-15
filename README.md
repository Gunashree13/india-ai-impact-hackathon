# AI Voice Detection System - Hackathon Winning Strategy

## EXECUTION PLAN (Tonight to Tomorrow 9 AM)

### Timeline

| Time | Task | Who |
|------|------|-----|
| 3:00-4:30 PM | Download human data (Common Voice) + Setup env | YOU |
| 3:00-5:00 PM | Generate AI samples using edge-tts | TEAMMATE |
| 4:30-5:30 PM | Generate more AI samples (gTTS) | YOU |
| 5:30-6:30 PM | Create data manifest + Train model | YOU |
| 6:30-7:30 PM | Test and iterate on model accuracy | YOU |
| 7:30-8:30 PM | Deploy API (Railway/Render) | YOU |
| 8:30-10:00 PM | Full testing + fine-tuning | BOTH |
| 10:00 PM-12 AM | Edge cases, robustness testing | BOTH |

## Quick Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate AI data
python generate_data.py edge     # Edge TTS (best quality)
python generate_data.py gtts     # Google TTS
python generate_data.py manifest # Create manifest

# Train
python train_model.py train

# Run API
python api_server.py

# Test
python test_api.py
```

## Human Data Sources
1. Mozilla Common Voice: https://commonvoice.mozilla.org/en/datasets
2. OpenSLR: https://openslr.org/resources.php
3. IndicTTS: https://www.iitm.ac.in/donlab/tts/database.php

## Project Structure
```
voice-detection/
├── api_server.py           # FastAPI server
├── feature_extractor.py    # 200+ audio feature extraction
├── train_model.py          # Model training pipeline
├── generate_data.py        # AI sample generation
├── test_api.py             # API testing
├── deploy.py               # Deployment configs
├── requirements.txt
├── Dockerfile
├── models/                 # Trained models
└── data/                   # Training data
    ├── human/
    ├── ai/
    └── manifest.json
```
