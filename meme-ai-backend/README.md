# Meme.AI Backend

A FastAPI-based backend service for the Meme.AI iMessage app. This service manages memes, handles Firebase Auth/Storage/Firestore, performs AI tagging/embeddings, and serves personalized meme recommendations.

## Tech Stack

- **Framework:** FastAPI + Uvicorn
- **Database:** Firestore (Firebase)
- **Storage:** Firebase Storage
- **AI Models:** CLIP (image tagging), LLaMA 3.1 8B (context understanding), FLAN-T5 (caption generation)
- **Language:** Python 3.13

## Setup Instructions

### 1. Clone and navigate to the project

```bash
cd meme-ai-backend
```

### 2. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\Activate.ps1  # Windows PowerShell
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the development server

```bash
uvicorn app.main:app --reload --port 8000
```

## Testing the API

### Root endpoint
```bash
curl http://127.0.0.1:8000/
```

Expected response:
```json
{"status":"ok","message":"Meme.AI backend is running"}
```

### Interactive API Documentation
Visit `http://127.0.0.1:8000/docs` in your browser for the automatic Swagger UI documentation.

## Project Structure

```
meme-ai-backend/
├── app/
│   └── main.py          # FastAPI application entry point
├── .venv/               # Python virtual environment (not in git)
├── .gitignore           # Git ignore rules
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## What We've Built So Far

✅ **Step 1 Complete:** Basic FastAPI server setup
- Created Python virtual environment
- Installed FastAPI, Uvicorn, Firebase Admin SDK, python-dotenv
- Created minimal API with health check endpoint
- Established version control with git
- Set up proper .gitignore for security

## Next Steps

- Connect Firebase Admin SDK
- Set up Firestore access with service account
- Create meme data model
- Add endpoints for creating and viewing memes

## Development Notes

- The `--reload` flag enables auto-reload when you save files
- Port 8000 is the default; change with `--port` if needed
- Always activate the virtual environment before running commands
- Never commit Firebase credentials to git!
