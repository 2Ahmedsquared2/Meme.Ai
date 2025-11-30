#!/bin/bash

echo "🚀 Meme.AI Backend Setup & Start"
echo "=================================="
echo ""

cd "/Users/ahmedahmed/Downloads/coding stuff/Meme.Ai/Meme.Ai/meme-ai-backend"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Install requirements if not installed
echo "📥 Checking dependencies..."
pip install -q fastapi uvicorn python-multipart 2>/dev/null

echo ""
echo "🎭 Starting server..."
echo "=================================="
echo "📡 Backend: http://localhost:8000"
echo "🎭 Admin Panel: http://localhost:8000/admin"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

