#!/bin/bash

# Quick Start Script for Admin Panel Testing

echo "🚀 Starting Meme.AI Backend + Admin Panel..."
echo ""

# Check if in correct directory
if [ ! -f "app/main.py" ]; then
    echo "❌ Error: Run this script from meme-ai-backend directory"
    exit 1
fi

# Start backend server
echo "📡 Starting FastAPI server on http://localhost:8000"
echo "🎭 Admin panel will be available at http://localhost:8000/admin"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

