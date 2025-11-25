from fastapi import FastAPI
from app.db import db 

@app.get("/")
async def root():
    return {"status": "ok", "message": "Meme.AI backend is running"}

app = FastAPI(title="Meme.AI Backend - Minimal")


