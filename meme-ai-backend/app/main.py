from fastapi import FastAPI  # pyright: ignore[reportMissingImports]
from app.db import db 

@app.get("/")  # pyright: ignore[reportUndefinedVariable]
async def root():
    return {"status": "ok", "message": "Meme.AI backend is running"}

app = FastAPI(title="Meme.AI Backend - Minimal")


