from fastapi import FastAPI

app = FastAPI(title="Meme.AI Backend - Minimal")

@app.get("/")
async def root():
    return {"status": "ok", "message": "Meme.AI backend is running"}
