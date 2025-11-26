from fastapi import FastAPI  # pyright: ignore[reportMissingImports]
from db import db, User, Meme, create_user, get_user, update_user, create_meme, get_meme, update_meme, delete_meme
from recommendation_engine import engine
from fastapi.middleware.cors import CORSMiddleware  # pyright: ignore[reportMissingImports]
from fastapi.responses import JSONResponse  # pyright: ignore[reportMissingImports]
from fastapi import HTTPException, status  # pyright: ignore[reportMissingImports]
from typing import Dict, List, Optional 
from datetime import datetime
from pydantic import BaseModel   # pyright: ignore[reportMissingImports]
import imagehash
from PIL import Image
import requests
from io import BytesIO

app = FastAPI(
    title="Meme.AI Backend",
    description="personalized meme recommendations engine",
    version="0.1.0"   
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # change later on to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ==================== REQUEST MODELS ====================
class CreateUserRequest(BaseModel):
    id: str
    email: str=None
    phone_number: str=None
class OnboardingRequest(BaseModel):
    scenario_choices: list = []
class CreateMemeRequest(BaseModel):
    id: str
    image_url: str
    user_tags: list = []
class InteractionRequest(BaseModel):
    user_id: str
    meme_id: str
class ReactionRequest(BaseModel):
    user_id: str
    meme_id: str
    reaction: str  # "up" or "down"
class FavoriteRequest(BaseModel):
    user_id: str
    meme_id: str


@app.get("/")  # pyright: ignore[reportUndefinedVariable]
async def root():
    return {"status": "ok", "message": "Meme.AI backend is running"}

@app.get("/health")
async def health_check():
    """
    Health Check endpoint -> checks database connection
    """
    try:
        _ = db.collection('_health_check').limit(1).get()
        return {"status": "ok", "db_connected": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail = "Service unavailable -  Database connection failed")

# ==================== USER ENDPOINTS ====================

@app.post("/users")
async def create_user_endpoint(request: CreateUserRequest):
    """ Creating new users"""
    try:
        #creates user objects
        new_user = User(
            id=request.id,
            email=request.email,
            phone_number=request.phone_number, 
        )
    
        #save to database 
        user_id = create_user(new_user)

        return {"status": "success", "user_id": user_id}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/users/{user_id}")
async def get_user_endpoint(user_id: str):
    """Get's userd data"""
    try:
        user = get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user.to_dict()
    except HTTPException:
        raise 
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.put("/users/{user_id}")
async def update_user_endpoint(user_id:str, request: CreateUserRequest):
    """Updates user data"""
    try:
        #check if user exists 
        user = get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        #update user object
        if request.email:
            user.email = request.email
        if request.phone_number:
            user.phone_number = request.phone_number
        
        #save to database
        update_user(user)
        return {"status": "success", "user_id": user_id}
    except HTTPException:
        raise 
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/users/{user_id}/onboarding")
async def submit_onboarding(user_id: str, request: OnboardingRequest):
    """Submits onboarding data(scenarios)"""
    try:
        #check if user exists
        user = get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        #validate submission 
        if not request.scenario_choices or len(request.scenario_choices) != 10:
            raise HTTPException(status_code=400, detail="Must complete all 10 scenarios")

        #mark onboarding as complete
        user.onboarding_completed = True
        
        #add choices to onboarding data
        for choice in request.scenario_choices:
            user.onboarding_data.append({
                "type": "scenario",
                "scenario_id": choice.get("scenario_id"),
                "meme_id": choice.get("meme_id"),
            })
        update_user(user)
        return {"status": "success", "user_id": user_id}
    except HTTPException:
        raise 
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

# ==================== MEME ENDPOINTS ====================
@app.get("/memes/{meme_id}")
async def get_meme_endpoint(meme_id: str):
    """Get meme by ID"""
    try:
        meme = get_meme(meme_id)
        if not meme:
            raise HTTPException(status_code=404, detail="Meme not found")
        return meme
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memes")
async def create_meme_endpoint(request: CreateMemeRequest):
    """Upload a new meme with duplicate detection"""
    try:        
        # Download and hash the image
        response = requests.get(request.image_url)
        img = Image.open(BytesIO(response.content))
        img_hash = str(imagehash.phash(img))
        
        # Check for exact duplicates
        existing = db.collection('memes').where('image_hash', '==', img_hash).limit(1).get()
        
        if len(list(existing)) > 0:
            raise HTTPException(
                status_code=409, 
                detail="This meme already exists in our database"
            )
        
        # Create new meme with hash
        new_meme = Meme(
            id=request.id,
            image_url=request.image_url,
            image_hash=img_hash,
            user_tags=request.user_tags,
            status="pending"  # Needs approval first!
        )
        
        meme_id = create_meme(new_meme)
        return {"status": "success", "meme_id": meme_id}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/memes/{meme_id}")
async def update_meme_endpoint(meme_id: str):
    """Update meme stats (used internally by recommendation engine)"""
    try:        
        meme_dict = get_meme(meme_id)
        if not meme_dict:
            raise HTTPException(status_code=404, detail="Meme not found")
        
        # Convert dict back to Meme object
        meme = Meme(**meme_dict)
        update_meme(meme)
        
        return {"status": "success", "meme_id": meme_id}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memes/{meme_id: str}")
async def delete_meme_endpoint(meme_id: str):
    """Delete a meme"""
    try:
        meme = get_meme(meme_id)
        if not meme: 
            raise HTTPException(status_code=404, details="Meme nto found")
        delete_meme(meme_id)
        return {"status": "success", "meme_id": meme_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/memes")
async def get_all_memes(limit: int = 30, status: str = "approved", tags: Optional[List[str]] = None):
    """Get top 30 memes based off popularity and filters"""
    try: 
        #collect the memes
        query = db.collection('memes').where('status', '==', status)
        #add tag filters
        if tags and len(tags) > 0:
            query = query.where('user_tags', 'array_contains_any', tags)

        #caps results at 30
        docs = query.limit(limit).get()

        memes = []
        for doc in docs:
            memes.append(doc.to_dict())
        return{"memes": memes, "count": len(memes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== tracking endpoints ====================

@app.post("/interactions/view")
async def track_view(request: InteractionRequest):
    """Track when a user views a meme"""
    try:
        # Get user and meme
        user = get_user(request.user_id)
        meme_dict = get_meme(request.meme_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if not meme_dict:
            raise HTTPException(status_code=404, detail="Meme not found")
        
        # Update user stats
        user.total_memes_viewed += 1
        update_user(user)
        
        # Update meme stats
        meme = Meme(**meme_dict)
        meme.total_views += 1
        update_meme(meme)
        
        return {"status": "success", "message": "View tracked"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interactions/reaction")
async def track_reaction(request: ReactionRequest):
    """Track thumbs up/down on a meme"""
    try:
        # Get user and meme
        user = get_user(request.user_id)
        meme_dict = get_meme(request.meme_id)
        
        # Check if they exist
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if not meme_dict:
            raise HTTPException(status_code=404, detail="Meme not found")
        
        # Validate reaction is "up" or "down"
        if request.reaction not in ["up", "down"]:
            raise HTTPException(status_code=400, detail="Reaction must be 'up' or 'down'")
        
        # Update user stats
        if request.reaction == "up":
            user.total_memes_thumbed_up += 1
        else:
            user.total_memes_thumbed_down += 1
        update_user(user)
        
        # Update meme stats
        meme = Meme(**meme_dict)
        if request.reaction == "up":
            meme.total_thumbs_up += 1
        else:
            meme.total_thumbs_down += 1
        update_meme(meme)
        
        return {"status": "success", "message": f"Thumbs {request.reaction} tracked"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interactions/favorite")
async def add_favorite(request: FavoriteRequest):
    """Add a meme to user's favorites"""
    try:
        # Get user and meme
        user = get_user(request.user_id)
        meme_dict = get_meme(request.meme_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if not meme_dict:
            raise HTTPException(status_code=404, detail="Meme not found")
        
        # Check if already favorited
        if request.meme_id not in user.favorited_meme_ids:
            # Add to favorites
            user.favorited_meme_ids.append(request.meme_id)
            user.total_memes_favorited += 1
            update_user(user)
            
            # Update meme stats
            meme = Meme(**meme_dict)
            meme.total_favorites += 1
            update_meme(meme)
            
            return {"status": "success", "message": "Meme added to favorites"}
        else:
            return {"status": "success", "message": "Meme already in favorites"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

@app.delete("/interactions/favorite/{user_id}/{meme_id}")
async def remove_favorite(user_id: str, meme_id: str):
    """Remove a meme from user's favorites"""
    try:
        # Get user and meme
        user = get_user(user_id)
        meme_dict = get_meme(meme_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if not meme_dict:
            raise HTTPException(status_code=404, detail="Meme not found")
        
        # Check if it's in favorites
        if meme_id in user.favorited_meme_ids:
            # Remove from favorites
            user.favorited_meme_ids.remove(meme_id)
            user.total_memes_favorited -= 1
            update_user(user)
            
            # Update meme stats
            meme = Meme(**meme_dict)
            meme.total_favorites -= 1
            update_meme(meme)
            
            return {"status": "success", "message": "Meme removed from favorites"}
        else:
            raise HTTPException(status_code=404, detail="Meme not in favorites")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/interactions/send")
async def track_send(request: FavoriteRequest):
    """Track when a user sends a meme to someone"""
    try:
        # Get user and meme
        user = get_user(request.user_id)
        meme_dict = get_meme(request.meme_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if not meme_dict:
            raise HTTPException(status_code=404, detail="Meme not found")
        
        # Update user stats
        user.total_memes_sent += 1
        update_user(user)
        
        # Update meme stats
        meme = Meme(**meme_dict)
        meme.total_shares += 1
        update_meme(meme)
        
        return {"status": "success", "message": "Meme send tracked"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memes/suggest")
async def suggest_memes(user_id: str, context: str = "", limit: int = 3):
    """Get AI-powered meme suggestions based on context"""
    try:
        user = get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Use recommendation engine
        memes = engine.get_personalized_memes(
            user_id=user_id,
            limit=limit,
            context_text=context if context else None
        )
        
        return {"memes": memes, "count": len(memes)}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== extra endpoints ====================
@app.post("/memes/{meme_id}/approve")
async def approve_meme(meme_id: str):
    """Admin endpoint to approve a meme"""
    try:
        # Get the meme
        meme_dict = get_meme(meme_id)
        
        if not meme_dict:
            raise HTTPException(status_code=404, detail="Meme not found")
        
        # Update status to approved
        meme = Meme(**meme_dict)
        meme.status = "approved"
        update_meme(meme)
        
        return {"status": "success", "message": "Meme approved"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memes/{meme_id}/reject")
async def reject_meme(meme_id: str):
    """Admin endpoint to reject a meme"""
    try:
        # Get the meme
        meme_dict = get_meme(meme_id)
        
        if not meme_dict:
            raise HTTPException(status_code=404, detail="Meme not found")
        
        # Update status to rejected
        meme = Meme(**meme_dict)
        meme.status = "rejected"
        update_meme(meme)
        
        return {"status": "success", "message": "Meme rejected"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))








