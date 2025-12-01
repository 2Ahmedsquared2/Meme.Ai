
from fastapi import FastAPI, HTTPException, UploadFile, File  # pyright: ignore[reportMissingImports]
from ml_model import (
    auto_tag_meme, 
    extract_context_tags,
    VISUAL_TYPE_OPTIONS,
    VISUAL_PEOPLE_OPTIONS,
    VISUAL_ACTION_OPTIONS,
    VISUAL_FORMAT_OPTIONS,
    EMOTION_OPTIONS,
    VIBE_OPTIONS,
    SITUATION_OPTIONS,
    SOCIAL_OPTIONS
)
from db import db, User, Meme, create_user, get_user, update_user, create_meme, get_meme, update_meme, delete_meme, upload_image_to_storage
from fastapi.middleware.cors import CORSMiddleware  # pyright: ignore[reportMissingImports]
from fastapi.responses import JSONResponse, FileResponse  # pyright: ignore[reportMissingImports]
from fastapi import HTTPException, status  # pyright: ignore[reportMissingImports]
from fastapi.staticfiles import StaticFiles  # pyright: ignore[reportMissingImports]
from typing import Dict, List, Optional 
from recommendation_engine import get_recommendations
from datetime import datetime
from pydantic import BaseModel   # pyright: ignore[reportMissingImports]
import imagehash
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path

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
class RecommendationRequest(BaseModel):
    user_id: str
    context: str
    session_id: Optional[str] = None
    required_tags: Optional[List[str]] = None
    batch_size: int = 3
class RecommendationFeedbackRequest(BaseModel):
    user_id: str
    meme_id: str
    context: str  # ← CRITICAL: The context they were in when they picked this meme
    session_id: Optional[str] = None  # Optional: for tracking session flow

class UpdateMemeTagsRequest(BaseModel):
    visual_tags: Optional[List[str]] = None
    contextual_tags: Optional[List[str]] = None
    user_tags: Optional[List[str]] = None
    all_tags: Optional[List[str]] = None

@app.get("/")  # pyright: ignore[reportUndefinedVariable]
async def root():
    return {"status": "ok", "message": "Meme.AI backend is running"}

# Mount admin panel
admin_path = Path(__file__).parent.parent / "admin"
if admin_path.exists():
    app.mount("/admin", StaticFiles(directory=str(admin_path), html=True), name="admin")

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

# ==================== HELPER FUNCTIONS ====================
def process_meme_upload(image_bytes: bytes, user_tags: List[str], meme_id: str, filename: str) -> Dict:
    """
    Check for duplicates BEFORE uploading to storage
    Run ML auto-tagging and save results
    """
    #Compute hash from bytes(image)
    img = Image.open(BytesIO(image_bytes))
    img_hash = str(imagehash.phash(img))
    
    # Check if duplicate exists
    existing = db.collection('memes').where('image_hash', '==', img_hash).limit(1).get()
    if len(list(existing)) > 0:
        raise HTTPException(
            status_code=409, 
            detail="This meme already exists in our database"
        )
    
    #Upload to Firebase Storage (only if it's unique)
    image_url = upload_image_to_storage(image_bytes, filename)
    
    #run auto tagging 
    print(f"🤖 Running ML auto-tagging for {meme_id}...")
    ml_tags = auto_tag_meme(image_url)
    print(f"✅ Generated {len(ml_tags['all_tags'])} tags!")
    
    # Create final meme object
    # Note: user_tags starts empty, will be added by admin in the admin panel
    new_meme = Meme(
        id=meme_id,
        image_url=image_url,
        image_hash=img_hash,
        user_tags=[],  # ← Empty! Admin will add custom tags
        visual_tags=ml_tags["visual_tags"],           # ← ML visual tags
        contextual_tags=ml_tags["contextual_tags"],   # ← ML contextual tags
        blip2_caption=ml_tags["blip2_caption"],       # ← BLIP caption
        all_tags=ml_tags["all_tags"],                 # ← Only ML tags initially
        status="pending"
    )
    
    create_meme(new_meme)
    
    return {
        "status": "success", 
        "meme_id": meme_id, 
        "image_url": image_url,
        "ml_tags_generated": len(ml_tags["all_tags"]),
        "note": "Meme uploaded with ML tags only. Add custom tags in admin panel."
    }
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
        # Check if user exists 
        user = get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Prepare update dict
        updates = {}
        if request.email:
            updates['email'] = request.email
        if request.phone_number:
            updates['phone_number'] = request.phone_number
        
        # Save to database
        update_user(user_id, updates)
        return {"status": "success", "user_id": user_id}
    except HTTPException:
        raise 
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/users/{user_id}/onboarding")
async def submit_onboarding(user_id: str, request: OnboardingRequest):
    """Submits onboarding data(scenarios)"""
    try:
        # Check if user exists
        user = get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Validate submission 
        if not request.scenario_choices or len(request.scenario_choices) != 10:
            raise HTTPException(status_code=400, detail="Must complete all 10 scenarios")

        # Build onboarding data
        user_dict = user if isinstance(user, dict) else user.to_dict()
        onboarding_data = user_dict.get('onboarding_data', [])
        
        for choice in request.scenario_choices:
            onboarding_data.append({
                "type": "scenario",
                "scenario_id": choice.get("scenario_id"),
                "meme_id": choice.get("meme_id"),
            })
        
        # Update user
        update_user(user_id, {
            'onboarding_completed': True,
            'onboarding_data': onboarding_data
        })
        
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

@app.post("/memes/upload")
async def upload_meme_file(file: UploadFile = File(...)):
    """Upload meme image file - user_tags will be added in admin panel"""
    try:
        # Read file
        file_bytes = await file.read()
        
        # Check file size
        max_size = 10 * 1024 * 1024  # 10 MB
        if len(file_bytes) > max_size:
            raise HTTPException(
                status_code=400, 
                detail="Image is too large. Maximum size is 10 MB."
            )
        
        # make sure it's an image
        try:
            img = Image.open(BytesIO(file_bytes))
            img.verify()
        except Exception:
            raise HTTPException(
                status_code=400, 
                detail="Uploaded file is not a valid image."
            )
        
        # get filename
        timestamp = datetime.utcnow().timestamp()
        filename = f"meme_{timestamp}_{file.filename}"
        meme_id = f"meme_{int(timestamp)}"
        
        # user_tags starts empty - will be added in admin panel
        tag_list = []
        
        # Process
        return process_meme_upload(file_bytes, tag_list, meme_id, filename)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.post("/memes")
async def create_meme_endpoint(request: CreateMemeRequest):
    """Upload a new meme w url"""
    try:
        # Make sure it's actually a URL
        if not request.image_url.startswith(('http://', 'https://')):
            raise HTTPException(
                status_code=400, 
                detail="Invalid URL format. Must start with http:// or https://"
            )
        
        # Download the image (with timeout so we don't wait forever)
        try:
            response = requests.get(
                request.image_url, 
                timeout=10,
                headers={'User-Agent': 'Meme.AI/1.0'}  # Some sites block bots
            )
        except requests.exceptions.Timeout:
            raise HTTPException(
                status_code=400, 
                detail="Image URL timed out. Please try a different image."
            )
        except requests.exceptions.ConnectionError:
            raise HTTPException(
                status_code=400, 
                detail="Could not connect to image URL. Please check the link."
            )
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Error downloading image: {str(e)}"
            )
        
        # Check if the URL actually worked
        if response.status_code != 200:
            raise HTTPException(
                status_code=400, 
                detail=f"Image URL returned status code {response.status_code}. Link may be broken."
            )
        
        img_bytes = response.content
        
        # Don't accept massive files
        max_size = 10 * 1024 * 1024  # 10 MB
        if len(img_bytes) > max_size:
            raise HTTPException(
                status_code=400, 
                detail="Image is too large. Maximum size is 10 MB."
            )
        
        # Make sure it's actually an image and not like a text file or something
        try:
            img = Image.open(BytesIO(img_bytes))
            img.verify()
        except Exception:
            raise HTTPException(
                status_code=400, 
                detail="URL does not point to a valid image file. Supported formats: JPEG, PNG, GIF, WebP"
            )

        # Generate unique filename
        timestamp = datetime.utcnow().timestamp()
        filename = f"meme_{timestamp}_{request.image_url.split('/')[-1]}"
        meme_id = f"meme_{int(timestamp)}"

        # Check for duplicates and process
        return process_meme_upload(img_bytes, request.user_tags, meme_id, filename)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.patch("/memes/{meme_id}/tags")
async def update_meme_tags(meme_id: str, request: UpdateMemeTagsRequest):
    """Update meme tags (used by admin panel)"""
    try:        
        meme_dict = get_meme(meme_id)
        if not meme_dict:
            raise HTTPException(status_code=404, detail="Meme not found")
        
        # Convert to Meme object
        meme = Meme(**meme_dict)
        
        # Update tags
        if request.visual_tags is not None:
            meme.visual_tags = request.visual_tags
        if request.contextual_tags is not None:
            meme.contextual_tags = request.contextual_tags
        if request.user_tags is not None:
            meme.user_tags = request.user_tags
        
        # Update all_tags to be combination of all tag types
        meme.all_tags = list(set(meme.visual_tags + meme.contextual_tags + meme.user_tags))
        
        update_meme(meme)
        
        return {
            "status": "success", 
            "meme_id": meme_id,
            "visual_tags": meme.visual_tags,
            "contextual_tags": meme.contextual_tags,
            "user_tags": meme.user_tags,
            "all_tags": meme.all_tags
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memes/{meme_id}")
async def update_meme_endpoint(meme_id: str, request: UpdateMemeTagsRequest = None):
    """Update meme tags (legacy endpoint - use PATCH /memes/{meme_id}/tags instead)"""
    try:        
        meme_dict = get_meme(meme_id)
        if not meme_dict:
            raise HTTPException(status_code=404, detail="Meme not found")
        
        # Convert dict back to Meme object
        meme = Meme(**meme_dict)
        
        # Update tags if provided
        if request:
            if request.visual_tags is not None:
                meme.visual_tags = request.visual_tags
            if request.contextual_tags is not None:
                meme.contextual_tags = request.contextual_tags
            if request.user_tags is not None:
                meme.user_tags = request.user_tags
            if request.all_tags is not None:
                meme.all_tags = request.all_tags
        
        update_meme(meme)
        
        return {"status": "success", "meme_id": meme_id}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memes/{meme_id}")
async def delete_meme_endpoint(meme_id: str):
    """Delete a meme"""
    try:
        meme = get_meme(meme_id)
        if not meme: 
            raise HTTPException(status_code=404, detail="Meme not found")
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
        
        # Get current values
        user_dict = user if isinstance(user, dict) else user.to_dict()
        meme = Meme(**meme_dict)
        
        # Update user stats
        update_user(request.user_id, {
            'total_memes_viewed': user_dict.get('total_memes_viewed', 0) + 1
        })
        
        # Update meme stats
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
        
        # Get current values
        user_dict = user if isinstance(user, dict) else user.to_dict()
        meme = Meme(**meme_dict)
        
        # Update user stats
        if request.reaction == "up":
            update_user(request.user_id, {
                'total_memes_thumbed_up': user_dict.get('total_memes_thumbed_up', 0) + 1
            })
            meme.total_thumbs_up += 1
        else:
            update_user(request.user_id, {
                'total_memes_thumbed_down': user_dict.get('total_memes_thumbed_down', 0) + 1
            })
            meme.total_thumbs_down += 1
        
        # Update meme stats
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
        user_dict = user if isinstance(user, dict) else user.to_dict()
        favorited = user_dict.get('favorited_meme_ids', [])
        
        if request.meme_id not in favorited:
            # Add to favorites
            favorited.append(request.meme_id)
            update_user(request.user_id, {
                'favorited_meme_ids': favorited,
                'total_memes_favorited': user_dict.get('total_memes_favorited', 0) + 1
            })
            
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
        
        # Get current values
        user_dict = user if isinstance(user, dict) else user.to_dict()
        favorited = user_dict.get('favorited_meme_ids', [])
        
        # Check if it's in favorites
        if meme_id in favorited:
            # Remove from favorites
            favorited.remove(meme_id)
            update_user(user_id, {
                'favorited_meme_ids': favorited,
                'total_memes_favorited': max(0, user_dict.get('total_memes_favorited', 0) - 1)
            })
            
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
        user_dict = user if isinstance(user, dict) else user.to_dict()
        favorited = user_dict.get('favorited_meme_ids', [])
        
        if meme_id in favorited:
            # Remove from favorites
            favorited.remove(meme_id)
            update_user(user_id, {
                'favorited_meme_ids': favorited,
                'total_memes_favorited': max(0, user_dict.get('total_memes_favorited', 0) - 1)
            })
            
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
        
        # Get current values
        user_dict = user if isinstance(user, dict) else user.to_dict()
        meme = Meme(**meme_dict)
        
        # Update user stats
        update_user(request.user_id, {
            'total_memes_sent': user_dict.get('total_memes_sent', 0) + 1
        })
        
        # Update meme stats
        meme.total_shares += 1
        update_meme(meme)
        
        return {"status": "success", "message": "Meme send tracked"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
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
        user_dict = user if isinstance(user, dict) else user.to_dict()
        update_user(request.user_id, {
            'total_memes_sent': user_dict.get('total_memes_sent', 0) + 1
        })
        
        # Update meme stats
        meme = Meme(**meme_dict)
        meme.total_shares += 1
        update_meme(meme)
        
        return {"status": "success", "message": "Meme send tracked"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Meme approval/rejection endpoints ====================
@app.get("/tags/options")
async def get_tag_options():
    """Return all predefined tag options from the ML model, grouped by category"""
    return {
        "visual_type": VISUAL_TYPE_OPTIONS,
        "visual_people": VISUAL_PEOPLE_OPTIONS,
        "visual_action": VISUAL_ACTION_OPTIONS,
        "visual_format": VISUAL_FORMAT_OPTIONS,
        "context_emotion": EMOTION_OPTIONS,
        "context_vibe": VIBE_OPTIONS,
        "context_situation": SITUATION_OPTIONS,
        "context_social": SOCIAL_OPTIONS
    }

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

# ==================== RECOMMENDATION ENDPOINT ====================

@app.post("/recommendations")
async def get_meme_recommendations(request: RecommendationRequest):
    """
    Get personalized meme recommendations based on user context
    
    This endpoint orchestrates the entire recommendation flow:
    - Session management
    - Exploration (new meme discovery)
    - Curated mix (personal favorites + global trends)
    - Pattern matching (personal + global)
    - Semantic search (cold start solution)
    - Thompson Sampling (quality + exploration balance)
    - Diversity filtering
    """    
    try:
        # Validate user exists
        user = get_user(request.user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"User {request.user_id} not found"
            )
        
        # Get recommendations
        result = get_recommendations(
            user_id=request.user_id,
            context=request.context,
            session_id=request.session_id,
            required_tags=request.required_tags,
            batch_size=request.batch_size
        )
        
        # Check for errors from recommendation engine
        if 'error' in result:
            raise HTTPException(
                status_code=500,
                detail=result['error']
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation engine error: {str(e)}"
        )

@app.post("/recommendations/feedback")
async def record_recommendation_feedback(request: RecommendationFeedbackRequest):
    """    
    Call this when a user SENDS a meme (not just views it)
    This records the successful context-meme pairing for future learning
    
    What this does:
    1. Gets the CLIP embedding for the context
    2. Updates the user's PERSONAL context patterns
    3. Updates the GLOBAL context patterns (community learning)
    4. Records meme in user's interaction history
    
    Flow:
    - User types: "stressed about work deadlines"
    - App calls /recommendations with that context
    - User picks "Hide the Pain Harold" and sends it
    - App calls THIS endpoint with context + meme_id
    - System learns: "stressed about work" → "Hide the Pain Harold" works!

    """
    from context_analyzer import (
        get_context_embedding,
        update_personal_pattern,
        update_global_patterns
    )
    
    try:
        # Validate user exists
        user = get_user(request.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Validate meme exists
        meme = get_meme(request.meme_id)
        if not meme:
            raise HTTPException(status_code=404, detail="Meme not found")
        
        # Get context embedding (for pattern matching)
        context_embedding = get_context_embedding(request.context)
        
        # Update PERSONAL context patterns
        # This learns what works for THIS user in similar situations
        personal_updated = update_personal_pattern(
            user_id=request.user_id,
            context_embedding=context_embedding,
            meme_id=request.meme_id
        )
        
        # Update GLOBAL context patterns
        # This learns what works for EVERYONE in similar situations
        global_updated = update_global_patterns(
            context_embedding=context_embedding,
            meme_id=request.meme_id
        )
        
        # Increment user's total context interaction count
        # This tracks user's overall experience level for exploration rate
        user_dict = user if isinstance(user, dict) else user.to_dict()
        current_count = user_dict.get('context_interaction_count', 0)
        new_count = current_count + 1
        
        update_user(request.user_id, {
            'context_interaction_count': new_count
        })
        
        # Return success with metadata
        return {
            "status": "success",
            "message": "Feedback recorded for pattern learning",
            "learning_updates": {
                "personal_pattern_updated": personal_updated,
                "global_pattern_updated": global_updated,
                "context_interaction_count": new_count,
                "context": request.context[:50] + "..." if len(request.context) > 50 else request.context
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Don't fail the user flow if learning fails
        # Log the error but return success
        print(f"❌ Learning error: {str(e)}")
        return {
            "status": "partial_success",
            "message": "Meme sent but learning update failed",
            "error": str(e)
        }

