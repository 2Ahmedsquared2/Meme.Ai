
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks, Form  # pyright: ignore[reportMissingImports]
from app.db import (
    db, User, Meme, Notification,
    create_user, get_user, update_user, create_meme, get_meme, update_meme, delete_meme, 
    upload_image_to_storage,
    create_notification, get_user_notifications, mark_notification_read, 
    mark_all_notifications_read, get_unread_notification_count
)
import uuid
from fastapi.middleware.cors import CORSMiddleware  # pyright: ignore[reportMissingImports]
from fastapi.responses import JSONResponse, FileResponse  # pyright: ignore[reportMissingImports]
from fastapi import HTTPException, status  # pyright: ignore[reportMissingImports]
from fastapi.staticfiles import StaticFiles  # pyright: ignore[reportMissingImports]
from typing import Dict, List, Optional
from app.recommendation_engine import update_tag_affinity
from app.recommendation_engine_v2 import recommend_memes_v2
from app.meme_analyzer import (
    analyze_meme as gemini_analyze_meme, 
    analyze_context, 
    analyze_context_from_upload,
    get_frontend_tags_by_category,
    get_rate_status as get_meme_analysis_rate_status
)
from app.meme_customizer import customize_meme
from app.transformer import (
    get_image_embedding,
    get_text_embedding
)
from app.ml_model import (
    VISUAL_TYPE_OPTIONS, VISUAL_PEOPLE_OPTIONS, VISUAL_ACTION_OPTIONS, VISUAL_FORMAT_OPTIONS,
    EMOTION_OPTIONS, VIBE_OPTIONS, SITUATION_OPTIONS, SOCIAL_OPTIONS
)
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
    context: Optional[str] = None
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
    user_tags: Optional[List[str]] = None          # Custom tags (descriptive)
    frontend_tags: Optional[List[str]] = None      # 62 predefined tags for rec engine
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
    Run Gemini auto-tagging and CLIP embedding
    
    NEW FLOW (Gemini + CLIP):
    1. Upload image to storage
    2. Gemini analyzes → use_case, visual_tags, contextual_tags, frontend_tags
    3. CLIP embeds the image → clip_embedding
    4. CLIP embeds the use_case → use_case_embedding
    5. Save everything to Firestore
    """
    # Compute hash from bytes (for duplicate detection)
    img = Image.open(BytesIO(image_bytes))
    img_hash = str(imagehash.phash(img))
    
    # Check if duplicate exists
    existing = db.collection('memes').where('image_hash', '==', img_hash).limit(1).get()
    if len(list(existing)) > 0:
        raise HTTPException(
            status_code=409, 
            detail="This meme already exists in our database"
        )
    
    # Upload to Firebase Storage (only if unique)
    image_url = upload_image_to_storage(image_bytes, filename)
    
    # ========== GEMINI TAGGING ==========
    print(f"🤖 Running Gemini analysis for {meme_id}...")
    gemini_result = gemini_analyze_meme(image_url)
    
    use_case = gemini_result.get("use_case", "")
    visual_tags = gemini_result.get("visual_tags", [])
    contextual_tags = gemini_result.get("contextual_tags", [])
    frontend_tags = gemini_result.get("frontend_tags", [])
    is_template = gemini_result.get("is_template", False)
    text_zones = gemini_result.get("text_zones", [])
    
    # Combine all tags
    all_tags = list(set(visual_tags + contextual_tags + frontend_tags))
    
    print(f"✅ Gemini generated: {len(all_tags)} tags, use_case: '{use_case[:50]}...'")
    
    # ========== CLIP EMBEDDINGS ==========
    print(f"🔄 Generating CLIP embeddings...")
    
    # Image embedding (for visual similarity)
    clip_embedding = get_image_embedding(image_url)
    
    # Use case embedding (for context matching)
    use_case_embedding = []
    if use_case:
        use_case_embedding = get_text_embedding(use_case)
    
    print(f"✅ CLIP embeddings: image={len(clip_embedding)} dims, use_case={len(use_case_embedding)} dims")
    
    # ========== CREATE MEME OBJECT ==========
    new_meme = Meme(
        id=meme_id,
        image_url=image_url,
        image_hash=img_hash,
        meme_type='customizable' if is_template else 'rec_engine',  # Gemini detects meme type
        is_template=is_template,  # Legacy - kept for backward compatibility
        text_zones=text_zones,  # Where text can be placed on templates
        user_tags=[],  # Empty - admin adds custom tags
        visual_tags=visual_tags,
        contextual_tags=contextual_tags,
        frontend_tags=frontend_tags,  # Gemini suggests frontend tags for rec engine
        blip2_caption="",  # Not using BLIP anymore
        all_tags=all_tags,
        clip_embedding=clip_embedding,
        use_case=use_case,
        use_case_embedding=use_case_embedding,
        status="pending"
    )
    
    create_meme(new_meme)
    
    # Get current rate status for response
    rate_status = get_meme_analysis_rate_status()
    
    return {
        "status": "success", 
        "meme_id": meme_id, 
        "image_url": image_url,
        "gemini_analysis": {
            "use_case": use_case,
            "visual_tags_count": len(visual_tags),
            "contextual_tags_count": len(contextual_tags),
            "frontend_tags": frontend_tags,
            "is_template": is_template,
            "text_zones_count": len(text_zones)
        },
        "embeddings": {
            "clip_image": len(clip_embedding),
            "use_case": len(use_case_embedding)
        },
        "rate_limit": rate_status,
        "note": "Meme analyzed with Gemini + CLIP. Add custom tags in admin panel."
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

@app.get("/users/{user_id}/tag-preferences")
async def get_user_tag_preferences(user_id: str, limit: int = 10):
    """
    Get user's top tag preferences (for debugging/analytics).
    Shows which tags the user has shown affinity for based on their interactions.
    """
    from app.recommendation_engine import get_user_top_tags
    
    try:
        top_tags = get_user_top_tags(user_id, limit=limit)
        
        if not top_tags:
            return {
                "user_id": user_id,
                "message": "No tag preferences recorded yet",
                "top_tags": []
            }
        
        return {
            "user_id": user_id,
            "top_tags": top_tags
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
# NOTE: Specific routes like /memes/pending must come BEFORE /memes/{meme_id}

@app.get("/memes/pending")
async def get_pending_memes_route(limit: int = 100):
    """Get pending memes for admin review"""
    try:
        docs = db.collection('memes').where('status', '==', 'pending').limit(limit).get()
        memes = [doc.to_dict() for doc in docs]
        return {"memes": memes, "count": len(memes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memes/approved")
async def get_approved_memes_route(limit: int = 100):
    """Get approved memes for admin gallery"""
    try:
        docs = db.collection('memes').where('status', '==', 'approved').limit(limit).get()
        memes = [doc.to_dict() for doc in docs]
        return {"memes": memes, "count": len(memes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memes/templates")
async def get_template_memes(limit: int = 30, tags: List[str] = Query(default=[])):
    """Get customizable meme templates (is_template=true or meme_type in ['customizable', 'both'])"""
    try:
        # Get approved templates
        query = db.collection('memes').where('status', '==', 'approved').where('is_template', '==', True)
        
        if tags and len(tags) > 0:
            query = query.where('frontend_tags', 'array_contains_any', tags)
        
        docs = query.limit(limit).get()
        
        memes = []
        for doc in docs:
            memes.append(doc.to_dict())
        
        return {"memes": memes, "count": len(memes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memes/search")
async def search_memes(limit: int = 30, tags: List[str] = Query(default=[])):
    """Search approved memes for the Search tab (excludes customizable-only memes)"""
    try:
        query = db.collection('memes').where('status', '==', 'approved')

        if tags and len(tags) > 0:
            query = query.where('frontend_tags', 'array_contains_any', tags)

        # Fetch more to account for filtering
        docs = query.limit(limit * 2).get()

        memes = []
        for doc in docs:
            meme_data = doc.to_dict()
            # Exclude memes that are ONLY for customization (not for rec engine/browsing)
            meme_type = meme_data.get('meme_type', 'rec_engine')
            if meme_type != 'customizable':
                memes.append(meme_data)
                if len(memes) >= limit:
                    break

        return {"memes": memes, "count": len(memes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

def background_analyze_meme(meme_id: str, image_url: str):
    """Background task to run Gemini analysis and CLIP embedding on a meme"""
    try:
        print(f"🔄 Background analysis started for {meme_id}")
        
        # Run Gemini analysis
        gemini_result = gemini_analyze_meme(image_url, wait_if_limited=True)
        
        use_case = gemini_result.get("use_case", "")
        visual_tags = gemini_result.get("visual_tags", [])
        contextual_tags = gemini_result.get("contextual_tags", [])
        frontend_tags = gemini_result.get("frontend_tags", [])
        is_template = gemini_result.get("is_template", False)
        text_zones = gemini_result.get("text_zones", [])
        all_tags = list(set(visual_tags + contextual_tags + frontend_tags))
        
        # Run CLIP embedding
        clip_embedding = get_image_embedding(image_url)
        use_case_embedding = get_text_embedding(use_case) if use_case else []
        
        # Update meme in database
        meme_type = 'customizable' if is_template else 'rec_engine'
        
        db.collection('memes').document(meme_id).update({
            'status': 'pending',  # Ready for admin review
            'meme_type': meme_type,
            'is_template': is_template,
            'text_zones': text_zones,
            'visual_tags': visual_tags,
            'contextual_tags': contextual_tags,
            'frontend_tags': frontend_tags,
            'all_tags': all_tags,
            'use_case': use_case,
            'clip_embedding': clip_embedding,
            'use_case_embedding': use_case_embedding
        })
        
        print(f"✅ Background analysis completed for {meme_id}")
        
    except Exception as e:
        print(f"❌ Background analysis failed for {meme_id}: {e}")
        # Update status to indicate analysis failed
        db.collection('memes').document(meme_id).update({
            'status': 'pending',  # Still pending, admin can retry or manually tag
            'analysis_error': str(e)
        })


@app.post("/memes/upload")
async def upload_meme_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(default=None)
):
    """
    Fast upload endpoint - returns quickly after duplicate check and storage upload.
    Gemini analysis runs in background.
    """
    try:
        print(f"📤 Upload request received: {file.filename} from user: {user_id}")
        
        # Read file
        file_bytes = await file.read()
        print(f"   File size: {len(file_bytes)} bytes")
        
        # Check file size
        max_size = 10 * 1024 * 1024  # 10 MB
        if len(file_bytes) > max_size:
            raise HTTPException(
                status_code=400, 
                detail="Image is too large. Maximum size is 10 MB."
            )
        
        # Validate it's an image
        try:
            img = Image.open(BytesIO(file_bytes))
            print(f"   Image format: {img.format}, size: {img.size}")
            img.verify()
        except Exception as img_err:
            print(f"   ❌ Image validation failed: {img_err}")
            raise HTTPException(
                status_code=400, 
                detail="Uploaded file is not a valid image."
            )
        
        # Check for duplicates using perceptual hash
        img = Image.open(BytesIO(file_bytes))  # Re-open after verify
        img_hash = str(imagehash.phash(img))
        
        existing = db.collection('memes').where('image_hash', '==', img_hash).limit(1).get()
        if len(list(existing)) > 0:
            raise HTTPException(
                status_code=409, 
                detail="This meme already exists in our database"
            )
        
        # Generate IDs
        timestamp = datetime.utcnow().timestamp()
        filename = f"meme_{timestamp}_{file.filename}"
        meme_id = f"meme_{int(timestamp)}"
        print(f"   Meme ID: {meme_id}")
        
        # Upload to Firebase Storage
        image_url = upload_image_to_storage(file_bytes, filename)
        print(f"   ✅ Uploaded to storage: {image_url[:50]}...")
        
        # Create meme with minimal data (pending_analysis status)
        new_meme = Meme(
            id=meme_id,
            image_url=image_url,
            image_hash=img_hash,
            status="pending_analysis",  # Will become "pending" after analysis
            submitted_by=user_id,
            meme_type='rec_engine',  # Default, will be updated by Gemini
            created_at=datetime.now()
        )
        create_meme(new_meme)
        
        # Queue background analysis
        background_tasks.add_task(background_analyze_meme, meme_id, image_url)
        print(f"   🔄 Queued background analysis")
        
        return {
            "status": "success",
            "message": "Meme submitted! We'll review it soon.",
            "meme_id": meme_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



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
        if request.frontend_tags is not None:
            meme.frontend_tags = request.frontend_tags

        # Update all_tags to be combination of all tag types
        meme.all_tags = list(set(meme.visual_tags + meme.contextual_tags + meme.user_tags + meme.frontend_tags))

        update_meme(meme)

        return {
            "status": "success",
            "meme_id": meme_id,
            "visual_tags": meme.visual_tags,
            "contextual_tags": meme.contextual_tags,
            "user_tags": meme.user_tags,
            "frontend_tags": meme.frontend_tags,
            "all_tags": meme.all_tags
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UpdateMemeRequest(BaseModel):
    """Request to update meme fields"""
    use_case: Optional[str] = None
    meme_type: Optional[str] = None  # "customizable", "rec_engine", or "both"
    is_template: Optional[bool] = None  # Legacy - kept for backward compatibility


@app.patch("/memes/{meme_id}")
async def update_meme_fields(meme_id: str, request: UpdateMemeRequest):
    """Update meme fields like use_case, is_template (used by admin panel)"""
    try:
        meme_dict = get_meme(meme_id)
        if not meme_dict:
            raise HTTPException(status_code=404, detail="Meme not found")

        meme = Meme(**meme_dict)

        # Update use_case if provided
        if request.use_case is not None:
            meme.use_case = request.use_case
            # Re-generate use_case embedding with the new text
            if request.use_case:
                from app.transformer import get_text_embedding
                meme.use_case_embedding = get_text_embedding(request.use_case)
            else:
                meme.use_case_embedding = []

        # Update meme_type if provided
        if request.meme_type is not None:
            if request.meme_type in ["customizable", "rec_engine", "both"]:
                meme.meme_type = request.meme_type
                # Also update legacy is_template for backward compatibility
                meme.is_template = request.meme_type in ["customizable", "both"]

        # Update is_template if provided (legacy support)
        if request.is_template is not None:
            meme.is_template = request.is_template

        update_meme(meme)

        return {
            "status": "success",
            "meme_id": meme_id,
            "use_case": meme.use_case,
            "meme_type": meme.meme_type,
            "is_template": meme.is_template
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
async def get_all_memes(limit: int = 30, status: str = "approved", tags: List[str] = Query(default=[]), include_all_types: bool = False):
    """Get memes based on filters. Set include_all_types=true for admin to see all meme types."""
    try:
        #collect the memes
        query = db.collection('memes').where('status', '==', status)
        #add tag filters
        if tags and len(tags) > 0:
            query = query.where('frontend_tags', 'array_contains_any', tags)

        # Fetch more to account for filtering (if needed)
        fetch_limit = limit if include_all_types else limit * 2
        docs = query.limit(fetch_limit).get()

        memes = []
        for doc in docs:
            meme_data = doc.to_dict()
            # If include_all_types, don't filter. Otherwise exclude customizable-only memes.
            if include_all_types:
                memes.append(meme_data)
            else:
                meme_type = meme_data.get('meme_type', 'rec_engine')
                if meme_type != 'customizable':
                    memes.append(meme_data)
            
            if len(memes) >= limit:
                break

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
        
        # Validate reaction - accept both formats
        reaction = request.reaction.lower()
        if reaction in ["thumbs_up", "up"]:
            reaction = "up"
        elif reaction in ["thumbs_down", "down"]:
            reaction = "down"
        else:
            raise HTTPException(status_code=400, detail="Reaction must be 'up', 'down', 'thumbs_up', or 'thumbs_down'")
        
        meme = Meme(**meme_dict)
        
        # Update meme stats
        if reaction == "up":
            user.total_memes_thumbed_up = (user.total_memes_thumbed_up or 0) + 1
            meme.total_thumbs_up += 1
            # Update tag affinities (positive signal)
            update_tag_affinity(request.user_id, request.meme_id, "thumbs_up")
        else:
            user.total_memes_thumbed_down = (user.total_memes_thumbed_down or 0) + 1
            meme.total_thumbs_down += 1
            # Update tag affinities (negative signal)
            update_tag_affinity(request.user_id, request.meme_id, "thumbs_down")
        
        update_user(user)
        update_meme(meme)
        
        return {"status": "success", "message": f"Thumbs {reaction} tracked"}
    
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
        favorited = user.favorited_meme_ids or []
        
        if request.meme_id not in favorited:
            # Add to favorites
            favorited.append(request.meme_id)
            user.favorited_meme_ids = favorited
            user.total_memes_favorited = (user.total_memes_favorited or 0) + 1
            update_user(user)
            
            # Update meme stats
            meme = Meme(**meme_dict)
            meme.total_favorites += 1
            update_meme(meme)
            
            # Update tag affinities (strong positive signal)
            update_tag_affinity(request.user_id, request.meme_id, "favorite")
            
            return {"status": "success", "message": "Meme added to favorites"}
        else:
            return {"status": "success", "message": "Meme already in favorites"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

@app.get("/interactions/favorites/{user_id}")
async def get_favorites(user_id: str):
    """Get all favorited memes for a user"""
    try:
        user = get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get favorited meme IDs
        favorited_ids = user.favorited_meme_ids or []
        
        # Fetch all favorited memes
        memes = []
        for meme_id in favorited_ids:
            meme_dict = get_meme(meme_id)
            if meme_dict:
                memes.append(meme_dict)
        
        return {"memes": memes, "count": len(memes)}
    
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
        
        # Get current favorites
        favorited = user.favorited_meme_ids or []
        
        # Check if it's in favorites
        if meme_id in favorited:
            # Remove from favorites
            favorited.remove(meme_id)
            user.favorited_meme_ids = favorited
            user.total_memes_favorited = max(0, (user.total_memes_favorited or 0) - 1)
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
        
        # Update tag affinities (strongest positive signal!)
        update_tag_affinity(request.user_id, request.meme_id, "send")
        
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

# ==================== Custom Tags Management (Persisted in Firestore) ====================
@app.get("/tags/custom")
async def get_custom_tags():
    """Get all custom tags from Firestore"""
    try:
        doc = db.collection('settings').document('custom_tags').get()
        if doc.exists:
            return {"tags": doc.to_dict().get('tags', [])}
        return {"tags": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tags/custom")
async def save_custom_tags(tags: List[str]):
    """Save custom tags to Firestore"""
    try:
        db.collection('settings').document('custom_tags').set({'tags': tags})
        return {"status": "success", "tags": tags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/tags/custom/{tag_name}")
async def delete_custom_tag(tag_name: str, remove_from_memes: bool = True):
    """Delete a custom tag and optionally remove it from all memes"""
    try:
        # Remove from custom tags list
        doc = db.collection('settings').document('custom_tags').get()
        if doc.exists:
            tags = doc.to_dict().get('tags', [])
            tags = [t for t in tags if t.lower() != tag_name.lower()]
            db.collection('settings').document('custom_tags').set({'tags': tags})
        
        # Remove from all memes if requested
        removed_count = 0
        if remove_from_memes:
            memes = db.collection('memes').stream()
            for meme_doc in memes:
                meme_data = meme_doc.to_dict()
                user_tags = meme_data.get('user_tags', [])
                original_len = len(user_tags)
                user_tags = [t for t in user_tags if t.lower() != tag_name.lower()]
                
                if len(user_tags) < original_len:
                    # Update meme with tag removed
                    all_tags = (meme_data.get('visual_tags', []) + 
                               meme_data.get('contextual_tags', []) + 
                               user_tags)
                    db.collection('memes').document(meme_doc.id).update({
                        'user_tags': user_tags,
                        'all_tags': all_tags
                    })
                    removed_count += 1
        
        return {
            "status": "success", 
            "tag_deleted": tag_name,
            "memes_updated": removed_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memes/{meme_id}/approve")
async def approve_meme(meme_id: str):
    """Admin endpoint to approve a meme"""
    try:
        # Get the meme
        meme_dict = get_meme(meme_id)

        if not meme_dict:
            raise HTTPException(status_code=404, detail="Meme not found")

        # Update status to approved and set approved_at timestamp
        db.collection('memes').document(meme_id).update({
            'status': 'approved',
            'approved_at': datetime.utcnow()
        })

        # Create notification if meme was submitted by a user
        submitted_by = meme_dict.get('submitted_by')
        if submitted_by:
            notification = Notification(
                id=f"notif_{uuid.uuid4().hex[:12]}",
                user_id=submitted_by,
                type="meme_approved",
                title="🎉 Your meme was approved!",
                message="Your submitted meme is now live. Tap to view and add to favorites!",
                meme_id=meme_id,
                meme_image_url=meme_dict.get('image_url'),
                created_at=datetime.utcnow()
            )
            create_notification(notification)

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

        # Create notification if meme was submitted by a user
        submitted_by = meme_dict.get('submitted_by')
        if submitted_by:
            notification = Notification(
                id=f"notif_{uuid.uuid4().hex[:12]}",
                user_id=submitted_by,
                type="meme_rejected",
                title="Meme not approved",
                message="Your submitted meme wasn't approved this time. Try submitting another one!",
                meme_id=meme_id,
                meme_image_url=meme_dict.get('image_url'),
                created_at=datetime.utcnow()
            )
            create_notification(notification)

        return {"status": "success", "message": "Meme rejected"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== NOTIFICATION ENDPOINTS ====================

@app.get("/notifications/{user_id}")
async def get_notifications(user_id: str, unread_only: bool = False, limit: int = 50):
    """Get notifications for a user"""
    try:
        notifications = get_user_notifications(user_id, unread_only=unread_only, limit=limit)
        unread_count = get_unread_notification_count(user_id)
        return {
            "notifications": notifications,
            "unread_count": unread_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/notifications/{user_id}/count")
async def get_notification_count(user_id: str):
    """Get unread notification count for a user"""
    try:
        count = get_unread_notification_count(user_id)
        return {"unread_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/notifications/{notification_id}/read")
async def mark_read(notification_id: str):
    """Mark a notification as read"""
    try:
        mark_notification_read(notification_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/notifications/{user_id}/read-all")
async def mark_all_read(user_id: str):
    """Mark all notifications as read for a user"""
    try:
        mark_all_notifications_read(user_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== RECOMMENDATION ENDPOINT ====================

@app.post("/recommendations")
async def get_meme_recommendations(request: RecommendationRequest):
    """
    Get personalized meme recommendations (V2 - Parallel Scoring)
    """    
    try:
        # Validate user exists
        user = get_user(request.user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"User {request.user_id} not found"
            )
        
        # Get context embedding
        from app.meme_analyzer import analyze_context
        from app.transformer import get_text_embedding
        
        result = analyze_context(text=request.context)
        query = result.get('meme_search_query', '')
        
        if not query:
            raise HTTPException(status_code=400, detail="Failed to analyze context")
        
        context_embedding = get_text_embedding(query)
        
        # Get recommendations using V2
        rec_result = recommend_memes_v2(
            user_id=request.user_id,
            context=request.context,
            context_embedding=context_embedding,
            exclude_ids=[],
            num_results=request.batch_size
        )
        
        # Check for errors
        if 'error' in rec_result.get('metadata', {}):
            raise HTTPException(
                status_code=500,
                detail=rec_result['metadata']['error']
            )
        
        return rec_result
        
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
    1. Uses Gemini to analyze context → CLIP embedding
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
    from app.transformer import (
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

# ============================================================================
# MEME CUSTOMIZATION ENDPOINTS
# ============================================================================

class AnalyzeMemeRequest(BaseModel):
    image_url: str

class CustomizeMemeRequest(BaseModel):
    image_url: str
    text_inputs: Dict[int, str]  # {zone_id: "user text"}
    text_zones: List[dict]       # From analyze response

@app.post("/memes/analyze")
async def analyze_meme_endpoint(request: AnalyzeMemeRequest):
    """Analyze a meme to get tags and text zones (for templates)"""
    result = gemini_analyze_meme(request.image_url)
    if not result.get("use_case"):
        raise HTTPException(status_code=500, detail="Failed to analyze meme")
    return result

@app.post("/memes/customize")
async def customize_meme_endpoint(request: CustomizeMemeRequest):
    """Generate a customized meme with user text"""
    # Convert string keys to int (JSON serialization issue)
    clean_inputs = {int(k): v for k, v in request.text_inputs.items()}
    
    result = customize_meme(request.image_url, clean_inputs, request.text_zones)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return {
        "image_data": result["image_data"],
        "mime_type": result["mime_type"]
    }

@app.get("/tags/frontend")
async def get_frontend_tags():
    """Get all frontend tags organized by category (for iOS filter UI)"""
    return get_frontend_tags_by_category()

@app.get("/memes/rate-status")
async def get_meme_rate_status():
    """
    Get rate limit status for meme analysis.
    
    Meme tagging is limited to 10/min to preserve Gemini quota for context analysis.
    Context analysis (user-facing) has NO rate limit - always priority.
    
    Returns:
        requests_used: How many requests made in last minute
        requests_remaining: How many more can be made
        max_requests: Limit per minute (10)
        window_seconds: Time window (60)
        wait_seconds: Seconds until next request allowed (0 if can proceed)
    """
    return get_meme_analysis_rate_status()

# ============================================================================
# CONTEXT ANALYSIS ENDPOINTS (Gemini-powered)
# ============================================================================

class AnalyzeContextRequest(BaseModel):
    text: Optional[str] = None
    image_url: Optional[str] = None

@app.post("/context/analyze")
async def analyze_context_endpoint(request: AnalyzeContextRequest):
    """
    Analyze user context (text and/or screenshot) to generate a meme search query.
    
    This uses Gemini to understand the context and output a description
    in the same style as meme use_cases for optimal CLIP matching.
    
    Returns:
        meme_search_query: A 1-2 sentence description ready for CLIP embedding
    """
    if not request.text and not request.image_url:
        raise HTTPException(status_code=400, detail="Must provide either text or image_url")
    
    result = analyze_context(text=request.text, image_url=request.image_url)
    
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    
    return {"meme_search_query": result["meme_search_query"]}

@app.post("/context/analyze/upload")
async def analyze_context_upload_endpoint(
    file: UploadFile = File(...),
    text: Optional[str] = None
):
    """
    Analyze context from an uploaded screenshot (for iOS direct upload).
    
    Args:
        file: Uploaded screenshot image
        text: Optional additional text context
        
    Returns:
        meme_search_query: A 1-2 sentence description ready for CLIP embedding
    """
    # Read file
    file_bytes = await file.read()
    
    # Validate it's an image
    try:
        img = Image.open(BytesIO(file_bytes))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    
    result = analyze_context_from_upload(image_bytes=file_bytes, text=text)
    
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    
    return {"meme_search_query": result["meme_search_query"]}

