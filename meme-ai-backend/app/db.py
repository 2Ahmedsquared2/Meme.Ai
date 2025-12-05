import firebase_admin   # pyright: ignore[reportMissingImports]
from firebase_admin import credentials, firestore, storage  # pyright: ignore[reportMissingImports]
import os
from dataclasses import dataclass, field 
from datetime import datetime 
from typing import List, Dict, Optional 


# Get the directory of this file, then go up to backend root
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_root = os.path.dirname(current_dir)
cred_path = os.path.join(backend_root, "Credentials4.json")

# Debug: print the path to verify it's correct
print(f"🔍 Looking for credentials at: {cred_path}")

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'projectId': 'get-meme-ai', 
    'storageBucket': 'get-meme-ai.appspot.com'
})

db = firestore.client()
bucket = storage.bucket()

@dataclass 
class Meme:
    """ Represents memes in the database """
    id: str
    image_url: str
    firebase_image_url: Optional[str] = None
    image_hash: str = ""
    status: str = "approved"

    # Meme tagging system 
    user_tags: List[str] = field(default_factory=list)
    visual_tags: List[str] = field(default_factory=list)
    contextual_tags: List[str] = field(default_factory=list)
    blip2_caption: str = ""
    clip_embedding: List[float] = field(default_factory=list)
    all_tags: List[str] = field(default_factory=list)
    
    # Use case - describes WHEN to use this meme (human or AI generated)
    use_case: str = ""
    use_case_embedding: List[float] = field(default_factory=list)

    # timestamps
    created_at: datetime = field(default_factory=datetime.now)
    upload_time: Optional[datetime] = None
    approved_at: Optional[datetime] = None

    # RL variables (using for thompson sampling)
    base_alpha: float = 1.0 # for approval
    base_beta: float = 1.0 # for disapproval

    # Meme engagement stats
    total_sends: int = 0
    total_likes: int = 0
    total_favorites: int = 0
    total_thumbs_up: int = 0
    total_thumbs_down: int = 0
    total_views: int = 0

    #time window engagement stats
    sends_today: int = 0
    sends_week: int = 0
    sends_month: int = 0
    sends_year: int = 0

    # meme popularity score
    trending_score: float = 0.0             # Growth rate (for trending badge)
    popularity_score_today: float = 0.0
    popularity_score_week: float = 0.0
    popularity_score_month: float = 0.0
    popularity_score_year: float = 0.0
    popularity_score_all_time: float = 0.0

    # Meme ranking 
    rank_trending: Optional[int] = None
    rank_today: Optional[int] = None
    rank_week: Optional[int] = None
    rank_month: Optional[int] = None
    rank_year: Optional[int] = None
    rank_all_time: Optional[int] = None

    # Meme Badges (top 3 for each category)
    badge_trending: Optional[str] = None    # "🔥 #1", "🔥 #2", "🔥 #3" (growth rate)
    badge_week: Optional[str] = None        # "⭐ #1", "⭐ #2", "⭐ #3" (popular now)
    badge_all_time: Optional[str] = None    # "👑 Gold", "👑 Silver", "👑 Bronze" (legendary)

    # Converts the objects to a dictionary for Firestore
    def to_dict(self) -> Dict:
        """Convert Meme object to Firestore-compatible dictionary"""
        return {
            'id': self.id,
            'image_url': self.image_url,
            'firebase_image_url': self.firebase_image_url,
            'image_hash': self.image_hash,
            'status': self.status,
            'user_tags': self.user_tags,
            'visual_tags': self.visual_tags,
            'contextual_tags': self.contextual_tags,
            'blip2_caption': self.blip2_caption,
            'all_tags': self.all_tags,
            'clip_embedding': self.clip_embedding,
            'use_case': self.use_case,
            'use_case_embedding': self.use_case_embedding,
            'created_at': self.created_at,
            'upload_time': self.upload_time,
            'approved_at': self.approved_at,
            'base_alpha': self.base_alpha,
            'base_beta': self.base_beta,
            'total_sends': self.total_sends,
            'total_likes': self.total_likes,
            'total_favorites': self.total_favorites,
            'total_thumbs_up': self.total_thumbs_up,
            'total_thumbs_down': self.total_thumbs_down,
            'total_views': self.total_views,
            'sends_today': self.sends_today,
            'sends_week': self.sends_week,
            'sends_month': self.sends_month,
            'sends_year': self.sends_year,
            'trending_score': self.trending_score,
            'popularity_score_today': self.popularity_score_today,
            'popularity_score_week': self.popularity_score_week,
            'popularity_score_month': self.popularity_score_month,
            'popularity_score_year': self.popularity_score_year,
            'popularity_score_all_time': self.popularity_score_all_time,
            'rank_trending': self.rank_trending,
            'rank_today': self.rank_today,
            'rank_week': self.rank_week,
            'rank_month': self.rank_month,
            'rank_year': self.rank_year,
            'rank_all_time': self.rank_all_time,
            'badge_trending': self.badge_trending,
            'badge_week': self.badge_week,
            'badge_all_time': self.badge_all_time,
        }


@dataclass
class User:
    """ Represents users in the database """
    #basic auth info
    id: str
    email: Optional[str] = None
    phone_number: Optional[str] = None
    auth_provider: str = "email"
    created_at: datetime = field(default_factory=datetime.now)

    # Legal/compliance
    terms_accepted: bool = False
    terms_accepted_at: Optional[datetime] = None

    #onboarding status
    onboarding_completed: bool = False
    onboarding_data: List[Dict] = field(default_factory=list)
    quick_rating_completed: bool = False


    # User preferences
    preference_embedding: List[float] = field(default_factory=list)
    tag_affinities: Dict[str, float] = field(default_factory=dict)

    #context patterns for rec engine 
    context_patterns: List[Dict] = field(default_factory=list)
    favorited_meme_ids: List[str] = field(default_factory=list)
    context_interaction_count: int = 0


    #user stats 
    total_memes_sent: int = 0
    total_memes_favorited: int = 0
    total_memes_thumbed_up: int = 0
    total_memes_thumbed_down: int = 0
    total_memes_viewed: int = 0
    last_active_at: Optional[datetime] = None


    # NEW: New meme exploration tracking
    exploration_query_count: int = 0
    next_exploration_at: int = 3  
    

    def to_dict(self) -> Dict:
        """Convert User object to dictionary for Firestore"""
        return {
            'id': self.id,
            'email': self.email,
            'phone_number': self.phone_number,
            'auth_provider': self.auth_provider,
            'created_at': self.created_at,
            'last_active_at': self.last_active_at,
            
            # Legal/compliance
            'terms_accepted': self.terms_accepted,
            'terms_accepted_at': self.terms_accepted_at,
            
            # Onboarding
            'onboarding_completed': self.onboarding_completed,
            'onboarding_data': self.onboarding_data,
            'quick_rating_completed': self.quick_rating_completed,

            #context patterns for rec engine 
            'context_patterns': self.context_patterns,
            'favorited_meme_ids': self.favorited_meme_ids,
            'context_interaction_count': self.context_interaction_count,
            
            # Preferences
            'preference_embedding': self.preference_embedding,
            'tag_affinities': self.tag_affinities,
            
            # Usage stats
            'total_memes_sent': self.total_memes_sent,
            'total_memes_favorited': self.total_memes_favorited,
            'total_memes_thumbed_up': self.total_memes_thumbed_up,
            'total_memes_thumbed_down': self.total_memes_thumbed_down,
            'total_memes_viewed': self.total_memes_viewed,

            # NEW: New meme exploration tracking
            'exploration_query_count': self.exploration_query_count,
            'next_exploration_at': self.next_exploration_at,
        }

@dataclass
class GlobalContextPattern:
    """
    Global context patterns learned from ALL users
    Used for cold-start recommendations for new users
    """
    context_text:str
    context_embedding: List[float]
    successful_meme_ids: List[str]
    common_tags: List[str]
    average_embedding: List[float]
    total_successes: int = 0
    total_attempts: int = 0
    success_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to Firestore-compatible dictionary"""
        return {
            'context_text': self.context_text,
            'context_embedding': self.context_embedding,
            'successful_meme_ids': self.successful_meme_ids,
            'common_tags': self.common_tags,
            'average_embedding': self.average_embedding,
            'total_successes': self.total_successes,
            'total_attempts': self.total_attempts,
            'success_rate': self.success_rate,
            'last_updated': self.last_updated
        }







    
# ==================== HELPER FUNCTIONS ====================
# MEME FUNCTIONS
def create_meme(meme: Meme) -> str:
    """
    Saves a new meme to Firestore
    ARGS:
        meme: Meme object to save
    Returns:
        meme_id: The ID of the new meme
    """
    doc_ref = db.collection('memes').document(meme.id)
    doc_ref.set(meme.to_dict())
    print(f"🎉 Meme created successfully with ID: {meme.id}")
    return meme.id

def get_meme(meme_id: str) -> Optional[Dict]:
    """
    Get a meme from Firestone by its meme ID

    ARGS:
        Meme_ID: the meme's ID

    Returns:
        Meme data as a dictionary, or None if not found
    """
    doc = db.collection('memes').document(meme_id).get()
    if doc.exists:
        return doc.to_dict()
    return None

def update_meme(meme: Meme) -> None:
    """
    Updates an existing meme in Firestore
    ARGS:
        meme: Meme object with updated data
    Returns:
        None
    """
    doc_ref = db.collection('memes').document(meme.id)
    doc_ref.set(meme.to_dict(), merge=True)
    print(f"🎉 Meme updated successfully with ID: {meme.id}")
    return None
def delete_meme(meme_id: str) -> None:
    """
    Deletes a meme from Firestore
    ARGS:
        meme_id: The ID of the meme to delete
    Returns:
        None
    """
    doc_ref = db.collection('memes').document(meme_id).delete()
    print(f"🗑️ Meme deleted successfully with ID: {meme_id}")


# USER FUNCTIONS
def create_user(user: User) -> str:
    """
    Saves a new user to Firestore
    ARGS:
        user: User object to save
    Returns:
        user_id: The ID of the new user
    """
    doc_ref = db.collection('users').document(user.id)
    doc_ref.set(user.to_dict())
    print(f"🎉 User created successfully with ID: {user.id}")
    return user.id

def get_user(user_id: str) -> Optional[User]:
    """
    Get a user from Firestore by their user ID
    ARGS:
        user_id: The ID of the user to retrieve
    Returns:
        User object, or None if not found
    """
    doc = db.collection('users').document(user_id).get()
    if doc.exists:
        data = doc.to_dict()
        return User(**data)
    return None

def update_user(user: User) -> None:
    """
    Updates an existing user in Firestore
    ARGS:
        user: User object with updated data
    Returns:
        None
    """
    doc_ref = db.collection('users').document(user.id)
    doc_ref.set(user.to_dict(), merge=True)
    print(f"🎉 User updated successfully with ID: {user.id}")

def delete_user(user_id: str) -> None:
    """
    Deletes a user from Firestore
    ARGS:
        user_id: The user's ID to delete
    Returns:
        None
    """
    db.collection('users').document(user_id).delete()
    print(f"🗑️ User deleted successfully with ID: {user_id}")

# MEME STORAGE FUNCTIONS
def upload_image_to_storage(file_bytes: bytes, filename: str) -> str:
    """
    Upload an image to Firebase Storage
    Returns the public URL of the uploaded image
    """
    #uploads the file bytes
    blob = bucket.blob(f"memes/{filename}")

    #upload the file bytes
    blob.upload_from_string(file_bytes, content_type='image/jpeg')
    
    #makes blob publicly 
    blob.make_public()

    #return the public url
    return blob.public_url
     
    





