"""
Transformer - CLIP embeddings and pattern matching
Handles: embeddings, cosine similarity, personal/global patterns
"""

import io
import numpy as np
import torch
import requests
from typing import List, Dict
from datetime import datetime
from PIL import Image

# Lazy-loaded CLIP
_clip_processor = None
_clip_model = None


def _load_clip():
    """Lazy load CLIP model"""
    global _clip_processor, _clip_model
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        print("Loading CLIP model...")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_processor, _clip_model


# =============================================================================
# EMBEDDINGS
# =============================================================================
def get_text_embedding(text: str) -> List[float]:
    """Get CLIP embedding for text"""
    processor, model = _load_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
    
    return embedding[0].tolist()


def get_image_embedding(image_url: str) -> List[float]:
    """Get CLIP embedding for image URL"""
    try:
        resp = requests.get(image_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        
        processor, model = _load_clip()
        inputs = processor(images=img, return_tensors="pt")
        
        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
        
        return embedding[0].tolist()
    except Exception as e:
        print(f"Image embedding error: {e}")
        return []


def get_image_embedding_from_bytes(image_bytes: bytes) -> List[float]:
    """Get CLIP embedding from raw image bytes"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        processor, model = _load_clip()
        inputs = processor(images=img, return_tensors="pt")
        
        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
        
        return embedding[0].tolist()
    except Exception as e:
        print(f"Image embedding error: {e}")
        return []


def get_context_embedding(context: str, image_url: str = None) -> List[float]:
    """
    Get embedding for user context.
    Uses meme_analyzer to rephrase context first for better matching.
    """
    from app.meme_analyzer import analyze_context
    
    # Get AI-rephrased query
    result = analyze_context(text=context, image_url=image_url)
    query = result.get("meme_search_query", "")
    
    # Fallback to raw context
    if not query:
        query = context
    
    return get_text_embedding(query)


# =============================================================================
# SIMILARITY
# =============================================================================
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Cosine similarity between two vectors"""
    if not vec1 or not vec2:
        return 0.0
    
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(v1, v2) / (norm1 * norm2))


def batch_cosine_similarity(query_vec: List[float], vectors: List[List[float]]) -> List[float]:
    """Compute similarity against multiple vectors at once"""
    if not query_vec or not vectors:
        return []
    
    q = np.array(query_vec)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return [0.0] * len(vectors)
    
    q = q / q_norm
    
    results = []
    for v in vectors:
        if not v:
            results.append(0.0)
            continue
        v_arr = np.array(v)
        v_norm = np.linalg.norm(v_arr)
        if v_norm == 0:
            results.append(0.0)
        else:
            results.append(float(np.dot(q, v_arr / v_norm)))
    
    return results


# =============================================================================
# PATTERN MATCHING
# =============================================================================
def find_personal_patterns(user_id: str, context_embedding: List[float], limit: int = 5) -> List[Dict]:
    """Find user's past successful patterns that match current context"""
    from app.db import get_user
    
    user = get_user(user_id)
    if not user:
        return []
    
    matches = []
    
    for pattern in user.context_patterns:
        if pattern.get("action") not in ["sent", "favorited"]:
            continue
        
        pattern_emb = pattern.get("context_embedding", [])
        if not pattern_emb:
            continue
        
        sim = cosine_similarity(context_embedding, pattern_emb)
        
        if sim > 0.7:
            conf = pattern.get("confidence", 1.0)
            matches.append({
                "similarity": sim,
                "confidence": conf,
                "tags": pattern.get("successful_tags", []),
                "embedding": pattern.get("successful_meme_embedding", []),
                "meme_ids": pattern.get("successful_meme_ids", []),
                "score": sim * conf
            })
    
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:limit]


def find_global_patterns(context_embedding: List[float], limit: int = 5) -> List[Dict]:
    """Find global patterns from all users that match context"""
    from app.db import db
    
    docs = db.collection("global_context_patterns").stream()
    matches = []
    
    for doc in docs:
        pattern = doc.to_dict()
        pattern_emb = pattern.get("context_embedding", [])
        if not pattern_emb:
            continue
        
        sim = cosine_similarity(context_embedding, pattern_emb)
        
        if sim > 0.5:
            rate = pattern.get("success_rate", 0.5)
            count = pattern.get("total_successes", 0)
            matches.append({
                "similarity": sim,
                "success_rate": rate,
                "sample_size": count,
                "tags": pattern.get("common_tags", []),
                "meme_ids": pattern.get("successful_meme_ids", []),
                "embedding": pattern.get("average_embedding", []),
                "score": sim * rate * min(count / 10, 1.0)
            })
    
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:limit]


def update_personal_pattern(user_id: str, context_embedding: List[float], meme_id: str) -> bool:
    """Update user's pattern after successful meme use"""
    from app.db import get_user, update_user, get_meme
    
    user = get_user(user_id)
    if not user:
        return False
    
    meme = get_meme(meme_id)
    if not meme:
        return False
    
    entry = {
        "context_embedding": context_embedding,
        "successful_tags": meme.get("all_tags", []),
        "successful_meme_embedding": meme.get("clip_embedding", []),
        "successful_meme_ids": [meme_id],
        "action": "sent",
        "timestamp": datetime.now(),
        "confidence": 1.0
    }
    
    # Check for similar existing pattern
    for existing in user.context_patterns:
        existing_emb = existing.get("context_embedding", [])
        if not existing_emb:
            continue
        
        sim = cosine_similarity(context_embedding, existing_emb)
        
        if sim > 0.85:
            # Merge into existing
            existing["confidence"] = existing.get("confidence", 1.0) + 0.5
            existing["successful_tags"] = list(set(
                existing.get("successful_tags", []) + entry["successful_tags"]
            ))
            meme_ids = existing.get("successful_meme_ids", [])
            if meme_id not in meme_ids:
                meme_ids.append(meme_id)
                existing["successful_meme_ids"] = meme_ids[-50:]
            existing["timestamp"] = datetime.now()
            update_user(user)
            return True
    
    # New pattern
    user.context_patterns.append(entry)
    
    # Keep last 100
    if len(user.context_patterns) > 100:
        user.context_patterns = sorted(
            user.context_patterns,
            key=lambda x: x.get("confidence", 0),
            reverse=True
        )[:100]
    
    update_user(user)
    return True


def update_global_patterns(context_embedding: List[float], meme_id: str) -> bool:
    """Update global patterns after successful meme use"""
    from app.db import db, get_meme, GlobalContextPattern
    
    meme = get_meme(meme_id)
    if not meme:
        return False
    
    pattern_ref = db.collection("global_context_patterns")
    docs = pattern_ref.stream()
    
    for doc in docs:
        pattern = doc.to_dict()
        pattern_emb = pattern.get("context_embedding", [])
        if not pattern_emb:
            continue
        
        sim = cosine_similarity(context_embedding, pattern_emb)
        
        if sim > 0.80:
            # Update existing
            pattern["total_successes"] = pattern.get("total_successes", 0) + 1
            
            meme_ids = pattern.get("successful_meme_ids", [])
            if meme_id not in meme_ids:
                meme_ids.append(meme_id)
                pattern["successful_meme_ids"] = meme_ids[-30:]
            
            pattern["common_tags"] = list(set(
                pattern.get("common_tags", []) + meme.get("all_tags", [])
            ))
            
            pattern["total_attempts"] = pattern.get("total_attempts", 0) + 1
            pattern["success_rate"] = pattern["total_successes"] / pattern["total_attempts"]
            pattern["last_updated"] = datetime.now()
            
            pattern_ref.document(doc.id).set(pattern)
            return True
    
    # Create new pattern
    new_pattern = GlobalContextPattern(
        context_text="[embedding-based]",
        context_embedding=context_embedding,
        successful_meme_ids=[meme_id],
        common_tags=meme.get("all_tags", []),
        average_embedding=meme.get("clip_embedding", []),
        total_successes=1,
        total_attempts=1,
        success_rate=1.0,
        last_updated=datetime.now()
    )
    
    pattern_ref.add(new_pattern.to_dict())
    return True


def get_context_recommendations(user_id: str, context: str, limit: int = 5) -> Dict:
    """
    Get pattern-based recommendations for context.
    Blends personal and global based on user's history.
    """
    from app.db import get_user
    
    user = get_user(user_id)
    if not user:
        raise ValueError(f"User {user_id} not found")
    
    context_emb = get_context_embedding(context)
    
    # Determine weights based on interaction count
    count = user.context_interaction_count
    if count < 10:
        phase = "new"
        global_w, personal_w = 0.8, 0.2
    elif count < 30:
        phase = "learning"
        global_w, personal_w = 0.5, 0.5
    else:
        phase = "established"
        global_w, personal_w = 0.2, 0.8
    
    global_patterns = find_global_patterns(context_emb, limit)
    personal_patterns = find_personal_patterns(user_id, context_emb, limit)
    
    return {
        "global_patterns": global_patterns,
        "personal_patterns": personal_patterns,
        "global_weight": global_w,
        "personal_weight": personal_w,
        "phase": phase,
        "context_embedding": context_emb
    }


def record_context_interaction(user_id: str, context: str, meme: Dict, action: str):
    """Record interaction for both personal and global learning"""
    context_emb = get_context_embedding(context)
    meme_id = meme.get("id", "")
    
    if action in ["sent", "favorited"]:
        update_personal_pattern(user_id, context_emb, meme_id)
        update_global_patterns(context_emb, meme_id)
    
    # Increment interaction count
    from app.db import get_user, update_user
    user = get_user(user_id)
    if user:
        user.context_interaction_count += 1
        update_user(user)


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("Testing Transformer")
    
    # Test embeddings
    emb1 = get_text_embedding("When your boss makes you redo work")
    emb2 = get_text_embedding("When your manager asks you to redo something")
    emb3 = get_text_embedding("Cute cat sleeping on couch")
    
    print(f"Similar contexts: {cosine_similarity(emb1, emb2):.3f}")
    print(f"Different contexts: {cosine_similarity(emb1, emb3):.3f}")
