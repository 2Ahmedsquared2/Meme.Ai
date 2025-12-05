"""
Context Analyzer - Handles context pattern learning and matching
Supports both personal and global context patterns
"""

from typing import List, Dict
import numpy as np
from datetime import datetime
from transformers import CLIPProcessor, CLIPModel
import torch
from app.db import db, get_user, update_user, get_meme, GlobalContextPattern
from app.ml_model import load_clip


#use lazy loading for CLIP model
clip_processor = None
clip_model = None
def get_context_embedding(text: str) -> List[float]:
    """
    Get CLIP embedding for a text string
    """
    global clip_processor, clip_model
    
    # Load model if not already loaded
    if clip_model is None:
        clip_processor, clip_model = load_clip()
    
    # Process text
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    
    # Generate embedding
    with torch.no_grad():
        text_embedding = clip_model.get_text_features(**inputs)

    return text_embedding[0].tolist()


def cosine_similarity(vec1:List[float], vec2:List[float]) -> float:
    """
    get the cosine similarity between the 2 vectors 
    AKA compare the similarity of 2 contexts
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.linalg.norm(vec1) ==0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def update_global_patterns(context:str, meme:dict, action:str):
    """
    this update global context pattern
    """
    #get context
    context_embedding = get_context_embedding(context)

    #checks if similar context pattern exists
    pattern_ref = db.collection("global_context_patterns")
    docs = pattern_ref.stream()

    for doc in docs:
        pattern= doc.to_dict()
        similarity = cosine_similarity(context_embedding, pattern["context_embedding"])
        #if similar pattern is found
        if similarity > 0.80:
            if action in ["sent", "favorited"]:
                pattern["total_successes"] += 1
                pattern["successful_meme_ids"].append(meme["id"])
                pattern["common_tags"] = list(set(pattern["common_tags"]+ meme["all_tags"]))
            #update pattern stats
            pattern["total_attempts"] += 1
            pattern["success_rate"] = pattern["total_successes"] / pattern["total_attempts"]
            pattern["last_updated"] = datetime.now()

            #keep to 30 
            pattern["successful_meme_ids"] = pattern["successful_meme_ids"][-30:]

            #update in db
            pattern_ref.document(doc.id).set(pattern)
            return
    
    #if no similar pattern is found, create new one
    if action in ["sent", "favorited"]:
        new_pattern = GlobalContextPattern(
            context_text=context,
            context_embedding=context_embedding,
            successful_meme_ids=[meme["id"]],
            common_tags=meme["all_tags"],
            average_embedding=meme.get("clip_embedding", []),
            total_successes=1,
            total_attempts=1,
            success_rate=1.0,
            last_updated=datetime.now()
        )
        
        # Save to Firestore (THIS creates the data!)
        pattern_ref.add(new_pattern.to_dict())
        return 


def update_personal_pattern(user_id: str, context: str, meme: Dict, action: str):
    """
    Update user's personal context pattern
    """
    user = get_user(user_id)
    context_embedding = get_context_embedding(context)
    
    pattern_entry = {
        "context_text": context,
        "context_embedding": context_embedding,
        "successful_tags": meme["all_tags"],
        "successful_meme_embedding": meme.get("clip_embedding", []),
        "action": action,
        "timestamp": datetime.now(),
        "confidence": 1.0
    }
    
    # Check if similar personal pattern exists
    for existing_pattern in user.context_patterns:
        similarity = cosine_similarity(
            context_embedding,
            existing_pattern["context_embedding"]
        )
        
        if similarity > 0.85:
            # Merge patterns
            existing_pattern["confidence"] += 0.5
            existing_pattern["successful_tags"] = list(set(
                existing_pattern["successful_tags"] + pattern_entry["successful_tags"]
            ))
            existing_pattern["timestamp"] = datetime.now()
            update_user(user)
            return
    
    # New personal pattern
    user.context_patterns.append(pattern_entry)
    
    # Keep only last 50 patterns
    if len(user.context_patterns) > 100:
        user.context_patterns = sorted(
            user.context_patterns,
            key=lambda x: x["confidence"],
            reverse=True
        )[:100]
    
    update_user(user)
            

def find_global_patterns(context_embedding:List[float], limit=5) -> List[Dict]:
    """
    Find global context patterns from all users and what worked
    """
    pattern_ref = db.collection("global_context_patterns").stream()

    matches = []

    for doc in pattern_ref:
        pattern = doc.to_dict()
        similarity = cosine_similarity(context_embedding, pattern["context_embedding"])

        if similarity > 0.5:
            matches.append({
                "similarity": similarity,
                "success_rate": pattern["success_rate"],
                "sample_size": pattern["total_successes"],
                "tags": pattern["common_tags"],
                "meme_ids": pattern["successful_meme_ids"],
                "embedding": pattern["average_embedding"],
                "score": similarity * pattern["success_rate"] * min(pattern["total_successes"]/10, 1.0)
            })
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:limit]


def find_personal_patterns(user_id:str, context_embedding:List[float], limit=5) -> List[Dict]:
    """
    find what worked foe the user
    """
    user = get_user(user_id)
    matches = []
    
    for pattern in user.context_patterns:
        #only look at successful patterns
        if pattern["action"] in ["sent", "favorited"]:
            similarity = cosine_similarity(context_embedding, pattern["context_embedding"])
            if similarity > 0.7:
                matches.append({
                    "similarity": similarity,
                    "confidence": pattern["confidence"],
                    "tags": pattern["successful_tags"],
                    "embedding": pattern["successful_meme_embedding"],
                    "score": similarity * pattern["confidence"]
                })
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:limit]


def get_context_recommendations(user_id:str, context: str, limit=5) -> Dict:
    """
    context matching:
    - New users: Use global patterns
    - Learning users: Use both
    - Established users: Prefer personal patterns
    """
    user = get_user(user_id)
    if not user:
        raise ValueError(f"User {user_id} not found")
    context_embedding = get_context_embedding(context)

    # get interaction count to determine phase of learning
    interaction_count = user.context_interaction_count
    if interaction_count < 10:
        phase = "new"
        global_weight = 0.8
        personal_weight = 0.2
    elif interaction_count <30:
        phase = "learning"
        global_weight = 0.5
        personal_weight = 0.5
    else:
        phase = "established"
        global_weight = 0.2
        personal_weight = 0.8

    print(f"User phase: {phase} ({interaction_count} context interactions)")

    #global context patterns 
    global_patterns = find_global_patterns(context_embedding, limit=limit)

    #personal context patterns 
    personal_patterns = find_personal_patterns(user_id, context_embedding, limit=limit)

    #combined results
    combined_results = {
        "global_patterns": global_patterns,
        "personal_patterns": personal_patterns,
        "global_weight": global_weight,
        "personal_weight": personal_weight,
        "phase": phase
    }
    return combined_results


def record_context_interaction(user_id: str, context: str, meme_sent: Dict, action: str):
    """
    Record interaction for both global and personal patterns
    Call this whenever user interacts with a meme
    """
    # Update personal pattern
    update_personal_pattern(user_id, context, meme_sent, action)
    
    # Update global pattern
    update_global_patterns(context, meme_sent, action)
    
    # Increment user's interaction count
    user = get_user(user_id)
    user.context_interaction_count += 1
    update_user(user)

