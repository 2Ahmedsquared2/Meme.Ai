"""
Recommendation Engine - Complete meme recommendation system
Handles context patterns, Thompson Sampling, exploration, and diversity
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
from ml_model import extract_context_tags  # For fallback tag matching
from db import db, get_user, update_user, get_meme, Meme
from context_analyzer import (
    get_context_embedding,
    cosine_similarity,
    find_personal_patterns,
    find_global_patterns
)

#====================== Session Storage ======================
# Stores user session data between recommendations
# Prevents repetitive recommendations for the same user
# Helps maintain context and user preferences
#=============================================================
@dataclass
class RecommendationSession:
    """ Represents a user session """
    session_id: str
    user_id: str
    context: str
    context_embedding: List[float]
    shown_meme_ids: List[str] = field(default_factory=list)  # ← Safe default
    all_candidates: List[Dict] = field(default_factory=list)  # ← Safe default
    created_at: datetime = field(default_factory=datetime.now)
    last_activity_at: datetime = field(default_factory=datetime.now)


#====================== Similarity calculations ======================
#compares memes to text fallback matching
def calculate_jaccard_similarity_with_context(
    context_embedding: List[float],
    meme_embedding: List[float],
    Context_tags: List[str],
    meme_tags: List[str]) -> float:
    """
    Calculate Jaccard-based similarity between context and meme
    Used for Tier 1 (high quality) matching
    
    Formula: (tag_jaccard × 0.6) + (clip_similarity × 0.4)
    Threshold: >= 0.5 for Tier 1
    """
    #look for tag similarity
    tags_context = set(Context_tags)
    tags_meme = set(meme_tags)
    intersection = tags_context & tags_meme
    union = tags_context | tags_meme
    tag_score = len(intersection) / len(union) if union else 0.0

    #look for clip semantic similarity 
    clip_score = cosine_similarity(context_embedding, meme_embedding)
    #combine scores
    total_score = (tag_score * 0.6) + (clip_score * 0.4)
    return total_score


#compares memes to text fallback matching
def simple_similarity_with_context(
    context_embedding: List[float],
    meme_embedding: List[float],
    Context_tags: List[str],
    meme_tags: List[str]) -> float:
    """
    Calculate simple similarity between context and meme
    Used for Tier 2 (safety net) matching
    
    Formula: (tag_simple × 0.6) + (clip_similarity × 0.4)
    Threshold: >= 0.5 for Tier 2
    """
    #Look for tag similarity 
    tags_context = set(context_tags)
    tags_meme = set(meme_tags)

    intersection = tags_context & tags_meme
    tag_score = len(intersection) / len(tags_context) if tags_context else 0.0
    
    # CLIP semantic similarity
    clip_score = cosine_similarity(context_embedding, meme_embedding)
    
    # Combined (favor tags 60/40)
    final_score = (tag_score * 0.6) + (clip_score * 0.4)
    
    return final_score


#compares memes to memes primary matching
def calculate_meme_to_meme_similarity(
    targert_meme: Dict,
    candidate_meme: Dict) -> float:
    """
    Calculate two-tier similarity between two memes
    Returns both Jaccard and Simple scores with tier classification
    
    Used when finding "similar memes" to a pattern meme
    """
    #get data from memes
    target_tags = set(target_meme.get("all_tags", []))
    candidate_tags = set(candidate_meme.get("all_tags", []))
    target_embedding = target_meme.get('clip_embedding', [])
    candidate_embedding = candidate_meme.get('clip_embedding', [])
    
    # Tag similarities
    intersection = target_tags & candidate_tags
    union = target_tags | candidate_tags
    
    # Jaccard (Tier 1)
    jaccard_tag_score = len(intersection) / len(union) if union else 0.0
    
    # Simple (Tier 2)
    simple_tag_score = len(intersection) / len(target_tags) if target_tags else 0.0
    
    # CLIP similarity
    clip_score = cosine_similarity(target_embedding, candidate_embedding) if target_embedding and candidate_embedding else 0.0
    
    # Combined scores
    jaccard_score = (jaccard_tag_score * 0.6) + (clip_score * 0.4)
    simple_score = (simple_tag_score * 0.6) + (clip_score * 0.4)
    
    # Determine tier
    if jaccard_score >= 0.5:
        tier = 1
        final_score = jaccard_score
    elif simple_score >= 0.5:
        tier = 2
        final_score = simple_score
    else:
        tier = 0  # Not similar enough
        final_score = max(jaccard_score, simple_score)
    
    return {
        'jaccard_score': jaccard_score,
        'simple_score': simple_score,
        'final_score': final_score,
        'tier': tier
    }





