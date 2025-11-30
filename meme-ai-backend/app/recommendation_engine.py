"""
Recommendation Engine - Complete meme recommendation system
Handles context patterns, Thompson Sampling, exploration, and diversity
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import random
from datetime import datetime, timedelta
from ml_model import extract_context_tags  # For fallback tag matching
from db import db, get_user, update_user, get_meme, Meme
from context_analyzer import (
    get_context_embedding,
    cosine_similarity,
    find_personal_patterns,
    find_global_patterns
)
from collections import Counter

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

recommendation_sessions = {}


#====================== Similarity calculations ======================
#compares memes to text fallback matching
def calculate_jaccard_similarity_with_context(
    context_embedding: List[float],
    meme_embedding: List[float],
    context_tags: List[str],
    meme_tags: List[str]) -> float:
    """
    Calculate Jaccard-based similarity between context and meme
    Used for Tier 1 (high quality) matching
    
    Formula: (tag_jaccard × 0.6) + (clip_similarity × 0.4)
    Threshold: >= 0.5 for Tier 1
    """
    #look for tag similarity
    tags_context = set(context_tags)
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
    context_tags: List[str],
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
    target_meme: Dict,
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


#====================== Scoring MULTIPLIERS  ======================
#factors in global meme popularity and engagement to the scoring system
def calculate_scoring_multipliers(meme: Dict) -> float:
    """
    Calculate quality score using Bayesian average
    Prevents new memes with 1 like from beating memes with 500 likes
    
    Formula: (likes + prior) / (total_votes + prior_total) × confidence
    """
    #base meme stats
    likes = meme.get('total_thumbs_up', 0)
    dislikes = meme.get('total_thumbs_down', 0)
    total_votes = likes + dislikes

    # get baysian prior assuming 5 likes and 1 dislike to start
    prior = 5.0
    prior_total = 6.0

    #create adjusted score
    adjusted_score = (likes + prior) / (total_votes + prior_total)

    #add confidence boost for popular memes
    confidence = min(total_votes/50,1.0)

    #final score
    multiplier = adjusted_score * (0.65 + 0.35 * confidence)
    return multiplier

#boosts newer memes 
def calculate_freshness_multiplier(meme: Dict) -> float:
    """
    Boost newer memes to keep recommendations feeling fresh
    
    < 7 days old:  1.25x boost (25% increase)
    < 30 days old: 1.1x boost (10% increase)
    Older:         1.0x (no boost)
    """
    #get created at
    created_at = meme.get('created_at')
    if not created_at:
        return 1.0
    
    #get age in days
    age_days = (datetime.now() - created_at).days
    if age_days < 7:
        return 1.25
    elif age_days < 30:
        return 1.1
    else:
        return 1.0


def calculate_recency_penalty(meme: Dict, user: Dict) -> float:
    """
    Penalize memes shown recently using exponential decay
    Avoids showing the same meme too often
    
    Today (0 days):    0.0x (hidden completely)
    1 day ago:         0.29x (71% penalty)
    2 days ago:        0.5x (50% penalty)
    4 days ago:        0.75x(25% penalty)
    7 days ago:        0.91x (9% penalty)
    14+ days ago:      0.99x (almost no penalty)
    Never shown:       1.0x (no penalty)
    """
    #check users last sent memes
    recently_sent = user.get("recently_shown_memes", [])

    #get meme id
    meme_id = meme.get("id")

    #check if meme has been shown recently and when
    for memes in recently_sent:
        if memes.get("meme_id") == meme_id:
            sent_time = memes.get("shown_at")

            if isinstance(sent_time, datetime):
                days_since = (datetime.now() - sent_time).days
            else:
                continue

            #calculate penalty
            penalty = 1.0 - (0.5 ** (days_since / 2))            
            return penalty
    return 1.0


def calculate_favorite_boost(meme: Dict, user: Dict, context_similarity: float) -> float:
    """
    Boost favorited memes based on context relevance
    Re-recommends user's favorites when they fit the situation
    
    Favorited + high relevance (>0.70):    1.5x boost (50% increase)
    Favorited + medium relevance (>0.50):  1.25x boost (25% increase)
    Favorited + low relevance (>0.30):     1.1x boost (10% increase)
    Favorited but not relevant (<0.30):    1.0x (no boost)
    Not favorited:                         1.0x (no boost)
    """
    #get meme id
    meme_id = meme.get("id")
    
    #check if user favorited this meme
    favorited_ids = user.get("favorited_meme_ids", [])
    
    #apply boost if meme is favorited
    if meme_id in favorited_ids:
        #only boost if context is highly relevant
        if context_similarity > 0.70:
            return 1.5  
        elif context_similarity > 0.50:
            return 1.25
        elif context_similarity > 0.30:
            return 1.1
        else:
            return 1.0
    #not favorited
    return 1.0  
    


#====================== Find similar memes ======================
#finds meme to use for context clustering
def weighted_selection_from_tags(recent_meme_ids: List[str], exclude_ids: List[str] = None) -> Dict:
    """
    Pick one representative meme from pattern using frequency + tag clustering
    
    1. Count frequency of each meme_id (how often user picked it)
    2. Get unique memes only (remove duplicates)
    3. Collect tags from unique memes (fair tag pool)
    4. Score each unique meme by tag match
    5. Take top 50% of unique memes
    6. Weighted random pick (frequency × tag_score)
    
    Args:
        recent_meme_ids: List of meme IDs from pattern (last 50 for personal, 200 for global)
        exclude_ids: Meme IDs to skip (already shown)
    
    Returns:
        Selected meme dict, or None if no matches
    """
    # Safety check
    if exclude_ids is None:
        exclude_ids = []
    
    #Count frequency of each meme_id
    meme_frequency = Counter(recent_meme_ids)
    # Example: {"meme_568": 20, "meme_571": 15, "meme_590": 10, ...}
    
    #Get unique meme_ids
    unique_meme_ids = list(meme_frequency.keys())
    # Example: ["meme_568", "meme_571", "meme_590", ...] (23 unique)
    
    if not unique_meme_ids:
        return None
    
    #Get full meme data for unique memes only
    unique_memes = []
    for meme_id in unique_meme_ids:
        # Skip memes already shown in this session
        if meme_id in exclude_ids:
            continue
        
        meme = get_meme(meme_id)
        if meme:
            unique_memes.append({
                "meme": meme,
                "meme_id": meme_id,
                "frequency": meme_frequency[meme_id]
            })
    
    if not unique_memes:
        return None
    
    #Collect tags from unique memes (each meme contributes only once)
    all_tags = []
    for meme_data in unique_memes:
        all_tags.extend(meme_data["meme"].get("all_tags", []))
    
    tag_counts = Counter(all_tags)
    
    # Get top 15 most common tags
    top_tags = [tag for tag, count in tag_counts.most_common(15)]
    
    if not top_tags:
        return None
    
    #Score each unique meme by how well it matches top tags
    top_tags_set = set(top_tags)
    
    for meme_data in unique_memes:
        meme_tags = set(meme_data["meme"].get("all_tags", []))
        overlap_count = len(meme_tags & top_tags_set)
        
        # Tag match score (coverage of top tags)
        tag_score = overlap_count / len(top_tags) if top_tags else 0.0
        
        # Store tag score
        meme_data["tag_score"] = tag_score
        meme_data["overlap"] = overlap_count
    
    #Sort by tag score, take top 50%
    unique_memes.sort(key=lambda x: x["tag_score"], reverse=True)
    
    top_50_percent = unique_memes[:max(1, len(unique_memes) // 2)]
    
    # Weighted random selection (frequency × tag_score)
    weights = []
    for meme_data in top_50_percent:
        # Combine frequency (how often picked) and tag_score (how representative)
        weight = meme_data["frequency"] * meme_data["tag_score"]
        weights.append(weight)
    
    # Select using weighted random
    selected = random.choices(top_50_percent, weights=weights, k=1)[0]
    
    return selected["meme"]


#primary method of suggesting memes
def find_similar_memes(
    reference_meme: Dict,
    exclude_ids: List[str] = None,
    limit: int = 50
) -> List[Dict]:
    """
    Find memes similar to a reference meme using two-tier similarity
    
    Uses calculate_meme_to_meme_similarity with:
    - Tier 1 (Jaccard): >= 0.5 score (high quality matches) + 15% boost
    - Tier 2 (Simple): >= 0.5 score (safety net matches) + no boost
    
    Args:
        reference_meme: The meme to find similar memes to
        exclude_ids: Meme IDs to skip (already shown, or the reference itself)
        limit: Max number of similar memes to return
    
    Returns:
        List of similar memes with their similarity data, sorted by adjusted score
    """
    # Safety check
    if exclude_ids is None:
        exclude_ids = []
    
    # Add reference meme to exclude list (don't return itself)
    reference_id = reference_meme.get("id")
    if reference_id:
        exclude_ids = exclude_ids + [reference_id]
    
    # Query all approved memes from database
    memes_ref = db.collection('memes').where('status', '==', 'approved').stream()
    
    similar_memes = []
    
    for doc in memes_ref:
        meme_data = doc.to_dict()
        meme_data['id'] = doc.id
        
        # Skip excluded memes
        if meme_data['id'] in exclude_ids:
            continue
        
        # Calculate two-tier similarity
        similarity_result = calculate_meme_to_meme_similarity(
            target_meme=reference_meme,
            candidate_meme=meme_data
        )
        
        # Only keep memes that pass either tier threshold
        if similarity_result['tier'] > 0:  
            base_score = similarity_result['final_score']
            
            # Apply tier boost to prioritize high-quality matches
            if similarity_result['tier'] == 1:
                # Tier 1 (Jaccard): High-quality match, apply 15% boost
                adjusted_score = base_score * 1.15
            else:
                # Tier 2 (Simple): Safety net, no boost
                adjusted_score = base_score
            
            similar_memes.append({
                'meme': meme_data,
                'similarity_score': adjusted_score,  # Used for sorting
                'base_score': base_score,  # Original score for reference
                'tier': similarity_result['tier'],
                'jaccard_score': similarity_result['jaccard_score'],
                'simple_score': similarity_result['simple_score']
            })
    
    # Sort by adjusted similarity score (Tier 1 memes naturally rank higher)
    similar_memes.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Return top matches up to limit
    return similar_memes[:limit]


#fallback method of scoring memes 
def score_memes_against_context(
    memes: List[Dict],
    context_embedding: List[float],
    context_tags: List[str],
    exclude_ids: List[str] = None,
    required_tags: List[str] = None,
    apply_freshness: bool = True
) -> List[Dict]:
    """
    REUSABLE scoring function for any list of memes against a context
    
    Used by:
    - get_popular_fallback_memes() - Score top 150 popular memes
    - get_exploration_meme() - Score top 10 new memes
    - (Future) Any other context-based scoring needs
    
    Process:
    1. Filter by required tags (strict)
    2. Calculate Jaccard + Simple similarity
    3. Apply 3-tier system (0.5, 0.5, 0.35 thresholds)
    4. Calculate quality multiplier
    5. Optionally apply freshness multiplier
    6. Sort by final score
    
    Args:
        memes: List of meme dicts to score (already fetched from DB)
        context_embedding: CLIP embedding of user's context
        context_tags: Extracted tags from user's context
        exclude_ids: Meme IDs to skip (already shown)
        required_tags: Tags that MUST ALL be present (strict)
        apply_freshness: Whether to include freshness boost (default True)
    
    Returns:
        List of scored memes, sorted by final_score (best first)
    """
    # Safety checks
    if exclude_ids is None:
        exclude_ids = []
    if required_tags is None:
        required_tags = []
    
    # Convert required_tags to lowercase set
    required_tags_set = set(tag.lower() for tag in required_tags)
    
    scored_candidates = []
    
    for meme_data in memes:
        # Skip excluded memes
        if meme_data.get('id') in exclude_ids:
            continue
        
        # Get meme's data
        meme_embedding = meme_data.get('clip_embedding', [])
        meme_tags = meme_data.get('all_tags', [])
        
        # Skip if missing critical data
        if not meme_embedding or not meme_tags:
            continue
        
        # STRICT required tags filter
        if required_tags_set:
            meme_tags_lower = set(tag.lower() for tag in meme_tags)
            if not required_tags_set.issubset(meme_tags_lower):
                continue
        
        # Calculate similarities
        jaccard_score = calculate_jaccard_similarity_with_context(
            context_embedding=context_embedding,
            meme_embedding=meme_embedding,
            context_tags=context_tags,
            meme_tags=meme_tags
        )
        
        simple_score = simple_similarity_with_context(
            context_embedding=context_embedding,
            meme_embedding=meme_embedding,
            context_tags=context_tags,
            meme_tags=meme_tags
        )
        
        # 3-Tier System
        if jaccard_score >= 0.5:
            tier = 1
            similarity_score = jaccard_score * 1.15
        elif simple_score >= 0.5:
            tier = 2
            similarity_score = simple_score
        elif jaccard_score >= 0.35 or simple_score >= 0.35:
            tier = 3
            best_score = max(jaccard_score, simple_score)
            similarity_score = best_score * 0.85
        else:
            continue  # Too low, skip
        
        # Calculate multipliers
        quality = calculate_scoring_multipliers(meme_data)
        
        # Optionally apply freshness
        if apply_freshness:
            freshness = calculate_freshness_multiplier(meme_data)
            final_score = similarity_score * quality * freshness
        else:
            freshness = 1.0
            final_score = similarity_score * quality
        
        scored_candidates.append({
            'meme': meme_data,
            'similarity_score': similarity_score,
            'quality_score': quality,
            'freshness_score': freshness,
            'final_score': final_score,
            'tier': tier,
            'jaccard_score': jaccard_score,
            'simple_score': simple_score
        })
    
    # Sort by final score
    scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
    
    return scored_candidates


#use fallback method on popular memes
def get_popular_fallback_memes(
    context_embedding: List[float],
    context_tags: List[str],
    exclude_ids: List[str] = None,
    required_tags: List[str] = None,
    check_limit: int = 150,
    return_limit: int = 100
) -> List[Dict]:
    """
    Get contextually relevant popular memes as fallback
    Uses the reusable score_memes_against_context() function
    """
    # Safety checks
    if exclude_ids is None:
        exclude_ids = []
    
    # Query top popular memes from database
    memes_ref = (
        db.collection('memes')
        .where('status', '==', 'approved')
        .order_by('total_sends', direction='DESCENDING')
        .limit(check_limit)
        .stream()
    )
    
    # Convert to list of dicts
    popular_memes = []
    for doc in memes_ref:
        meme_data = doc.to_dict()
        meme_data['id'] = doc.id
        popular_memes.append(meme_data)
    
    # Score using reusable function
    scored_memes = score_memes_against_context(
        memes=popular_memes,
        context_embedding=context_embedding,
        context_tags=context_tags,
        exclude_ids=exclude_ids,
        required_tags=required_tags,
        apply_freshness=True  # Popular fallback considers freshness
    )
    
    # Return top matches
    return scored_memes[:return_limit]


# use to include diverse memes in the recommendation system
def get_exploration_meme(
    user_id: str,
    context_embedding: List[float],
    context_tags: List[str],
    exclude_ids: List[str] = None,
    required_tags: List[str] = None,
    pool_size: int = 100,      # ← NEW: Fetch 100 memes
    candidate_limit: int = 10  # ← Keep: Score 10 random memes
) -> Optional[Dict]:
    """
    Get a contextually relevant NEW meme for exploration (every 3-5 queries)
    
    Strategy:
    1. Fetch 100 newest memes (< 7 days old)
    2. Randomly pick 10 from the 100
    3. Score the 10 against context
    4. Return the best one
    
    Returns None if:
    - No new memes exist
    - required_tags filters out all 10
    - None pass tier threshold
    
    Args:
        user_id: User ID (check what they've seen)
        context_embedding: CLIP embedding of context
        context_tags: Tags from context
        exclude_ids: Memes to skip (current session)
        required_tags: Tags that must be present (strict)
        pool_size: How many new memes to fetch (default 100)
        candidate_limit: How many to score from pool (default 10)
    """
    # Safety check
    if exclude_ids is None:
        exclude_ids = []
    
    # Get user's already-seen memes
    user = get_user(user_id)
    if user:
        user_shown_ids = [entry.get('meme_id') for entry in user.get('recently_shown_memes', [])]
        exclude_ids = exclude_ids + user_shown_ids
    
    # Calculate cutoff date (7 days ago)
    seven_days_ago = datetime.now() - timedelta(days=7)
    
    # Query 100 newest memes (< 7 days old) ← CHANGED: Get 100 instead of 10
    memes_ref = (
        db.collection('memes')
        .where('status', '==', 'approved')
        .where('created_at', '>=', seven_days_ago)
        .order_by('created_at', direction='DESCENDING')
        .limit(pool_size)  # ← CHANGED: Get 100 newest
        .stream()
    )
    
    # Convert to list
    new_memes_pool = []  # ← CHANGED: Renamed to indicate it's a pool
    for doc in memes_ref:
        meme_data = doc.to_dict()
        meme_data['id'] = doc.id
        new_memes_pool.append(meme_data)
    
    # If no new memes found at all, return None
    if not new_memes_pool:
        return None
    
    # *** NEW: Randomly select 10 from the pool ***
    # If pool has fewer than 10, use all of them
    sample_size = min(candidate_limit, len(new_memes_pool))
    new_memes_sample = random.sample(new_memes_pool, sample_size)
    
    # Score the random sample using reusable function
    scored_new_memes = score_memes_against_context(
        memes=new_memes_sample,  # ← CHANGED: Score the random 10
        context_embedding=context_embedding,
        context_tags=context_tags,
        exclude_ids=exclude_ids,
        required_tags=required_tags,
        apply_freshness=False  # Don't apply freshness (all new already)
    )
    
    # If no memes passed filters/thresholds, return None
    if not scored_new_memes:
        return None
    
    # Return top-ranked new meme (best contextual fit from random sample)
    return scored_new_memes[0]


#makes sure the seleted memes are diverse
def apply_diversity_filter(
    candidates: List[Dict],
    diversity_threshold: float = 0.85,
    preserve_exploration: bool = True
) -> List[Dict]:
    """
    Filter candidates to ensure visual diversity
    Prevents showing multiple variations of the same meme format
    
    Process:
    1. Start with empty diverse pool
    2. For each candidate, check CLIP similarity to already-selected memes
    3. If too similar (>= 0.85), skip it
    4. If diverse enough, add to pool
    
    Args:
        candidates: Thompson-sampled candidates (sorted by score)
        diversity_threshold: CLIP similarity cutoff (default 0.85)
        preserve_exploration: Keep exploration memes regardless (default True)
    
    Returns:
        Diverse subset of candidates
    """
    # If not enough candidates, return all
    if len(candidates) <= 1:
        return candidates
    
    diverse_pool = []
    
    for candidate in candidates:
        # Check if this is an exploration meme (if tracking that)
        is_exploration = candidate.get('is_exploration', False)
        
        # Always preserve exploration memes (new content injection)
        if preserve_exploration and is_exploration:
            diverse_pool.append(candidate)
            continue
        
        # Check diversity against already-selected memes
        is_diverse = True
        candidate_embedding = candidate['meme'].get('clip_embedding', [])
        
        # Skip if no embedding
        if not candidate_embedding:
            continue
        
        for selected in diverse_pool:
            selected_embedding = selected['meme'].get('clip_embedding', [])
            
            # Skip if selected meme has no embedding
            if not selected_embedding:
                continue
            
            # Calculate CLIP similarity
            similarity = cosine_similarity(candidate_embedding, selected_embedding)
            
            # If too similar to an already-selected meme, mark as not diverse
            if similarity >= diversity_threshold:
                is_diverse = False
                break
        
        # Add if diverse enough
        if is_diverse:
            diverse_pool.append(candidate)
    
    return diverse_pool


#====================== ZERO-SHOT SEMANTIC SEARCH ======================
#failssafe foe cold starting 
def get_semantic_search_candidates(
    context_embedding: List[float],
    exclude_ids: List[str],
    limit: int = 30
) -> List[Dict]:
    """
    ZERO-SHOT RETRIEVAL: Direct text-to-image matching
    
    Finds memes that visually/conceptually match the text description directly.
    Crucial for unique, new user queries that have no historical patterns.
    
    How it works:
    1. Get ALL approved memes from database
    2. Compare context embedding directly to each meme's CLIP image embedding
    3. Return top matches (no tags, no history - pure semantic similarity)
    
    Args:
        context_embedding: CLIP embedding of user's text context
        exclude_ids: Meme IDs to skip (already shown)
        limit: How many candidates to return (default 30)
    
    Returns:
        List of candidate dicts with semantic similarity scores
    """
    # Get ALL approved memes (fast for <10k memes, use vector DB for larger scale)
    memes_ref = db.collection('memes').where('status', '==', 'approved').stream()
    
    candidates = []
    
    for doc in memes_ref:
        meme = doc.to_dict()
        meme['id'] = doc.id
        
        # Skip already shown memes
        if meme['id'] in exclude_ids:
            continue
        
        # Skip memes without embeddings
        if not meme.get('clip_embedding'):
            continue
        
        # Direct cosine similarity (text embedding vs image embedding)
        # This relies purely on CLIP's ability to match text descriptions to images
        similarity = cosine_similarity(
            context_embedding,
            meme.get('clip_embedding', [])
        )
        
        # Threshold: Keep lower for semantic search (0.23-0.30)
        # CLIP text-to-image similarities are typically lower than image-to-image
        if similarity > 0.23:
            # Apply quality multiplier (still favor high-quality memes)
            quality_mult = calculate_quality_multiplier(meme)
            
            # Calculate freshness boost (favor newer memes slightly)
            days_old = (datetime.now() - meme.get('created_at', datetime.now())).days
            freshness_boost = max(0.8, 1.0 - (days_old / 365) * 0.2)  # Gentle decay over a year
            
            # Final score = semantic similarity * quality * freshness
            final_score = similarity * quality_mult * freshness_boost
            
            candidates.append({
                'meme': meme,
                'final_score': final_score,
                'tier': 'semantic_search',
                'similarity_score': similarity,
                'quality_score': quality_mult,
                'is_semantic': True
            })
    
    # Sort by final score (best matches first)
    candidates.sort(key=lambda x: x['final_score'], reverse=True)
    
    return candidates[:limit]

#====================== Thompson Sampling ======================

def thompson_sampling_selection(
    candidates: List[Dict],
    num_selections: int = 10
) -> List[Dict]:
    """
    Select memes using Thompson Sampling to balance proven winners and new memes
    
    Process:
    1. Sample success rate from Beta(thumbs_up + 1, thumbs_down + 1)
    2. Multiply by context score
    3. Return top N
    
    Why this works:
    - New memes: High uncertainty → sometimes rank high (exploration)
    - Proven memes: Low uncertainty → consistently rank high (exploitation)
    - Bad memes: Low sample → filtered out
    
    Args:
        candidates: Scored candidate memes
        num_selections: How many to pick (default 10)
    
    Returns:
        Top memes sorted by combined score (context × thompson sample)
    """
    #score results
    thompson_scores = []
    
    #loop through candidates
    for candidate in candidates:
        meme = candidate['meme']
        
        # Get engagement stats
        thumbs_up = meme.get('total_thumbs_up', 0)
        thumbs_down = meme.get('total_thumbs_down', 0)
        
        # Beta distribution parameters (add 1 to avoid 0)
        alpha = thumbs_up + 1  # successes
        beta = thumbs_down + 1  # failures
        
        # Sample from Beta distribution
        # This represents our "belief" about the true success rate
        #basicly confidence in the each thompson score
        sampled_rate = np.random.beta(alpha, beta)
        
        #combine with context similarity score(multiply them)(60/40 split)
        combined_score = (candidate['final_score'] ** 0.6) * (sampled_rate ** 0.4)
        
        thompson_scores.append({
            'meme': meme,
            'original_score': candidate['final_score'],
            'thompson_sample': sampled_rate,
            'combined_score': combined_score,
            'tier': candidate.get('tier', 0),
            'similarity_score': candidate.get('similarity_score', 0),
            'quality_score': candidate.get('quality_score', 0)
        })
    
    # Sort by combined score (context fit × predicted success)
    thompson_scores.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Return top N
    return thompson_scores[:num_selections]


#====================== MAIN RECOMMENDATION ENGINE ======================
#extra diversity filter to help new users
def get_curated_mix(
    user_id: str, 
    context_embedding: List[float], 
    exclude_ids: List[str],
    num_personal: int = 2, 
    num_global: int = 1
) -> List[Dict]:
    """
    Get a mix of personal favorites and global trending memes
    
    This provides diversity by combining:
    - Top personal favorites (contextually relevant)
    - Top global trending memes (what's working for everyone)
    
    Flow:
    1. Get user's favorited memes
    2. Score them by context similarity
    3. Get global top memes
    4. Score them by context similarity
    5. Return curated list (no duplicates)
    
    Args:
        user_id: User's unique ID
        context_embedding: CLIP embedding of user's context
        exclude_ids: Meme IDs to exclude (already shown)
        num_personal: How many personal favorites to return (default 2)
        num_global: How many global trending to return (default 1)
    
    Returns:
        List of candidate dicts with meme data and scores
    """
    curated = []
    
    # Get personal favorites
    user = get_user(user_id)
    if not user:
        return []
    
    user_dict = user if isinstance(user, dict) else user.to_dict()
    favorite_ids = user_dict.get('favorited_meme_ids', [])
    
    if favorite_ids:
        # Get memes and score them by context similarity
        scored_favorites = []
        for meme_id in favorite_ids[:20]:  # Check last 20 favorites
            if meme_id in exclude_ids:
                continue
                
            meme = get_meme(meme_id)
            if meme and meme.get('status') == 'approved':
                # Score by context match
                meme_dict = meme if isinstance(meme, dict) else meme.to_dict()
                meme_dict['id'] = meme_id
                
                similarity = cosine_similarity(
                    context_embedding,
                    meme_dict.get('clip_embedding', [])
                )
                
                scored_favorites.append({
                    'meme': meme_dict,
                    'final_score': similarity,
                    'tier': 'personal_curated',
                    'similarity_score': similarity,
                    'quality_score': 0,
                    'is_curated': True
                })
        
        # Sort by score, take top N
        scored_favorites.sort(key=lambda x: x['final_score'], reverse=True)
        curated.extend(scored_favorites[:num_personal])
    
    # Get global trending memes
    memes_ref = db.collection('memes')
    global_candidates = (
        memes_ref
        .where('status', '==', 'approved')
        .order_by('upvotes', direction=firestore.Query.DESCENDING)
        .limit(30)  # Check top 30 popular
        .stream()
    )
    
    scored_global = []
    for doc in global_candidates:
        meme = doc.to_dict()
        meme['id'] = doc.id
        
        # Skip if already in favorites or excluded
        if meme['id'] in [c['meme']['id'] for c in curated] or meme['id'] in exclude_ids:
            continue
        
        # Score by context match
        similarity = cosine_similarity(
            context_embedding,
            meme.get('clip_embedding', [])
        )
        
        scored_global.append({
            'meme': meme,
            'final_score': similarity,
            'tier': 'global_curated',
            'similarity_score': similarity,
            'quality_score': 0,
            'is_curated': True
        })
    
    # Sort by score, take top N
    scored_global.sort(key=lambda x: x['final_score'], reverse=True)
    curated.extend(scored_global[:num_global])
    
    return curated



# we did it!!!
def get_recommendations(
    user_id: str,
    context: str,
    session_id: Optional[str] = None,
    required_tags: Optional[List[str]] = None,
    batch_size: int = 3
) -> Dict:
    """
    Main recommendation engine - orchestrates entire recommendation flow
    
    Flow:
    1. Get/create session
    2. Extract context embedding and tags
    3. Check exploration trigger (only on first request of new session)
    4. Check curated mix injection (ONLY if no exploration)
    5. Get candidates from personal patterns → global patterns → semantic search → popular fallback
    6. Apply Thompson Sampling (balances quality and exploration)
    7. Apply diversity filter (prevents repetitive formats)
    8. Return final batch
    
    Args:
        user_id: User's unique ID
        context: User's text context (e.g., "stressed about work deadlines")
        session_id: Optional session ID (creates new if None)
        required_tags: Optional tags that must be present in all memes
        batch_size: How many memes to return (default 3)
    
    Returns:
        Dict with memes, session_id, and metadata
    """
    # Step 1: Get or create session
    if session_id and session_id in recommendation_sessions:
        session = recommendation_sessions[session_id]
        # Update activity timestamp
        session.last_activity_at = datetime.now()
    else:
        # Create new session
        session_id = f"{user_id}_{datetime.now().timestamp()}"
        
        # Get context embedding and tags
        context_embedding = get_context_embedding(context)
        context_tags = extract_context_tags(context)
        
        session = RecommendationSession(
            session_id=session_id,
            user_id=user_id,
            context=context,
            context_embedding=context_embedding
        )
        recommendation_sessions[session_id] = session
    
    # Get user data
    user = get_user(user_id)
    if not user:
        return {
            'error': 'User not found',
            'session_id': session_id
        }
    
    # Extract context data (use session's stored data)
    context_embedding = session.context_embedding
    context_tags = extract_context_tags(context)
    
    # Step 2: Check exploration trigger (only on first request of new session)
    exploration_candidates = []
    user_dict = user if isinstance(user, dict) else user.to_dict()
    
    # Determine if this is the FIRST request of a NEW session
    # True if: session has no shown memes yet (brand new)
    # False if: session already has shown memes ("show more" request)
    is_first_request = len(session.shown_meme_ids) == 0
    
    # Only check exploration on the FIRST request of each session
    if is_first_request:
        # Get current counters
        query_count = user_dict.get('exploration_query_count', 0)
        next_exploration = user_dict.get('next_exploration_at', 3)
        
        # Check if exploration should trigger
        if query_count >= next_exploration:
            # Try to get exploration meme
            exploration_meme = get_exploration_meme(
                user_id=user_id,
                context_embedding=context_embedding,
                context_tags=context_tags,
                exclude_ids=session.shown_meme_ids,
                required_tags=required_tags
            )
            
            if exploration_meme:
                # Mark as exploration meme
                exploration_candidates = [{
                    'meme': exploration_meme['meme'],
                    'final_score': exploration_meme['final_score'],
                    'tier': exploration_meme['tier'],
                    'similarity_score': exploration_meme.get('similarity_score', 0),
                    'quality_score': exploration_meme.get('quality_score', 0),
                    'is_exploration': True
                }]
                
                # Reset counter after successful exploration
                query_count = 0
                next_exploration = random.randint(3, 5)
            else:
                # Exploration failed, still reset and try again later
                query_count = 0
                next_exploration = random.randint(3, 5)
        else:
            # Not time yet, increment counter for next session
            query_count += 1
        
        # Save to user DB (persists across sessions)
        update_user(user_id, {
            'exploration_query_count': query_count,
            'next_exploration_at': next_exploration
        })
    
    # Step 3: Inject curated mix (ONLY if no exploration happened)
    curated_injection = []
    
    # Inject curated memes ONLY if:
    # - No exploration happened this round (mutually exclusive)
    # This ensures every first request has EITHER exploration OR curated diversity
    if len(exploration_candidates) == 0:
        curated_injection = get_curated_mix(
            user_id=user_id,
            context_embedding=context_embedding,
            exclude_ids=session.shown_meme_ids,
            num_personal=2,
            num_global=1
        )
    
    # Step 4: Get candidates from patterns
    all_candidates = []
    
    # 4a. Try personal patterns
    personal_patterns = find_personal_patterns(
        user_id=user_id,
        context_embedding=context_embedding
    )
    
    for pattern in personal_patterns:
        # Pick representative meme from pattern
        reference_meme = weighted_selection_from_tags(
            recent_meme_ids=pattern.get('recent_meme_ids', []),
            exclude_ids=session.shown_meme_ids
        )
        
        if reference_meme:
            # Find similar memes
            similar = find_similar_memes(
                reference_meme=reference_meme,
                exclude_ids=session.shown_meme_ids,
                limit=30
            )
            all_candidates.extend(similar)
    
    # 4b. Try global patterns if not enough candidates
    if len(all_candidates) < 50:
        global_patterns = find_global_patterns(
            context_embedding=context_embedding
        )
        
        for pattern in global_patterns:
            reference_meme = weighted_selection_from_tags(
                recent_meme_ids=pattern.get('recent_meme_ids', []),
                exclude_ids=session.shown_meme_ids
            )
            
            if reference_meme:
                similar = find_similar_memes(
                    reference_meme=reference_meme,
                    exclude_ids=session.shown_meme_ids,
                    limit=30
                )
                all_candidates.extend(similar)
    
    # 4c. Zero-shot semantic search if patterns are weak
    # This handles unique queries that have no historical patterns
    if len(all_candidates) < 30:
        semantic_matches = get_semantic_search_candidates(
            context_embedding=context_embedding,
            exclude_ids=session.shown_meme_ids,
            limit=30
        )
        all_candidates.extend(semantic_matches)
    
    # 4d. Popular fallback if STILL not enough (last resort)
    if len(all_candidates) < 20:
        fallback = get_popular_fallback_memes(
            context_embedding=context_embedding,
            context_tags=context_tags,
            exclude_ids=session.shown_meme_ids,
            required_tags=required_tags,
            check_limit=150,
            return_limit=100
        )
        all_candidates.extend(fallback)
    
    # Step 5: Remove duplicates (same meme from different sources)
    unique_candidates = {}
    for candidate in all_candidates:
        meme_id = candidate['meme']['id']
        if meme_id not in unique_candidates:
            unique_candidates[meme_id] = candidate
    
    candidate_list = list(unique_candidates.values())
    
    # Step 6: Apply Thompson Sampling
    if len(candidate_list) > 0:
        thompson_selected = thompson_sampling_selection(
            candidates=candidate_list,
            num_selections=min(30, len(candidate_list))
        )
    else:
        thompson_selected = []
    
    # Step 7: Combine all sources (exploration OR curated + thompson)
    combined_candidates = exploration_candidates + curated_injection + thompson_selected
    
    # Step 8: Apply diversity filter
    diverse_final = apply_diversity_filter(
        candidates=combined_candidates,
        diversity_threshold=0.85,
        preserve_exploration=True
    )
    
    # Step 9: Take batch_size memes
    final_batch = diverse_final[:batch_size]
    
    # Update session with shown memes
    for candidate in final_batch:
        session.shown_meme_ids.append(candidate['meme']['id'])
    
    # Store candidates for progressive consumption
    session.all_candidates = diverse_final
    
    # Step 10: Return results
    return {
        'memes': [candidate['meme'] for candidate in final_batch],
        'session_id': session_id,
        'metadata': {
            'total_candidates': len(candidate_list),
            'after_thompson': len(thompson_selected),
            'after_diversity': len(diverse_final),
            'returned': len(final_batch),
            'has_exploration': len(exploration_candidates) > 0,
            'has_curated': len(curated_injection) > 0,
            'curated_count': len(curated_injection),
            'exploration_query_count': query_count if is_first_request else user_dict.get('exploration_query_count', 0)
        }
    }