"""
Recommendation Engine V2 - Parallel Scoring System

How it works:
1. Semantic search finds ALL potentially good matches (base similarity)
2. Apply multipliers in parallel:
   - Personal boost: Has user interacted with this meme? (1.0-2.0x)
   - Global boost: Does this meme work well for everyone? (1.0-1.3x)
   - Engagement: User's thumbs up/down/send/favorite (0.5-1.2x)
   - Tag affinity: User likes these tag types (1.0-1.3x)

Final score = semantic_similarity × personal × global × engagement × tag_affinity

All signals work together - best of all worlds!
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
from app.db import db, get_user, get_meme
from app.transformer import cosine_similarity

# Context-independent tags (can boost regardless of context)
CONTEXT_INDEPENDENT_TAGS = [
    "quote", "sports", "wholesome", "dark", "ironic", "sarcastic",
    "reddit", "twitter", "classic", "movies", "tv_shows", "anime",
    "celebrities", "animals", "objects", "text_heavy", "screenshot"
]


def _safe_datetime_diff_days(dt1: datetime, dt2: datetime) -> int:
    """Safely compute difference in days between two datetimes."""
    if dt1.tzinfo is not None:
        dt1 = dt1.replace(tzinfo=None)
    if dt2.tzinfo is not None:
        dt2 = dt2.replace(tzinfo=None)
    return (dt1 - dt2).days


def _now_naive() -> datetime:
    """Get current time as naive datetime."""
    return datetime.utcnow()


# =============================================================================
# STEP 1: SEMANTIC SEARCH (Base Scoring)
# =============================================================================

def get_all_candidates_with_semantic_scores(
    context_embedding: List[float],
    exclude_ids: List[str],
    threshold: float = 0.80
) -> List[Dict]:
    """
    Get ALL memes with semantic similarity scores.
    This is the base - we'll apply multipliers on top.
    
    Lower threshold (0.80) to get more candidates, then boost best ones.
    """
    memes_ref = db.collection('memes').where('status', '==', 'approved').stream()
    
    candidates = []
    for doc in memes_ref:
        meme = doc.to_dict()
        meme_id = doc.id
        
        if meme_id in exclude_ids:
            continue
        
        # Skip customizable-only memes
        if meme.get('meme_type') == 'customizable':
            continue
        
        # Prefer use_case_embedding (text-to-text)
        meme_embedding = meme.get('use_case_embedding') or meme.get('clip_embedding', [])
        if not meme_embedding:
            continue
        
        # Calculate semantic similarity
        similarity = cosine_similarity(context_embedding, meme_embedding)
        
        # Threshold filter
        if similarity < threshold:
            continue
        
        meme['id'] = meme_id
        candidates.append({
            'meme': meme,
            'semantic_score': similarity,
            'base_score': similarity,  # Will be multiplied
            'boosts': {}  # Track what boosts were applied
        })
    
    return candidates


# =============================================================================
# MULTIPLIERS
# =============================================================================

def calculate_personal_boost(candidate: Dict, user_id: str) -> float:
    """
    Personal boost based on user's history with THIS specific meme.
    
    - User sent this meme: 2.0x (strong signal they like it)
    - User favorited: 1.8x
    - User thumbed up: 1.3x
    - No interaction: 1.0x (neutral)
    """
    user = get_user(user_id)
    if not user:
        return 1.0
    
    user_dict = user if isinstance(user, dict) else user.to_dict()
    meme_id = candidate['meme']['id']
    
    # Check interactions
    sent_ids = user_dict.get('sent_meme_ids', [])
    favorite_ids = user_dict.get('favorited_meme_ids', [])
    thumbs_up_ids = user_dict.get('thumbs_up_meme_ids', [])
    
    if meme_id in sent_ids:
        return 2.0  # Strong: user chose to send this
    elif meme_id in favorite_ids:
        return 1.8
    elif meme_id in thumbs_up_ids:
        return 1.3
    else:
        return 1.0


def calculate_global_boost(candidate: Dict) -> float:
    """
    Global boost based on how well this meme performs for everyone.
    
    Formula: thumbs_up_ratio with confidence scaling
    - New meme (0 votes): 1.0x (neutral, no data)
    - 80%+ approval with 10+ votes: up to 1.3x
    - 50% approval: 1.0x
    - <50% approval: 0.9-1.0x
    """
    meme = candidate['meme']
    thumbs_up = meme.get('total_thumbs_up', 0)
    thumbs_down = meme.get('total_thumbs_down', 0)
    total_votes = thumbs_up + thumbs_down
    
    if total_votes == 0:
        return 1.0  # No data, neutral
    
    # Approval ratio
    approval = thumbs_up / total_votes
    
    # Confidence (more votes = more confidence in the boost)
    confidence = min(total_votes / 20, 1.0)  # Full confidence at 20 votes
    
    # Convert approval to boost (50% = 1.0x, 80%+ = 1.3x)
    if approval >= 0.5:
        boost_amount = (approval - 0.5) * 0.6  # 0-30% boost for >50%
        boost = 1.0 + (boost_amount * confidence)
    else:
        penalty = (0.5 - approval) * 0.2  # Up to 10% penalty for <50%
        boost = 1.0 - (penalty * confidence)
    
    return max(0.9, min(boost, 1.3))  # Clamp between 0.9-1.3


def calculate_engagement_multiplier(candidate: Dict, user_id: str) -> float:
    """
    Engagement multiplier for user's feedback on this meme.
    
    - Thumbs down: 0.5x (strong negative)
    - Thumbs up: 1.1x
    - Send: 1.15x
    - Favorite: 1.2x
    
    Note: This is separate from personal_boost which tracks "has seen before"
    """
    user = get_user(user_id)
    if not user:
        return 1.0
    
    user_dict = user if isinstance(user, dict) else user.to_dict()
    meme_id = candidate['meme']['id']
    
    # Check thumbs down FIRST (hard filter)
    thumbs_down_ids = user_dict.get('thumbs_down_meme_ids', [])
    if meme_id in thumbs_down_ids:
        return 0.5
    
    # Otherwise neutral (handled in personal_boost)
    return 1.0


def calculate_tag_affinity_boost(candidate: Dict, user_id: str) -> float:
    """
    Tag affinity boost for context-independent tags.
    
    Max boost: 1.3x
    Based on user's historical preference for tag types.
    """
    user = get_user(user_id)
    if not user:
        return 1.0
    
    user_dict = user if isinstance(user, dict) else user.to_dict()
    tag_affinity = user_dict.get('tag_affinity', {})
    
    if not tag_affinity:
        return 1.0
    
    # Get meme's context-independent tags
    meme_tags = candidate['meme'].get('frontend_tags', [])
    context_indie_tags = [t for t in meme_tags if t in CONTEXT_INDEPENDENT_TAGS]
    
    if not context_indie_tags:
        return 1.0
    
    # Calculate average affinity
    affinities = [tag_affinity.get(tag, 0.0) for tag in context_indie_tags]
    avg_affinity = sum(affinities) / len(affinities) if affinities else 0.0
    
    # Convert to multiplier (0.0 → 1.0x, 5.0+ → 1.3x)
    boost = 1.0 + (min(avg_affinity, 5.0) / 5.0 * 0.3)
    return boost


# =============================================================================
# DIVERSITY FILTER (Minimal)
# =============================================================================

def apply_minimal_diversity(
    candidates: List[Dict],
    threshold: float = 0.90
) -> List[Dict]:
    """
    Remove visually identical memes (CLIP embedding > 0.90).
    Keeps recommendations feeling varied.
    """
    if len(candidates) <= 1:
        return candidates
    
    diverse = []
    
    for candidate in candidates:
        candidate_emb = candidate['meme'].get('clip_embedding', [])
        if not candidate_emb:
            diverse.append(candidate)
            continue
        
        # Check against already selected
        is_duplicate = False
        for selected in diverse:
            selected_emb = selected['meme'].get('clip_embedding', [])
            if not selected_emb:
                continue
            
            similarity = cosine_similarity(candidate_emb, selected_emb)
            if similarity >= threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            diverse.append(candidate)
    
    return diverse


# =============================================================================
# MAIN RECOMMENDATION FUNCTION
# =============================================================================

def recommend_memes_v2(
    user_id: str,
    context: str,
    context_embedding: List[float],
    exclude_ids: List[str] = None,
    num_results: int = 3
) -> Dict:
    """
    V2 Parallel Scoring System
    
    Flow:
    1. Get ALL candidates with semantic scores (base)
    2. Apply parallel multipliers:
       - Personal boost (has user seen/liked this?)
       - Global boost (does this work well for everyone?)
       - Engagement (thumbs down penalty)
       - Tag affinity (context-independent tags)
    3. Calculate final score = semantic × personal × global × engagement × tags
    4. Apply minimal diversity filter
    5. Return top N
    
    All signals contribute - best memes rise to top!
    
    Args:
        user_id: User ID
        context: User's text context
        context_embedding: CLIP embedding of context
        exclude_ids: Meme IDs to exclude
        num_results: Number of memes to return (default 3)
    
    Returns:
        {
            'memes': [...],
            'metadata': {
                'total_candidates': N,
                'avg_semantic': X,
                'after_diversity': M
            }
        }
    """
    if exclude_ids is None:
        exclude_ids = []
    
    # Step 1: Get all candidates with semantic scores
    candidates = get_all_candidates_with_semantic_scores(
        context_embedding=context_embedding,
        exclude_ids=exclude_ids,
        threshold=0.80  # Lower threshold, let multipliers boost best ones
    )
    
    if not candidates:
        return {'memes': [], 'metadata': {'error': 'No candidates found'}}
    
    # Step 2: Apply all multipliers in parallel
    for candidate in candidates:
        # Get all boosts
        personal = calculate_personal_boost(candidate, user_id)
        global_boost = calculate_global_boost(candidate)
        engagement = calculate_engagement_multiplier(candidate, user_id)
        tag_affinity = calculate_tag_affinity_boost(candidate, user_id)
        
        # Calculate final score
        base = candidate['base_score']
        candidate['final_score'] = base * personal * global_boost * engagement * tag_affinity
        
        # Track boosts for debugging
        candidate['boosts'] = {
            'personal': personal,
            'global': global_boost,
            'engagement': engagement,
            'tag_affinity': tag_affinity
        }
    
    # Step 3: Sort by final score
    candidates.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Step 3b: GEMINI RE-RANKING - Use AI to reorder top 30 candidates
    # This provides intelligent context understanding beyond CLIP similarity
    top_30 = candidates[:30]  # Gemini will rank these
    
    if len(top_30) > 0:
        # Prepare meme data for Gemini
        meme_data_for_gemini = [
            {
                'id': c['meme']['id'],
                'use_case': c['meme'].get('use_case', ''),
                'frontend_tags': c['meme'].get('frontend_tags', c['meme'].get('visual_tags', [])),
            }
            for c in top_30
        ]
        
        # Get Gemini's ranking
        try:
            from app.meme_analyzer import rank_memes_with_gemini
            ranked_ids = rank_memes_with_gemini(
                context=context,
                meme_candidates=meme_data_for_gemini,
                batch_size=30
            )
            
            # Reorder candidates based on Gemini's ranking
            if ranked_ids:
                id_to_candidate = {c['meme']['id']: c for c in top_30}
                reordered = []
                for meme_id in ranked_ids:
                    if meme_id in id_to_candidate:
                        reordered.append(id_to_candidate[meme_id])
                
                # Add any candidates Gemini missed (shouldn't happen, but safety)
                seen_ids = set(ranked_ids)
                for c in top_30:
                    if c['meme']['id'] not in seen_ids:
                        reordered.append(c)
                
                # Replace top 30 with reordered version
                candidates = reordered + candidates[30:]
                print(f"✅ Gemini reordered {len(reordered)} candidates for context: '{context[:40]}...'")
        except Exception as e:
            print(f"⚠️ Gemini ranking failed, using CLIP order: {e}")
    
    # Step 4: Apply minimal diversity filter
    diverse = apply_minimal_diversity(candidates, threshold=0.90)
    
    # Check if we have any results
    if not diverse:
        return {
            'memes': [],
            'has_more': False,
            'error': 'no_matching_memes',
            'message': 'No matching memes found for your context. Try a different description!',
            'metadata': {
                'total_candidates': len(candidates),
                'exhausted': True
            }
        }
    
    # Step 5: Return top N
    final_memes = [c['meme'] for c in diverse[:num_results]]
    remaining = len(diverse) - num_results
    
    # Calculate metadata
    avg_semantic = sum(c['semantic_score'] for c in candidates) / len(candidates) if candidates else 0
    
    return {
        'memes': final_memes,
        'session_id': f"{user_id}_{datetime.utcnow().timestamp()}",
        'has_more': remaining > 0,
        'metadata': {
            'total_candidates': len(candidates),
            'avg_semantic_score': round(avg_semantic, 3),
            'after_diversity': len(diverse),
            'returned': len(final_memes),
            'remaining': remaining,
            'gemini_ranked': True
        }
    }
