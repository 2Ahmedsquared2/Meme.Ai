import numpy as np
from typing import List, Dict
from db import db, get_user
from ml_models import get_image_embeddings, get_text_similarity

class ThompsonSamplingEngine:
    """Thompson Sampling recommendation engine for memes"""
    
    def __init__(self):
        self.alpha = 1.0  # Success count (prior)
        self.beta = 1.0   # Failure count (prior)
    
    def get_meme_score(self, meme_dict: dict) -> float:
        """Calculate Thompson Sampling score for a meme"""
        # Get engagement stats
        thumbs_up = meme_dict.get('total_thumbs_up', 0)
        thumbs_down = meme_dict.get('total_thumbs_down', 0)
        
        # Thompson Sampling: sample from Beta distribution
        alpha = self.alpha + thumbs_up
        beta = self.beta + thumbs_down
        
        # Sample a score from Beta(alpha, beta)
        score = np.random.beta(alpha, beta)
        
        return score
    
    def get_personalized_memes(
        self, 
        user_id: str, 
        limit: int = 30,
        context_text: str = None
    ) -> List[dict]:
        """Get personalized meme recommendations"""
        
        # Get user preferences
        user = get_user(user_id)
        if not user:
            return []
        
        # Get candidate memes (approved only)
        all_memes = db.collection('memes').where('status', '==', 'approved').get()
        
        meme_list = []
        for doc in all_memes:
            meme_dict = doc.to_dict()
            
            # Calculate Thompson Sampling score
            ts_score = self.get_meme_score(meme_dict)
            
            # If context provided, use CLIP to boost relevant memes
            if context_text:
                try:
                    # Compare meme to context
                    similarity = get_text_similarity(
                        meme_dict['image_url'],
                        [context_text, "random unrelated text"]
                    )
                    context_boost = similarity.get(context_text, 0.5)
                    
                    # Combine scores
                    final_score = (ts_score * 0.5) + (context_boost * 0.5)
                except:
                    final_score = ts_score
            else:
                final_score = ts_score
            
            meme_dict['score'] = final_score
            meme_list.append(meme_dict)
        
        # Sort by score and return top N
        meme_list.sort(key=lambda x: x['score'], reverse=True)
        return meme_list[:limit]
    
    def get_tag_affinity(self, user_id: str) -> Dict[str, float]:
        """Calculate user's affinity for different tags"""
        user = get_user(user_id)
        if not user:
            return {}
        
        tag_scores = {}
        
        # Get all interactions (thumbs up = positive signal)
        # For now, use onboarding data
        for choice in user.onboarding_data:
            if choice.get('type') == 'scenario':
                meme_id = choice.get('meme_id')
                if meme_id:
                    # Get meme tags
                    meme_doc = db.collection('memes').document(meme_id).get()
                    if meme_doc.exists:
                        meme_tags = meme_doc.to_dict().get('user_tags', [])
                        for tag in meme_tags:
                            tag_scores[tag] = tag_scores.get(tag, 0) + 1
        
        # Normalize scores
        total = sum(tag_scores.values())
        if total > 0:
            tag_scores = {k: v/total for k, v in tag_scores.items()}
        
        return tag_scores

# Global instance
engine = ThompsonSamplingEngine()

