from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import torch 
from typing import List, Dict

# Global model storage (lazy loading)
blip_caption_processor = None
blip_caption_model = None
clip_processor = None
clip_model = None

# ==================== MODEL LOADING ====================

def load_blip_caption():
    """Load BLIP model for image captioning"""
    global blip_caption_processor, blip_caption_model
    if blip_caption_model is None:
        print("Loading BLIP captioning model...")
        blip_caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return blip_caption_processor, blip_caption_model

def load_clip():
    """Load CLIP model for image-text similarity"""
    global clip_processor, clip_model
    if clip_model is None:
        print("Loading CLIP model...")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return clip_processor, clip_model

# ==================== BLIP FUNCTIONS ====================

def get_meme_caption(image_url: str) -> str:
    """Generate descriptive caption using BLIP"""
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        processor, model = load_blip_caption()
        
        inputs = processor(img, return_tensors="pt")
        out = model.generate(**inputs, max_length=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        return caption
    except Exception as e:
        print(f"Error generating caption: {e}")
        return ""

# ==================== CLIP FUNCTIONS ====================

def get_text_similarity(image_url: str, text_options: List[str]) -> Dict[str, float]:
    """Compare image to multiple text options using CLIP"""
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        
        processor, model = load_clip()
        
        inputs = processor(text=text_options, images=img, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        
        results = {}
        for i, text in enumerate(text_options):
            results[text] = float(probs[0][i])
        
        return results
    except Exception as e:
        print(f"Error: {e}")
        return {}

# ==================== COMPLETE AUTO-TAGGING SYSTEM ====================

def auto_tag_meme(image_url: str) -> Dict:
    """
    Complete hybrid tagging system using CLIP + BLIP
    Uses category-specific CLIP calls to avoid softmax dilution
    Returns visual tags, contextual tags, and caption
    """
    
    # === VISUAL TAGS - SPLIT INTO 4 SMALLER GROUPS ===
    
    # VISUAL GROUP 1: IMAGE TYPE & TEXT (20 tags)
    visual_type_options = [
        "portrait photo", "landscape photo", "screenshot", "illustration",
        "cartoon", "digital art", "stock photo", "selfie", "comic panel",
        "text-heavy meme", "minimal text", "no text",
        "caption at top", "caption at bottom", "text overlay", "speech bubble",
        "black and white", "colorful", "dark themed", "bright themed"
    ]
    
    visual_type_scores = get_text_similarity(image_url, visual_type_options)
    visual_type_tags = [tag for tag, score in visual_type_scores.items() if score > 0.25]
    
    # VISUAL GROUP 2: PEOPLE & SUBJECTS (21 tags)
    visual_people_options = [
        "no people", "one person", "two people", "group of people", "crowd",
        "woman", "man", "child", "elderly person", "celebrity", "influencer",
        "cartoon character", "animal character",
        "animal", "cat", "dog", "car", "food", "computer", "phone",
        "nature scenery", "urban environment"
    ]
    
    visual_people_scores = get_text_similarity(image_url, visual_people_options)
    visual_people_tags = [tag for tag, score in visual_people_scores.items() if score > 0.25]
    
    # VISUAL GROUP 3: ACTIONS & EXPRESSIONS (18 tags)
    visual_action_options = [
        "running", "crying", "laughing", "yelling", "pointing",
        "sleeping", "dancing", "arguing", "smiling", "facepalm",
        "looking away", "staring", "shocked face", "side glance",
        "making a choice", "comparing two things", "multiple people interacting",
        "zoomed-in face"
    ]
    
    visual_action_scores = get_text_similarity(image_url, visual_action_options)
    visual_action_tags = [tag for tag, score in visual_action_scores.items() if score > 0.25]
    
    # VISUAL GROUP 4: MEME FORMATS (21 tags)
    visual_format_options = [
        "two-panel", "three-panel", "four-panel", "side-by-side comparison",
        "before and after", "expectation vs reality", "text message screenshot",
        "choice comparison", "ascending levels", "descending quality",
        "approval rejection", "reaction image", "emoji visible",
        # Popular specific formats
        "Drake meme", "distracted boyfriend", "expanding brain",
        "guy looking back", "woman yelling at cat",
        "Batman slapping", "Success Kid", "Harold"
    ]
    
    visual_format_scores = get_text_similarity(image_url, visual_format_options)
    visual_format_tags = [tag for tag, score in visual_format_scores.items() if score > 0.25]
    
    # Combine all visual tags
    visual_tags = visual_type_tags + visual_people_tags + visual_action_tags + visual_format_tags
    
    # === CONTEXTUAL TAGS - SPLIT INTO 4 SMALLER GROUPS ===
    
    # CONTEXT GROUP 1: CORE EMOTIONS (24 tags)
    emotion_options = [
        "happy", "sad", "angry", "confused", "shocked", "disgusted",
        "tired", "excited", "bored", "anxious", "proud", "embarrassed",
        "nervous", "uncomfortable", "smug", "mocking", "judging",
        "surprised", "frustrated", "relieved", "scared", "confident",
        "awkward", "guilty"
    ]
    
    emotion_scores = get_text_similarity(image_url, emotion_options)
    emotion_tags = [tag for tag, score in emotion_scores.items() if score > 0.15]
    
    # CONTEXT GROUP 2: MOODS & VIBES (28 tags)
    vibe_options = [
        "funny", "wholesome", "relatable", "sarcastic", "ironic",
        "dark humor", "absurd", "cursed", "blessed", "chaotic",
        "celebration", "disappointment", "approval", "rejection",
        "realization", "facepalm", "cringe",
        "sassy", "passive aggressive", "dry humor",
        "dramatic", "overreacting", "underreacting",
        "delusional confidence", "defensive", "playful teasing",
        "joking insult", "self roast"
    ]
    
    vibe_scores = get_text_similarity(image_url, vibe_options)
    vibe_tags = [tag for tag, score in vibe_scores.items() if score > 0.15]
    
    # CONTEXT GROUP 3: SITUATIONS & STATES (32 tags)
    situation_options = [
        "being called out", "exposed", "caught lying",
        "panic moment", "embarrassing mistake", "miscommunication",
        "failure", "success moment", "unexpected twist",
        "overthinking", "procrastinating", "trying your best",
        "responsibility avoidance",
        "distracted", "tempted", "caught looking", "wandering eyes",
        "choosing between options", "torn between two things", "can't decide",
        "jealous partner", "side eye", "unfaithful", "comparing",
        "grass is greener", "FOMO", "regret", "what if",
        "monday mood", "friday feeling", "2am thoughts", "existential crisis"
    ]
    
    situation_scores = get_text_similarity(image_url, situation_options)
    situation_tags = [tag for tag, score in situation_scores.items() if score > 0.15]
    
    # CONTEXT GROUP 4: SOCIAL CONTEXTS & CATEGORIES (32 tags)
    social_options = [
        "work meme", "school meme", "relationship meme", "friendship meme",
        "family meme", "internet culture", "gaming meme", "sports meme",
        "best friends", "siblings", "crush", "dating",
        "teammates", "coworkers", "group chat",
        "parents vs kids", "teacher vs student",
        "adulting struggles", "social anxiety", "introvert problems", "extrovert energy",
        "chronically online", "video call awkwardness",
        "gen z humor", "millennial humor",
        "classic meme", "reaction image", "text meme", "tweet meme",
        "relatable pain", "motivational", "coping humor"
    ]
    
    social_scores = get_text_similarity(image_url, social_options)
    social_tags = [tag for tag, score in social_scores.items() if score > 0.15]
    
    # === COMBINE ALL CONTEXTUAL TAGS ===
    all_contextual_tags = emotion_tags + vibe_tags + situation_tags + social_tags
    
    # === BLIP CAPTION ===
    caption = get_meme_caption(image_url)
    
    # === RETURN RESULTS ===
    return {
        "visual_tags": visual_tags,
        "contextual_tags": list(set(all_contextual_tags)),  # Remove duplicates
        "blip2_caption": caption,
        "all_tags": visual_tags + list(set(all_contextual_tags))
    }
