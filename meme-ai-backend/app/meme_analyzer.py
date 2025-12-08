"""
Meme Analyzer - 5 Gemini API Calls Structure
1. Use Case: When/where to use this meme (1-2 sentences, broad/universal)
2. Identity Tags: NICHE - meme name, characters, specific visuals (free-form)
3. Reaction Tags: NICHE - internet slang, specific reactions, cultural refs (free-form)
4. Template Detection: Is customizable? Where are editable zones?
5. Frontend Tags: CONSTRAINED list for user filtering & rec engine tag affinity

Plus: Context Analyzer for "Get Meme" feature
"""

import os
import json
import requests
import io
import time
import threading
from datetime import datetime
from collections import deque
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

load_dotenv()

# Gemini config - using 1.5 Flash for better accuracy
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY required in .env")
genai.configure(api_key=GEMINI_API_KEY)

# Model - using stable flash model with deterministic config
GEMINI_MODEL = "gemini-flash-latest"

# Maximum determinism: temp=0, top_k=1, top_p=0.1
GENERATION_CONFIG = genai.GenerationConfig(
    temperature=0.0,
    top_k=1,
    top_p=0.1,
)

# Relax safety settings for edgy meme content (roasts, dark humor)
from google.generativeai.types import HarmCategory, HarmBlockThreshold
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


def _get_model():
    """Get Gemini model with consistent deterministic config."""
    return genai.GenerativeModel(
        GEMINI_MODEL, 
        generation_config=GENERATION_CONFIG,
        safety_settings=SAFETY_SETTINGS
    )


# =============================================================================
# RATE LIMITER (for Gemini API)
# =============================================================================
class RateLimiter:
    def __init__(self, max_requests: int = 15, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_times = deque()
        self.lock = threading.Lock()
    
    def _cleanup(self):
        cutoff = time.time() - self.window_seconds
        while self.request_times and self.request_times[0] < cutoff:
            self.request_times.popleft()
    
    def can_proceed(self) -> bool:
        with self.lock:
            self._cleanup()
            return len(self.request_times) < self.max_requests
    
    def wait_time(self) -> float:
        with self.lock:
            self._cleanup()
            if len(self.request_times) < self.max_requests:
                return 0
            return max(0, (self.request_times[0] + self.window_seconds) - time.time())
    
    def record(self):
        with self.lock:
            self.request_times.append(time.time())
    
    def status(self) -> dict:
        with self.lock:
            self._cleanup()
            return {
                "used": len(self.request_times),
                "remaining": max(0, self.max_requests - len(self.request_times)),
                "max": self.max_requests,
                "window": self.window_seconds
            }

_rate_limiter = RateLimiter(max_requests=15, window_seconds=60)


# =============================================================================
# FRONTEND TAG CATEGORIES (for reference)
# =============================================================================
FRONTEND_TAGS = {
    "emotion": {
        "desc": "How it makes you feel",
        "tags": ["happy", "sad", "angry", "confused", "shocked", "tired", "cringe", "nervous", "proud", "smug", "disappointed", "betrayed", "frustrated", "satisfied", "anxious"]
    },
    "source": {
        "desc": "Where it comes from",
        "tags": ["anime", "movies", "tv_shows", "gaming", "twitter", "tiktok", "classic", "sports", "nba", "football", "soccer", "music", "youtube", "reddit"]
    },
    "format": {
        "desc": "Visual format",
        "tags": ["reaction", "text_post", "comic", "caption", "comparison", "no_text", "multi_panel", "screenshot", "quotes", "single_panel", "labeled", "drake_format", "expanding_brain", "distracted_boyfriend"]
    },
    "tone": {
        "desc": "The vibe/mood",
        "tags": ["sarcastic", "wholesome", "savage", "relatable", "dark", "chaotic", "ironic", "dramatic", "absurd", "chill", "passive_aggressive", "deadpan", "over_the_top"]
    },
    "situation": {
        "desc": "When to use",
        "tags": ["work", "school", "relationship", "friendship", "online", "family", "monday_mood", "weekend", "night_owl", "morning", "dating", "single_life", "procrastination", "success", "failure"]
    },
    "subject": {
        "desc": "What's in it",
        "tags": ["animals", "celebrities", "cartoon", "people", "food", "nature", "tech", "fictional", "text_only", "objects"]
    },
    "vibe": {
        "desc": "The energy",
        "tags": ["flex", "roast", "cope", "celebration", "disappointment", "realization", "called_out", "exposed", "caught", "winning", "losing", "petty", "unbothered", "down_bad", "apology", "denial"]
        }
    }

def get_all_frontend_tags() -> set:
    tags = set()
    for cat in FRONTEND_TAGS.values():
        tags.update(cat["tags"])
    return tags

def get_frontend_tags_by_category() -> dict:
    return {k: {"desc": v["desc"], "tags": v["tags"]} for k, v in FRONTEND_TAGS.items()}


# =============================================================================
# HELPER: Download image
# =============================================================================
def _download_image(url: str) -> tuple:
    """Download image, return (PIL Image, bytes) or (None, None) on failure"""
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        resp.raise_for_status()
        image_bytes = resp.content
        image_pil = Image.open(io.BytesIO(image_bytes))
        return image_pil, image_bytes
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None, None


# =============================================================================
# GEMINI CALL 1: USE CASE
# =============================================================================
def _gemini_use_case(image_pil: Image.Image) -> str:
    """
    Get 1-2 sentence use case description - BROAD and UNIVERSAL, not specific examples.
    """
    prompt = """You are a meme expert who deeply understands internet culture and meme usage.

Analyze this meme and provide a BROAD, UNIVERSAL description of when someone would use this meme.

IMPORTANT RULES:
- Start with "When..."
- Be BROAD and ABSTRACT - describe the universal feeling/situation, not a specific example
- Should apply to MANY different real-life scenarios
- Focus on the underlying EMOTION or DYNAMIC, not specific contexts
- Maximum 30 words

GOOD examples (broad, universal):
- "When you are enthusiastically reaching for something positive, but an unavoidable responsibility or harsh reality unexpectedly pulls you back."
- "When you strongly prefer one option over another, making your choice clear and unquestionable."
- "When external circumstances prevent you from experiencing something you were looking forward to."
- "When someone or something disrupts your focus or derails your plans at the worst possible moment."

BAD examples (too specific):
- "When your friend finally understands after you explained it 10 times" ❌ (too specific)
- "When you try to pursue a fun hobby but work calls" ❌ (too specific)
- "When your boss makes you redo work" ❌ (too specific)

Just respond with the broad, universal use case sentence, nothing else."""

    try:
        model = _get_model()
        response = model.generate_content(
            [prompt, image_pil],
            request_options={"timeout": 30}
        )
        use_case = response.text.strip()
        # Clean up any quotes
        use_case = use_case.strip('"').strip("'")
        # Truncate if too long (allow longer for broader descriptions)
        if len(use_case) > 200:
            use_case = use_case[:197] + "..."
        return use_case
    except Exception as e:
        print(f"   ❌ Use case error: {e}")
        return ""


# =============================================================================
# GEMINI CALL 2: VISUAL/IDENTITY TAGS (Free-form, niche)
# =============================================================================
def _gemini_visual_tags(image_pil: Image.Image) -> list:
    """
    Get NICHE, SPECIFIC visual/identity tags - NOT generic categories.
    Returns: list of specific tags like ["skeptical_kid", "side_eye", "yoda", "star_wars"]
    """
    prompt = """You are a meme historian and internet culture expert. Analyze this meme and provide SPECIFIC, NICHE tags.

Your job is to tag this meme with SPECIFIC identifiers that would help someone FIND this exact meme.

PROVIDE TAGS FOR:

1. MEME IDENTITY (if recognizable):
   - The actual meme name (e.g., "skeptical_third_world_kid", "distracted_boyfriend", "drake_hotline_bling")
   - Shorter variations people might search (e.g., "skeptical_kid", "drake_meme")

2. CHARACTER/PERSON IDENTIFICATION:
   - Who is in the meme? (e.g., "yoda", "leonardo_dicaprio", "michael_scott", "shrek")
   - Franchise/source if recognizable (e.g., "star_wars", "the_office", "spongebob")

3. SPECIFIC VISUAL DESCRIPTORS:
   - What SPECIFIC expression/pose? (e.g., "side_eye", "pointing", "raising_eyebrow", "smirking")
   - NOT generic like "person" or "reaction" - be SPECIFIC

RULES:
- Use snake_case for multi-word tags
- Be SPECIFIC, not generic (NO: "funny", "meme", "reaction", "people")
- Tags should help FIND this specific meme
- 5-10 tags total
- If you don't recognize the meme, focus on specific visual details

EXAMPLES:
- Skeptical kid meme → ["skeptical_third_world_kid", "skeptical_kid", "side_eye", "african_child", "disbelief"]
- Yoda meme → ["yoda", "star_wars", "baby_yoda", "wise_expression", "green_alien"]
- Drake meme → ["drake", "drake_hotline_bling", "rejection_approval", "two_panel"]

Respond with JSON array only:
["tag1", "tag2", "tag3", ...]"""

    try:
        model = _get_model()
        response = model.generate_content(
            [prompt, image_pil],
            request_options={"timeout": 30}
        )
        text = response.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        tags = json.loads(text)
        if isinstance(tags, list):
            # Clean tags - lowercase, snake_case
            return [t.lower().replace(" ", "_").replace("-", "_") for t in tags if t]
        return []
    except Exception as e:
        print(f"   ❌ Visual tags error: {e}")
        return []


# =============================================================================
# GEMINI CALL 3: CONTEXT/REACTION TAGS (Free-form, niche)
# =============================================================================
def _gemini_context_tags(image_pil: Image.Image) -> list:
    """
    Get NICHE reaction/context tags - internet slang, specific reactions, cultural references.
    Returns: list of specific tags like ["side_eye", "sus", "doubt", "reality_check"]
    """
    prompt = """You are an internet culture expert who deeply understands meme reactions and Gen-Z/millennial slang.

Analyze this meme and provide SPECIFIC, NICHE reaction/context tags.

PROVIDE TAGS FOR:

1. SPECIFIC REACTIONS (not generic emotions):
   - What SPECIFIC reaction is this? (e.g., "side_eye", "sus", "caught_in_4k", "dead_inside", "unbothered_king")
   - NOT generic like "happy" or "sad" - be SPECIFIC to internet culture

2. INTERNET SLANG & PHRASES:
   - What slang fits? (e.g., "its_giving", "understood_the_assignment", "main_character_energy", "no_thoughts_head_empty")
   - Common meme phrases (e.g., "but_thats_none_of_my_business", "and_i_oop", "he_has_a_point")

3. SITUATIONAL TRIGGERS:
   - What situations trigger use? (e.g., "reality_check", "called_out", "caught_lacking", "math_aint_mathing")
   - Setup phrases (e.g., "so_youre_telling_me", "when_they_ask", "me_explaining")

4. CULTURAL CONTEXT:
   - Era/vibe (e.g., "2012_meme", "vintage_internet", "tiktok_era")
   - Community references (e.g., "black_twitter", "stan_twitter", "reddit_moment")

RULES:
- Use snake_case for multi-word tags
- Be NICHE and SPECIFIC - these should feel like insider internet knowledge
- NO generic emotions (happy, sad, angry) - use SPECIFIC internet reactions
- Think: what would someone TYPE to find this reaction?
- 5-10 tags total

EXAMPLES:
- Skeptical kid → ["side_eye", "sus", "doubt", "not_buying_it", "bombastic_side_eye", "so_youre_telling_me", "first_world_problems"]
- Kermit sipping tea → ["but_thats_none_of_my_business", "spilling_tea", "unbothered", "petty", "sipping_tea"]
- This is fine dog → ["everything_is_fine", "denial", "coping", "internal_screaming", "fake_calm"]

Respond with JSON array only:
["tag1", "tag2", "tag3", ...]"""

    try:
        model = _get_model()
        response = model.generate_content(
            [prompt, image_pil],
            request_options={"timeout": 30}
        )
        text = response.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        tags = json.loads(text)
        if isinstance(tags, list):
            # Clean tags - lowercase, snake_case
            return [t.lower().replace(" ", "_").replace("-", "_") for t in tags if t]
        return []
    except Exception as e:
        print(f"   ❌ Context tags error: {e}")
        return []


# =============================================================================
# GEMINI CALL 4: FRONTEND TAGS (Constrained list for rec engine filtering)
# =============================================================================
def _gemini_frontend_tags(image_pil: Image.Image) -> list:
    """
    Get frontend tags from the CONSTRAINED predefined list.
    These are used for user filtering and rec engine tag affinity.
    """
    # Build the tag list string
    all_tags = []
    for cat, data in FRONTEND_TAGS.items():
        all_tags.extend(data["tags"])
    tags_str = ", ".join(all_tags)
    
    prompt = f"""You are tagging a meme for a filtering system. Pick tags ONLY from this list:

{tags_str}

Rules:
- Pick 5-12 tags that apply to this meme
- ONLY use tags from the list above
- These are for user filtering, so pick what users would search for
- Include a mix: 1-2 emotions, 1-2 sources, 1-2 formats, 1-2 tones, etc.

Respond with JSON array only:
["tag1", "tag2", "tag3", ...]"""

    try:
        model = _get_model()
        response = model.generate_content(
            [prompt, image_pil],
            request_options={"timeout": 30}
        )
        text = response.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        tags = json.loads(text)
        if isinstance(tags, list):
            # Filter to only valid frontend tags
            valid_tags = get_all_frontend_tags()
            return [t.lower().replace(" ", "_") for t in tags if t.lower().replace(" ", "_") in valid_tags]
        return []
    except Exception as e:
        print(f"   ❌ Frontend tags error: {e}")
        return []


# =============================================================================
# GEMINI CALL 5: TEMPLATE DETECTION + TEXT ZONES
# =============================================================================
def _gemini_template_detection(image_pil: Image.Image) -> dict:
    """
    Detect if meme is customizable and where text can be added.
    Returns: {"is_template": bool, "text_zones": [...]}
    """
    prompt = """You are a meme expert analyzing if this meme can be customized.

A meme is CUSTOMIZABLE (is_template = true) if:
- It has EMPTY/BLANK areas where users typically add their own text
- It's a known meme format where people change the text (Drake, Distracted Boyfriend, Two Buttons, etc.)
- Even if there IS text, if it's a format where that text is meant to be replaced

A meme is NOT customizable (is_template = false) if:
- The text is integral to the joke and shouldn't be changed
- It's a screenshot/reaction with no editable areas
- It's complete as-is

If is_template is TRUE, identify the text zones:
- Each zone needs: id (number), label (what goes there like "top text", "option 1", "rejected thing"), and bounding box
- Bounding box format: [ymin, xmin, ymax, xmax] as percentages 0-100 of image dimensions

Respond in JSON:
{
  "is_template": true/false,
  "text_zones": [
    {"id": 1, "label": "top text", "box_2d": [0, 10, 15, 90]},
    {"id": 2, "label": "bottom text", "box_2d": [85, 10, 100, 90]}
  ]
}

If is_template is false, text_zones should be empty array []."""

    try:
        model = _get_model()
        response = model.generate_content(
            [prompt, image_pil],
            request_options={"timeout": 30}
        )
        # Parse JSON from response
        text = response.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        data = json.loads(text)
        return {
            "is_template": data.get("is_template", False),
            "text_zones": data.get("text_zones", [])
        }
    except Exception as e:
        print(f"   ❌ Template detection error: {e}")
        return {"is_template": False, "text_zones": []}


# =============================================================================
# MAIN: ANALYZE MEME (4 Gemini calls)
# =============================================================================
def analyze_meme(image_url: str, wait_if_limited: bool = True) -> dict:
    """
    Full meme analysis with 5 Gemini API calls:
    1. Use Case
    2. Identity Tags (niche - meme name, characters)
    3. Reaction Tags (niche - internet slang, specific reactions)
    4. Template Detection
    5. Frontend Tags (constrained list for filtering)
    """
    
    # Check rate limit (5 calls per meme)
    for _ in range(5):
        if not _rate_limiter.can_proceed():
            wait = _rate_limiter.wait_time()
            if wait_if_limited:
                print(f"   ⏳ Rate limited, waiting {wait:.1f}s...")
                time.sleep(wait + 1)
            else:
                return {
                    "use_case": "", "frontend_tags": [], "contextual_tags": [],
                    "visual_tags": [], "is_template": False, "text_zones": [],
                    "rate_limited": True, "wait_seconds": wait
                }
    
    # Download image
    print(f"📥 Downloading: {image_url[:60]}...")
    image_pil, image_bytes = _download_image(image_url)
    if image_pil is None:
        return {
            "use_case": "", "frontend_tags": [], "contextual_tags": [],
            "visual_tags": [], "is_template": False, "text_zones": []
        }
    
    result = {
        "use_case": "",
        "use_case_embedding": [],  # NEW: Text embedding for semantic search
        "frontend_tags": [],
        "contextual_tags": [],
        "visual_tags": [],
        "is_template": False,
        "text_zones": []
    }

    # === CALL 1: Use Case ===
    print("🔄 Gemini 1/5: Use Case...")
    _rate_limiter.record()
    result["use_case"] = _gemini_use_case(image_pil)
    print(f"   ✓ {result['use_case'][:60]}...")
    
    # Generate text embedding for the use_case (for semantic search)
    if result["use_case"]:
        from app.transformer import get_text_embedding
        print("🔄 Generating use_case embedding...")
        result["use_case_embedding"] = get_text_embedding(result["use_case"])
        print(f"   ✓ Embedding generated ({len(result['use_case_embedding'])} dims)")

    # Small delay between calls
    time.sleep(0.3)
    
    # === CALL 2: Visual/Identity Tags (NICHE) ===
    print("🔄 Gemini 2/4: Identity Tags...")
    _rate_limiter.record()
    result["visual_tags"] = _gemini_visual_tags(image_pil)
    print(f"   ✓ Identity: {result['visual_tags'][:5]}{'...' if len(result['visual_tags']) > 5 else ''}")
    
    time.sleep(0.3)
    
    # === CALL 3: Context/Reaction Tags (NICHE) ===
    print("🔄 Gemini 3/4: Reaction Tags...")
    _rate_limiter.record()
    result["contextual_tags"] = _gemini_context_tags(image_pil)
    print(f"   ✓ Reactions: {result['contextual_tags'][:5]}{'...' if len(result['contextual_tags']) > 5 else ''}")
    
    time.sleep(0.3)
    
    # === CALL 4: Template Detection ===
    print("🔄 Gemini 4/4: Template Detection...")
    _rate_limiter.record()
    template_data = _gemini_template_detection(image_pil)
    result["is_template"] = template_data.get("is_template", False)
    result["text_zones"] = template_data.get("text_zones", [])
    print(f"   ✓ Is Template: {result['is_template']}")
    print(f"   ✓ Text Zones: {len(result['text_zones'])}")
    
    time.sleep(0.3)
    
    # === CALL 5: Frontend Tags (constrained list for rec engine) ===
    print("🔄 Gemini 5/5: Frontend Tags...")
    _rate_limiter.record()
    result["frontend_tags"] = _gemini_frontend_tags(image_pil)
    print(f"   ✓ Frontend: {result['frontend_tags'][:5]}{'...' if len(result['frontend_tags']) > 5 else ''}")
    
    total_niche = len(result["visual_tags"]) + len(result["contextual_tags"])
    print(f"✅ Done: {total_niche} niche tags, {len(result['frontend_tags'])} frontend tags, template={result['is_template']}")
    return result


# =============================================================================
# CONTEXT ANALYZER (for "Get Meme" feature)
# =============================================================================
# This prompt is designed to match the database's abstract "When..." structure
# for optimal vector similarity (cosine similarity)
CONTEXT_PROMPT = """You are a "Meme Translator." Your goal is to convert a specific user situation into a BROAD, UNIVERSAL meme use-case.

INPUT: A user's text or screenshot describing a specific moment.
OUTPUT: A single sentence starting with "When..." that describes the underlying dynamic/emotion abstractly.

CRITICAL RULES FOR VECTOR MATCHING:
1. Do NOT describe the specific details (names, specific items, specific text).
2. Describe the UNIVERSAL FEELING or DYNAMIC.
3. This output will be compared against a database of generic meme descriptions.

EXAMPLES:
- User Input: "My boss just made me redo 3 hours of work."
  - BAD Output: "When your boss makes you redo work." (Too specific)
  - GOOD Output: "When you have to restart a difficult task that you thought was finished."

- User Input: "Happy meme"
  - BAD Output: "Happy joyful celebration." (Keywords don't match sentence vectors well)
  - GOOD Output: "When you experience a sudden moment of pure success and joy."

- User Input: "Sad meme"
  - GOOD Output: "When something disappointing happens and you feel defeated."

- User Input: [Screenshot of a text where someone cancels plans last minute]
  - BAD Output: "When Sarah cancels dinner."
  - GOOD Output: "When someone cancels plans you were secretly hoping to avoid."

- User Input: "When my code finally works"
  - GOOD Output: "When you finally solve a frustrating problem after many failed attempts."

Analyze the context below and provide ONLY the "When..." sentence (nothing else).

"""


def analyze_context(text: str = None, image_url: str = None) -> dict:
    """
    Analyze user's context (text and/or screenshot) and return meme search query.
    Used for the "Get Meme" feature.
    Outputs abstract "When..." sentences to match database vector structure.
    """
    if not text and not image_url:
        return {"meme_search_query": "", "error": "Need text or image"}
    
    content = [CONTEXT_PROMPT]
    
    # Add text context
    if text:
        content.append(f"\nUSER CONTEXT TEXT: {text}")
    
    # Add image context
    if image_url:
        try:
            resp = requests.get(image_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
            content.append(img)
            content.append("\n[Instruction: Analyze the screenshot above. Ignore specific names/dates. Extract the abstract situation.]")
        except Exception as e:
            if not text:
                return {"meme_search_query": "", "error": str(e)}
    
    try:
        model = _get_model()
        response = model.generate_content(content, request_options={"timeout": 30})
        query = response.text.strip()
        # Clean up quotes
        query = query.strip('"').strip("'")
        
        # Fallback: If model fails to start with "When", force it (helps vector alignment)
        if query and not query.lower().startswith("when"):
            query = f"When {query}"
            
        return {"meme_search_query": query, "error": None}
    except Exception as e:
        return {"meme_search_query": "", "error": str(e)}


def analyze_context_from_upload(image_bytes: bytes, text: str = None) -> dict:
    """Analyze context from uploaded image bytes (for direct uploads)"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except:
        return {"meme_search_query": "", "error": "Invalid image"}
    
    content = [CONTEXT_PROMPT]
    if text:
        content.append(f"\nUSER CONTEXT TEXT: {text}")
    content.append(img)
    content.append("\n[Instruction: Analyze the screenshot above. Ignore specific names/dates. Extract the abstract situation.]")
    
    try:
        model = _get_model()
        response = model.generate_content(content, request_options={"timeout": 30})
        query = response.text.strip()
        query = query.strip('"').strip("'")
        
        # Fallback: If model fails to start with "When", force it (helps vector alignment)
        if query and not query.lower().startswith("when"):
            query = f"When {query}"
            
        return {"meme_search_query": query, "error": None}
    except Exception as e:
        return {"meme_search_query": "", "error": str(e)}


# =============================================================================
# EXPORTS
# =============================================================================
def get_rate_status() -> dict:
    return _rate_limiter.status()


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Meme Analyzer (4 Gemini Calls)")
    print("=" * 60)
    
    # Test with Drake meme
    test_url = "https://i.imgflip.com/30b1gx.jpg"
    print(f"\nTesting with: {test_url}")
    
    result = analyze_meme(test_url)
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Use Case: {result['use_case']}")
    print(f"Frontend Tags: {result['frontend_tags']}")
    print(f"Visual Tags: {result['visual_tags']}")
    print(f"Contextual Tags: {result['contextual_tags']}")
    print(f"Is Template: {result['is_template']}")
    print(f"Text Zones: {result['text_zones']}")
    
    # Test context analysis
    print("\n" + "=" * 60)
    print("Testing Context Analyzer")
    print("=" * 60)
    ctx = analyze_context(text="my boss just made me redo 3 hours of work")
    print(f"Search Query: {ctx['meme_search_query']}")


# =============================================================================
# GEMINI MEME RANKER - Re-rank CLIP candidates with Gemini intelligence
# =============================================================================
def rank_memes_with_gemini(
    context: str,
    meme_candidates: list,
    batch_size: int = 30
) -> list:
    """
    Use Gemini to re-rank CLIP-selected meme candidates for better context matching.
    
    Args:
        context: User's context (e.g., "sad meme", "when my friend cancels plans")
        meme_candidates: List of meme dicts with 'id', 'use_case', 'frontend_tags', 'image_url'
        batch_size: How many memes to rank (default 30, can reduce if context too long)
    
    Returns:
        List of meme IDs in ranked order (best match first)
    """
    if not meme_candidates:
        return []
    
    # Limit to batch size
    candidates = meme_candidates[:batch_size]
    
    # Build meme descriptions for Gemini
    meme_descriptions = []
    for i, meme in enumerate(candidates):
        meme_id = meme.get('id', f'meme_{i}')
        use_case = meme.get('use_case', 'No description')[:150]  # Truncate long descriptions
        tags = meme.get('frontend_tags', [])[:8]  # Limit tags
        
        meme_descriptions.append(f"{i+1}. ID: {meme_id}\n   Use: {use_case}\n   Tags: {', '.join(tags)}")
    
    memes_text = "\n\n".join(meme_descriptions)
    
    prompt = f"""You are a meme expert. A user wants a meme for this context:

USER CONTEXT: "{context}"

Here are {len(candidates)} meme candidates. Rank them from BEST match to WORST match for the user's context.

MEMES:
{memes_text}

INSTRUCTIONS:
1. Consider the emotional tone (sad, happy, frustrated, etc.)
2. Consider the situation described
3. Rank ALL memes from 1 (best) to {len(candidates)} (worst)
4. Return ONLY a JSON array of meme IDs in ranked order

EXAMPLE OUTPUT:
["meme_123", "meme_456", "meme_789"]

YOUR RANKING (JSON array only, no explanation):"""

    try:
        model = _get_model()
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            print("❌ Gemini ranking: Empty response")
            return [m.get('id') for m in candidates]  # Return original order
        
        # Parse JSON response
        text = response.text.strip()
        
        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        
        ranked_ids = json.loads(text)
        
        # Validate we got a list of IDs
        if not isinstance(ranked_ids, list):
            print(f"❌ Gemini ranking: Expected list, got {type(ranked_ids)}")
            return [m.get('id') for m in candidates]
        
        print(f"✅ Gemini ranked {len(ranked_ids)} memes for context: '{context[:50]}...'")
        return ranked_ids
        
    except json.JSONDecodeError as e:
        print(f"❌ Gemini ranking JSON error: {e}")
        print(f"   Raw response: {response.text[:200] if response else 'None'}")
        return [m.get('id') for m in candidates]
    except Exception as e:
        print(f"❌ Gemini ranking error: {e}")
        return [m.get('id') for m in candidates]


def rank_memes_batch(
    context: str,
    meme_candidates: list,
    batch_size: int = 10
) -> list:
    """
    Fallback: Rank memes in smaller batches if full ranking fails.
    
    Args:
        context: User's context
        meme_candidates: All candidates
        batch_size: Size of each batch (default 10)
    
    Returns:
        List of meme IDs in ranked order
    """
    all_ranked = []
    
    for i in range(0, len(meme_candidates), batch_size):
        batch = meme_candidates[i:i + batch_size]
        ranked = rank_memes_with_gemini(context, batch, batch_size=batch_size)
        all_ranked.extend(ranked)
    
    return all_ranked
