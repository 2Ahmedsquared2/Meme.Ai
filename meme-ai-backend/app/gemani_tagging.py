"""
Gemini-powered meme tagging service
Generates use cases and tags for memes using Gemini 1.5 Pro Vision
"""

import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure Gemini with API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Check your .env file.")

genai.configure(api_key=api_key)

# Use Gemini 1.5 Pro for vision tasks
model = genai.GenerativeModel('gemini-1.5-pro')


def analyze_meme(image_url: str) -> dict:
    """
    Analyze a meme image and generate use case + tags
    
    Args:
        image_url: URL of the meme image
        
    Returns:
        dict with keys: use_case, visual_tags, contextual_tags
    """
    
    prompt = """You are a meme expert who understands internet culture and meme usage.

Analyze this meme image and provide:

1. use_case: Write 1-2 sentences describing WHEN someone would send this meme in a text conversation. Be specific about the situation, emotion, or context. Think about real texting scenarios.

2. visual_tags: List 5-8 tags describing what's visually in the image (e.g., "reaction image", "Anime", "surprised face", "two people talking", "cartoon", "Sports", "Basketball", "twitter meme")

3. contextual_tags: List 5-8 tags describing the emotional/situational context when this meme is used (e.g., "being called out", "awkward moment", "roasting friend", "self-deprecating", "sarcastic response")

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{
  "use_case": "When your friend says something so obvious that you can't believe they just realized it",
  "visual_tags": ["reaction image", "surprised face", "cartoon"],
  "contextual_tags": ["obvious realization", "disbelief", "teasing friend"]
}"""

    try:
        # Send image URL to Gemini
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_url}
        ])
        
        # Parse the JSON response
        result_text = response.text.strip()
        
        # Clean up response if it has markdown code blocks
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_text = result_text.strip()
        
        result = json.loads(result_text)
        
        return {
            "use_case": result.get("use_case", ""),
            "visual_tags": result.get("visual_tags", []),
            "contextual_tags": result.get("contextual_tags", [])
        }
        
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse Gemini response as JSON: {e}")
        print(f"Raw response: {response.text}")
        return {"use_case": "", "visual_tags": [], "contextual_tags": []}
        
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return {"use_case": "", "visual_tags": [], "contextual_tags": []}


# Test function
if __name__ == "__main__":
    # Test with a sample meme
    test_url = "https://i.imgflip.com/1bij.jpg"  # Drake meme
    print(f"Testing with: {test_url}")
    
    result = analyze_meme(test_url)
    print("\n✅ Result:")
    print(f"Use case: {result['use_case']}")
    print(f"Visual tags: {result['visual_tags']}")
    print(f"Contextual tags: {result['contextual_tags']}")

