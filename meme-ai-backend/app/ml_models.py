from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import torch

# Initialize models (lazy loading)
blip_processor = None
blip_model = None
clip_processor = None
clip_model = None

def load_blip():
    """Load BLIP-2 model for image captioning"""
    global blip_processor, blip_model
    if blip_model is None:
        print("Loading BLIP-2 model...")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return blip_processor, blip_model

def load_clip():
    """Load CLIP model for image-text similarity"""
    global clip_processor, clip_model
    if clip_model is None:
        print("Loading CLIP model...")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return clip_processor, clip_model

def get_meme_caption(image_url: str) -> str:
    """Extract caption from meme using BLIP-2"""
    try:
        # Load image
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Load BLIP
        processor, model = load_blip()
        
        # Generate caption
        inputs = processor(img, return_tensors="pt")
        out = model.generate(**inputs, max_length=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        return caption
    except Exception as e:
        print(f"Error generating caption: {e}")
        return ""

def get_image_embeddings(image_url: str):
    """Get CLIP embeddings for an image"""
    try:
        # Load image
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Load CLIP
        processor, model = load_clip()
        
        # Get embeddings
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
        
        return embeddings.numpy().tolist()[0]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

def get_text_similarity(image_url: str, text_options: list) -> dict:
    """Compare image to multiple text options using CLIP"""
    try:
        # Load image
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Load CLIP
        processor, model = load_clip()
        
        # Process image and texts
        inputs = processor(text=text_options, images=img, return_tensors="pt", padding=True)
        
        # Get similarity scores
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Return text options with scores
        results = {}
        for i, text in enumerate(text_options):
            results[text] = float(probs[0][i])
        
        return results
    except Exception as e:
        print(f"Error computing similarity: {e}")
        return {}

