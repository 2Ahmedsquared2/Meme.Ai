"""
Meme Customizer - PIL-based text overlay for meme templates
Uses text_zones from meme_analyzer to place text
"""

import io
import base64
import textwrap
import requests
from PIL import Image, ImageDraw, ImageFont


def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Get font that works cross-platform"""
    paths = [
        "./fonts/Impact.ttf",
        "./fonts/Anton-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except:
            continue
    
    return ImageFont.load_default()


def draw_text_in_box(draw: ImageDraw.Draw, box: list, text: str, 
                     font: ImageFont.FreeTypeFont, width: int, height: int):
    """Draw text centered in box with wrapping and outline"""
    ymin, xmin, ymax, xmax = box
    
    # Convert 0-1000 coords to pixels
    x1 = int(xmin * width / 1000)
    y1 = int(ymin * height / 1000)
    x2 = int(xmax * width / 1000)
    y2 = int(ymax * height / 1000)
    
    box_width = x2 - x1
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Get font size for wrapping calc
    try:
        font_size = font.size
    except:
        font_size = 20
    
    # Wrap text to fit box
    chars_per_line = max(5, int(box_width / (font_size * 0.6)))
    lines = textwrap.wrap(text.upper(), width=chars_per_line)
    
    line_height = font_size * 1.2
    total_height = len(lines) * line_height
    current_y = center_y - (total_height / 2) + (line_height / 2)
    
    for line in lines:
        # Draw black outline
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                draw.text((center_x + dx, current_y + dy), line, 
                         font=font, fill="black", anchor="mm")
        # Draw white text
        draw.text((center_x, current_y), line, font=font, fill="white", anchor="mm")
        current_y += line_height


def customize_meme(image_url: str, text_inputs: dict, text_zones: list) -> dict:
    """
    Overlay text onto meme template.
    
    Args:
        image_url: URL of the template image
        text_inputs: {zone_id: "text to add"} mapping
        text_zones: List of zones from meme_analyzer (id, label, box_2d)
    
    Returns:
        {"success": bool, "image_data": base64, "mime_type": str, "error": str}
    """
    
    # Download image
    try:
        resp = requests.get(image_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
    except Exception as e:
        return {"success": False, "image_data": None, "mime_type": None, "error": str(e)}
    
    # Scale font to image
    font_size = max(20, min(width, height) // 12)
    font = get_font(font_size)
    
    # Draw text in each zone
    for zone in text_zones:
        zone_id = zone.get("id")
        box = zone.get("box_2d", [])
        
        # Skip if no text for this zone
        if zone_id not in text_inputs:
            continue
        
        text = text_inputs[zone_id]
        if not text or len(box) != 4:
            continue
        
        draw_text_in_box(draw, box, text, font, width, height)
    
    # Export as base64 JPEG
    try:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "success": True,
            "image_data": img_b64,
            "mime_type": "image/jpeg",
            "error": None
        }
    except Exception as e:
        return {"success": False, "image_data": None, "mime_type": None, "error": str(e)}


def customize_meme_from_bytes(image_bytes: bytes, text_inputs: dict, text_zones: list) -> dict:
    """Same as customize_meme but from raw bytes instead of URL"""
    
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
    except Exception as e:
        return {"success": False, "image_data": None, "mime_type": None, "error": str(e)}
    
    font_size = max(20, min(width, height) // 12)
    font = get_font(font_size)
    
    for zone in text_zones:
        zone_id = zone.get("id")
        box = zone.get("box_2d", [])
        
        if zone_id not in text_inputs:
            continue
        
        text = text_inputs[zone_id]
        if not text or len(box) != 4:
            continue
        
        draw_text_in_box(draw, box, text, font, width, height)
    
    try:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "success": True,
            "image_data": img_b64,
            "mime_type": "image/jpeg",
            "error": None
        }
    except Exception as e:
        return {"success": False, "image_data": None, "mime_type": None, "error": str(e)}


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("Testing Meme Customizer")
    
    # Example usage with Drake meme template
    test_url = "https://i.imgflip.com/30b1gx.jpg"
    test_zones = [
        {"id": 1, "label": "top text", "box_2d": [0, 500, 500, 1000]},
        {"id": 2, "label": "bottom text", "box_2d": [500, 500, 1000, 1000]}
    ]
    test_text = {
        1: "Writing documentation",
        2: "Just shipping it and hoping for the best"
    }
    
    result = customize_meme(test_url, test_text, test_zones)
    
    if result["success"]:
        # Save test output
        import base64
        with open("test_custom.jpg", "wb") as f:
            f.write(base64.b64decode(result["image_data"]))
        print("✓ Saved test_custom.jpg")
    else:
        print(f"✗ Error: {result['error']}")
