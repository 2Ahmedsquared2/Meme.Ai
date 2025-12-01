"""
Imgflip Meme Scraper - Gets classic, timeless meme templates
These are broad-context memes that work for any situation
Perfect for training data!
"""
import sys
sys.path.insert(0, 'app')

import requests
import time
from io import BytesIO
from PIL import Image
import imagehash
from datetime import datetime
from db import db, Meme, create_meme
from ml_model import auto_tag_meme

# Imgflip's top 100 most popular meme templates
IMGFLIP_API = "https://api.imgflip.com/get_memes"

def get_imgflip_templates():
    """Get popular meme templates from Imgflip API"""
    try:
        response = requests.get(IMGFLIP_API, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return data['data']['memes']
    except Exception as e:
        print(f"❌ Error getting templates: {e}")
    return []

def process_meme_template(template):
    """Process a meme template through CLIP + BLIP"""
    try:
        url = template['url']
        name = template['name']
        
        # Download for hash check
        img_response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        if img_response.status_code != 200:
            return False, "download_failed"
        
        # Validate image
        try:
            img = Image.open(BytesIO(img_response.content))
            img_hash = str(imagehash.phash(img))
        except:
            return False, "invalid_image"
        
        # Check duplicates
        existing = db.collection('memes').where('image_hash', '==', img_hash).limit(1).get()
        if len(list(existing)) > 0:
            return False, "duplicate"
        
        # Generate ID
        timestamp = int(datetime.now().timestamp())
        meme_id = f"imgflip_{template['id']}_{timestamp}"
        
        print(f"   🤖 Running CLIP + BLIP...")
        
        # Run ML tagging
        ml_tags = auto_tag_meme(url)
        
        # Create meme
        new_meme = Meme(
            id=meme_id,
            image_url=url,
            firebase_image_url=url,
            image_hash=img_hash,
            user_tags=[],  # Empty - you can add custom tags later
            visual_tags=ml_tags["visual_tags"],
            contextual_tags=ml_tags["contextual_tags"],
            blip2_caption=ml_tags["blip2_caption"],
            all_tags=ml_tags["all_tags"],
            clip_embedding=ml_tags.get("clip_embedding", []),
            status="pending",
            upload_time=datetime.now()
        )
        
        create_meme(new_meme)
        return True, "success"
        
    except Exception as e:
        return False, str(e)[:100]

def main():
    print("🎯 Imgflip Classic Meme Template Scraper")
    print("📡 Getting timeless, broad-context memes")
    print("🔄 Processing through CLIP + BLIP")
    print("="*60)
    
    templates = get_imgflip_templates()
    print(f"\n✅ Found {len(templates)} popular meme templates from Imgflip")
    print("\nThese are the CLASSIC memes everyone knows:")
    print("  - Drake Yes/No")
    print("  - Distracted Boyfriend")
    print("  - Two Buttons")
    print("  - Expanding Brain")
    print("  - Success Kid")
    print("  - And 95 more...\n")
    print("="*60)
    
    uploaded = 0
    skipped = 0
    
    for i, template in enumerate(templates, 1):
        name = template['name']
        print(f"\n[{i}/{len(templates)}] {name}")
        
        success, message = process_meme_template(template)
        if success:
            uploaded += 1
            print(f"   ✅ Processed!")
        else:
            skipped += 1
            if message == "duplicate":
                print(f"   ⏭️  Duplicate")
            else:
                print(f"   ❌ {message}")
        
        time.sleep(0.5)  # Rate limiting
    
    print("\n" + "="*60)
    print(f"🎉 COMPLETE!")
    print(f"✅ Processed: {uploaded} classic meme templates")
    print(f"⏭️  Skipped: {skipped}")
    print("\n🎯 These memes are:")
    print("   ✅ Timeless (not tied to current events)")
    print("   ✅ Universal (everyone recognizes them)")
    print("   ✅ Versatile (work in many contexts)")
    print("   ✅ Perfect for training!")
    print("\n📍 Go to http://localhost:8000/admin to review!")

if __name__ == "__main__":
    main()

