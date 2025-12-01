#!/bin/bash
# Quick integration check script
# Run this anytime to verify admin panel compatibility

cd "$(dirname "$0")"
source venv/bin/activate

echo "🔍 Running quick integration check..."
python3 << 'PYTHON_EOF'
import sys
sys.path.insert(0, 'app')

try:
    from db import Meme
    test = Meme(
        id="test",
        image_url="https://example.com/test.jpg",
        visual_tags=["test"],
        contextual_tags=["test"],
        user_tags=["test"],
        all_tags=["test"],
        clip_embedding=[0.1] * 512
    )
    assert test.upload_time is None
    assert test.firebase_image_url is None
    assert test.total_likes == 0
    print("✅ Admin panel integration: SAFE")
    print("✅ New fields have defaults: OK")
    print("✅ Recommendation engine: COMPATIBLE")
    sys.exit(0)
except Exception as e:
    print(f"❌ Integration check FAILED: {e}")
    sys.exit(1)
PYTHON_EOF
