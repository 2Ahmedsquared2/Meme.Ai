from app.ml_models import get_meme_caption, get_text_similarity

# Test with a sample meme image
test_image = "https://i.imgflip.com/30b1gx.jpg"  # Drake meme

print("Testing BLIP-2 caption generation...")
caption = get_meme_caption(test_image)
print(f"Caption: {caption}")

print("\nTesting CLIP text similarity...")
text_options = ["happy", "sad", "funny", "confused"]
scores = get_text_similarity(test_image, text_options)
print(f"Scores: {scores}")

print("\n✅ ML models working!")

