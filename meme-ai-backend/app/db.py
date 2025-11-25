import firebase_admin 
from firebase_admin import credentials, firestore
import os

# Get the directory of this file, then go up to backend root
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_root = os.path.dirname(current_dir)
cred_path = os.path.join(backend_root, "Credentials4.json")

# Debug: print the path to verify it's correct
print(f"🔍 Looking for credentials at: {cred_path}")

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'projectId': 'get-meme-ai'
})

db = firestore.client()