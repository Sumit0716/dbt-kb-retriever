from sentence_transformers import SentenceTransformer

print("⏬ Downloading model from Hugging Face...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("💾 Saving locally...")
model.save("./models/all-MiniLM-L6-v2")

print("✅ Model saved to ./models/all-MiniLM-L6-v2")
