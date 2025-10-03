import os, certifi, json, faiss, numpy as np
from pathlib import Path
from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer

# ----------------------------
# SSL FIX (works without admin rights)
# ----------------------------
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

# ----------------------------
# Globals for KB + FAISS
# ----------------------------
KB_PATH = Path("dbt_kb.json")
MODEL_DIR = Path("./models/all-MiniLM-L6-v2")

kb = []
index = None
model = None

# ----------------------------
# Model Loader
# ----------------------------
def load_model():
    if not MODEL_DIR.exists():
        raise FileNotFoundError("❌ Model directory not found. Please download model files into ./models/all-MiniLM-L6-v2")
    try:
        print(f"📂 Trying to load SentenceTransformer from {MODEL_DIR}")
        return SentenceTransformer(str(MODEL_DIR))
    except Exception as e:
        print(f"⚠️ Could not load as SentenceTransformer ({e}). Falling back to AutoModel + Pooling...")
        word_embedding_model = models.Transformer(str(MODEL_DIR))
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        return SentenceTransformer(modules=[word_embedding_model, pooling_model])

# ----------------------------
# KB + FAISS Builder
# ----------------------------
def build_index():
    global kb, index, model
    if not KB_PATH.exists():
        raise FileNotFoundError("❌ dbt_kb.json not found. Please generate it first.")

    with open(KB_PATH) as f:
        kb = json.load(f)

    texts = [rec["text"] for rec in kb]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(embeddings)

    print(f"✅ FAISS index rebuilt with {len(kb)} chunks")

# ----------------------------
# Init Model + KB
# ----------------------------
model = load_model()
build_index()

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="DBT Knowledge Base Search API")

@app.get("/search")
def search(q: str = Query(..., description="Search query"), k: int = 5):
    """Semantic search over dbt knowledge base."""
    q_emb = model.encode([q], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, k)

    results = []
    for idx, score in zip(I[0], D[0]):
        rec = kb[idx]
        results.append({
            "score": float(score),
            "id": rec["id"],
            "object_type": rec["object_type"],
            "name": rec["name"],
            "text": rec["text"],
            "metadata": rec["metadata"]
        })

    return {"query": q, "results": results}

@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok", "kb_chunks": len(kb)}

@app.post("/rebuild")
def rebuild():
    """Rebuild FAISS index from latest dbt_kb.json without restart."""
    build_index()
    return {"status": "reloaded", "kb_chunks": len(kb)}
