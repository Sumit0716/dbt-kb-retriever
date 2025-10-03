import os, certifi, json, faiss, numpy as np
from pathlib import Path
from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer

# ----------------------------
# SSL FIX (works without admin rights)
# ----------------------------
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

# ----------------------------
# Globals for KB + FAISS
# ----------------------------
KB_PATH = Path("dbt_kb.json")

kb = []
index = None
model = None

# ----------------------------
# Model Loader (always from Hugging Face)
# ----------------------------
def load_model():
    print("📂 Loading Hugging Face model (auto-download if not cached)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Model loaded successfully")
    return model

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
