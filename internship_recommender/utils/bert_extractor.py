from sentence_transformers import SentenceTransformer, util

# small model, downloads on first run
MODEL_NAME = "all-MiniLM-L6-v2"

def get_model():
    try:
        model = SentenceTransformer(MODEL_NAME)
        return model
    except Exception as e:
        print("SentenceTransformer model load failed:", e)
        return None

# semantic similarity helper
def semantic_match(job_requirements, resume_text, top_k=5):
    model = get_model()
    if model is None:
        return []
    req_emb = model.encode(job_requirements, convert_to_tensor=True)
    res_emb = model.encode(resume_text, convert_to_tensor=True)
    scores = util.semantic_search(res_emb, req_emb, top_k=top_k)
    # returns top matches; interpret as needed
    return scores
