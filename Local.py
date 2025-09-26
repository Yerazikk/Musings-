from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# database for notes 
documents = [
    {"id": 1, "text": "Nobu is a nice sushi resturant."},
    {"id": 2, "text": "Friend said that when giving someone a present it's important to spend time on the packaging"},
    {"id": 3, "text": "Cute love quotes for anniversaries."},
    {"id": 4, "text": "Emily loves green."},
    {"id": 5, "text": "Doctor appointment August 12 at 1pm"},
    {"id": 6, "text": "Buy the cats a concret slab"},
    {"id": 7, "text": "Do laundry, get milk."},
    {"id": 8, "text": "Gift Emily a notebook for her drawings"},
    {"id": 9, "text": "buy label maker to try for fun"},
    {"id": 10, "text": "Mark loves green"},
    {"id": 11, "text": "Mark is a fan of green"},
    {"id": 12, "text": "Emily likes yellowtail with yozu"},
    {"id": 13, "text": "Samie loves flowers"},
    {"id": 14, "text": "Samie is my girlfriend"},


]


#Embedding creation
#models: all-MiniLM-L6-v2 , all-MiniLM-L12-v2 , paraphrase-MiniLM-L6-v2 (works best) , paraphrase-mpnet-base-v2 , 
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  #insert any model
texts = [doc["text"] for doc in documents]
vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)  # shape: (num_docs, dim)

# FAISS index
dimension = vectors.shape[1]
index = faiss.IndexFlatIP(dimension)  # cosine similarity
index.add(vectors)

print("Finding Match for Question...")
#Search function (hybrid keyword + semantic)
def search(query, top_k=5):
    results = []

    # Keyword match
    keyword_matches = [doc for doc in documents if any(word.lower() in doc["text"].lower() for word in query.split())]

    # Semantic matchpython -m venv venv
    query_vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(query_vector, top_k)
    semantic_matches = [documents[i] for i in I[0]]

    # Merge and deduplicate
    merged = {doc["id"]: doc for doc in keyword_matches + semantic_matches}
    return list(merged.values())

# Example query
query = "girlfriend birthday"
results = search(query)
print("Search results for:", query)
for r in results:
    print("-", r["text"])
