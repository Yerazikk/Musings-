# pip install sentence-transformers faiss-cpu rank-bm25 spacy
# python -m spacy download en_core_web_sm

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import spacy

# -----------------------------
# 0) spaCy + tokenizer FIRST
# -----------------------------
nlp = spacy.load("en_core_web_sm")

SELF_PRONOUNS = {"i", "me", "my", "mine", "myself"}
CONTENT_POS = {"NOUN", "PROPN", "ADJ"}

def tokenize_content_words(s: str):
    doc = nlp(s.lower())
    tokens = []
    for t in doc:
        if not t.is_alpha:
            continue
        if t.text in SELF_PRONOUNS:          # keep self pronouns
            tokens.append(t.text)
        elif t.pos_ in CONTENT_POS and not t.is_stop:
            tokens.append(t.lemma_)          # keep lemma for nouns/adj/propn
    return tokens

# -----------------------------
# 1) Notes (minimal structure)
# -----------------------------
notes = [
    # 🚗 Fix my car
    {"id": 1, "title": "Auto repair", "text": "Find a good auto repair shop near me."},
    {"id": 2, "title": "Car mods", "text": "That guy turned his car into a PC with custom parts."},
    {"id": 3, "title": "Rivian fan", "text": "I love Rivian electric trucks."},
    {"id": 4, "title": "Gas reminder", "text": "Don’t forget to get gas before Monday."},
    {"id": 5, "title": "Car check", "text": "Check tire pressure for the long drive."},

    # 😐 I'm bored
    {"id": 6, "title": "Weekend idea", "text": "Try rock climbing this weekend."},
    {"id": 7, "title": "Origami hobby", "text": "Learn how to make origami cranes."},
    {"id": 8, "title": "Road trip", "text": "Go for a spontaneous road trip."},
    {"id": 9, "title": "Board games", "text": "Play board games with friends."},
    {"id": 10, "title": "Art museum", "text": "Visit the art museum on Sunday."},

    # 📅 Appointments
    {"id": 11, "title": "Doctor visit", "text": "Doctor appointment Tuesday at 2pm."},
    {"id": 12, "title": "Dentist", "text": "Dentist appointment October 5th."},
    {"id": 13, "title": "Team meeting", "text": "Meeting with project team Friday morning."},
    {"id": 14, "title": "Professor call", "text": "Call with professor about research paper."},
    {"id": 15, "title": "Eye check", "text": "Eye check-up scheduled for next month."},

    # 🍳 What to cook
    {"id": 16, "title": "Chicken dinner", "text": "Recipe for chicken tikka masala."},
    {"id": 17, "title": "Italian pasta", "text": "Make spaghetti carbonara tonight."},
    {"id": 18, "title": "Vegan idea", "text": "Try vegan curry with chickpeas."},
    {"id": 19, "title": "Seafood", "text": "Baked salmon with garlic butter."},
    {"id": 20, "title": "Baking", "text": "Learn to make sourdough bread."},

    # 🍴 Restaurants
    {"id": 21, "title": "Sushi", "text": "Nobu sushi restaurant is amazing."},
    {"id": 22, "title": "Ramen", "text": "New ramen place downtown looks good."},
    {"id": 23, "title": "Italian food", "text": "Emily suggested we try the Italian spot."},
    {"id": 24, "title": "Tacos", "text": "Mexican tacos from the food truck were great."},
    {"id": 25, "title": "Steakhouse", "text": "Steakhouse has a happy hour menu."},

    # 🎬 What to watch
    {"id": 26, "title": "Movie night", "text": "Watch Oppenheimer in theaters."},
    {"id": 27, "title": "Netflix show", "text": "Start the new season of Stranger Things."},
    {"id": 28, "title": "Sci-fi", "text": "Check out that sci-fi show on Netflix."},
    {"id": 29, "title": "Rewatch", "text": "Rewatch The Lord of the Rings trilogy."},
    {"id": 30, "title": "Documentary", "text": "Documentary about Rivian cars on YouTube."},

    # 📝 Random unrelated clutter
    {"id": 31, "title": "Shopping list", "text": "Eggs, milk, bread, detergent."},
    {"id": 32, "title": "Password reminder", "text": "Netflix password is saved in password manager."},
    {"id": 33, "title": "Random thought", "text": "What if cats could understand quantum mechanics?"},
    {"id": 34, "title": "Workout log", "text": "Ran 3 miles today, did 20 push-ups."},
    {"id": 35, "title": "Quote", "text": "Stay hungry, stay foolish."},
    {"id": 36, "title": "Todo", "text": "Do laundry, clean kitchen, call mom."},
    {"id": 37, "title": "Dream log", "text": "I dreamed I was flying over a city made of glass."},
    {"id": 38, "title": "Finance", "text": "Check credit card bill, due on the 15th."},
    {"id": 39, "title": "Gift list", "text": "Ideas: headphones, candle, backpack."},
    {"id": 40, "title": "Joke", "text": "Why don’t programmers like nature? Too many bugs."},

    # ❤️ relationships
    {"id": 41, "title": "Samie flowers", "text": "Samie loves flowers. She always keeps them fresh in a vase."},
    {"id": 42, "title": "Samie relationship", "text": "Samie is my girlfriend and we planned her birthday."},
    {"id": 43, "title": "Emily gifts", "text": "Gift Emily a notebook for her drawings and think about birthday flowers."},
    {"id": 44, "title": "Emily", "text": "Emily loves green."},
    {"id": 45, "title": "Love quotes", "text": "Cute love quotes for anniversaries."},
    {"id": 46, "title": "Debating", "text": "My girlfriend loves debating."},
    {"id": 47, "title": "Reading", "text": "Some women love books."},
]

# -----------------------------
# 2) Build indices (once / on change)
# -----------------------------
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def build_indices():
    # Embed TITLE + TEXT so titles matter in dense search too
    dense_texts = [f"{n['title']}. {n['text']}" for n in notes]
    vecs = model.encode(dense_texts, convert_to_numpy=True, normalize_embeddings=True)
    dim = vecs.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)   # cosine via dot on normalized vectors
    faiss_index.add(vecs)

    # BM25 over content words (with self-pronoun keep)
    tokenized = [tokenize_content_words(f"{n['title']} {n['text']}") for n in notes]
    bm25 = BM25Okapi(tokenized)
    return faiss_index, bm25, vecs

faiss_index, bm25, cached_vecs = build_indices()

# -----------------------------
# 3) Hybrid search
# -----------------------------
def hybrid_search(query, top_k=5, w_dense=0.70, w_lex=0.30):
    # Lexical (BM25) — use the SAME tokenizer as indexing
    q_tokens = tokenize_content_words(query)
    bm_scores = bm25.get_scores(q_tokens)
    bm_scores = bm_scores / (bm_scores.max() if bm_scores.max() > 0 else 1.0)

    # Semantic (FAISS)
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = faiss_index.search(qv, len(notes))  # similarity in [0,1]
    sem_scores = np.zeros(len(notes))
    for score, idx in zip(D[0], I[0]):
        sem_scores[idx] = score

    # Fusion
    final = w_dense * sem_scores + w_lex * bm_scores

    # Rank & return
    order = np.argsort(final)[::-1][:top_k]
    return [
        {
            "id": notes[i]["id"],
            "title": notes[i]["title"],
            "text": notes[i]["text"],
            "score": float(final[i]),
        }
        for i in order
    ]

# -----------------------------
# 4) Example
# -----------------------------
while True:
    s = input("Enter a search: ")
    for q in [s]:
        print(f"\nQuery: {q}")
        for r in hybrid_search(q, top_k=7):
            print(f"- ({r['score']:.3f}) [{r['id']}] {r['title']}: {r['text']}")
