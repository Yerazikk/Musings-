# main.py
# pip install sentence-transformers faiss-cpu rank-bm25 spacy
# python -m spacy download en_core_web_sm

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import spacy
import shlex

# --- your DB layer ---
from database import NoteDB  # <- file you created with Note/NoteDB (dict-backed, has edit/get/as_dicts)

# -----------------------------
# 0) spaCy + tokenizer FIRST
# -----------------------------
nlp = spacy.load("en_core_web_sm")

SELF_PRONOUNS = {"i", "me", "my", "mine", "myself"}
CONTENT_POS = {"NOUN", "PROPN", "ADJ"}

def tokenize_content_words(s: str | None):
    s = s or ""  # handle None text
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
# 1) Initialize database (seed optional)
# -----------------------------
db = NoteDB()

# seed notes (ids will be assigned by NoteDB; we ignore the "id" field in the seed rows)
_initial_seed =  [
    {"title": "Auto repair", "text": "Find a good auto repair shop near me."},
    {"title": "Car mods", "text": "That guy turned his car into a PC with custom parts."},
    {"title": "Rivian fan", "text": "I love Rivian electric trucks."},
    {"title": "Gas reminder", "text": "Don’t forget to get gas before Monday."},
    {"title": "Car check", "text": "Check tire pressure for the long drive."},

    {"title": "Weekend idea", "text": "Try rock climbing this weekend."},
    {"title": "Origami hobby", "text": "Learn how to make origami cranes."},
    {"title": "Road trip", "text": "Go for a spontaneous road trip."},
    {"title": "Board games", "text": "Play board games with friends."},
    {"title": "Art museum", "text": "Visit the art museum on Sunday."},

    {"title": "Doctor visit", "text": "Doctor appointment Tuesday at 2pm."},
    {"title": "Dentist", "text": "Dentist appointment October 5th."},
    {"title": "Team meeting", "text": "Meeting with project team Friday morning."},
    {"title": "Professor call", "text": "Call with professor about research paper."},
    {"title": "Eye check", "text": "Eye check-up scheduled for next month."},

    {"title": "Chicken dinner", "text": "Recipe for chicken tikka masala."},
    {"title": "Italian pasta", "text": "Make spaghetti carbonara tonight."},
    {"title": "Vegan idea", "text": "Try vegan curry with chickpeas."},
    {"title": "Seafood", "text": "Baked salmon with garlic butter."},
    {"title": "Baking", "text": "Learn to make sourdough bread."},

    {"title": "Sushi", "text": "Nobu sushi restaurant is amazing."},
    {"title": "Ramen", "text": "New ramen place downtown looks good."},
    {"title": "Italian food", "text": "Emily suggested we try the Italian spot."},
    {"title": "Tacos", "text": "Mexican tacos from the food truck were great."},
    {"title": "Steakhouse", "text": "Steakhouse has a happy hour menu."},

    {"title": "Movie night", "text": "Watch Oppenheimer in theaters."},
    {"title": "Netflix show", "text": "Start the new season of Stranger Things."},
    {"title": "Sci-fi", "text": "Check out that sci-fi show on Netflix."},
    {"title": "Rewatch", "text": "Rewatch The Lord of the Rings trilogy."},
    {"title": "Documentary", "text": "Documentary about Rivian cars on YouTube."},

    {"title": "Shopping list", "text": "Eggs, milk, bread, detergent."},
    {"title": "Password reminder", "text": "Netflix password is saved in password manager."},
    {"title": "Random thought", "text": "What if cats could understand quantum mechanics?"},
    {"title": "Workout log", "text": "Ran 3 miles today, did 20 push-ups."},
    {"title": "Quote", "text": "Stay hungry, stay foolish."},
    {"title": "Todo", "text": "Do laundry, clean kitchen, call mom."},
    {"title": "Dream log", "text": "I dreamed I was flying over a city made of glass."},
    {"title": "Finance", "text": "Check credit card bill, due on the 15th."},
    {"title": "Gift list", "text": "Ideas: headphones, candle, backpack."},
    {"title": "Joke", "text": "Why don’t programmers like nature? Too many bugs."},

    {"title": "Samie flowers", "text": "Samie loves flowers. She always keeps them fresh in a vase."},
    {"title": "Samie relationship", "text": "Samie is my girlfriend and we planned her birthday."},
    {"title": "Emily gifts", "text": "Gift Emily a notebook for her drawings and think about birthday flowers."},
    {"title": "Emily", "text": "Emily loves green."},
    {"title": "Love quotes", "text": "Cute love quotes for anniversaries."},
    {"title": "Debating", "text": "My girlfriend loves debating."},
    {"title": "Reading", "text": "Some women love books."},
]
for n in _initial_seed:
    db.add(title=n.get("title"), text=n.get("text"))

# -----------------------------
# 2) Indices (incremental strategy)
# -----------------------------
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Global state
faiss_index = None                # faiss.IndexIDMap2 over IndexFlatIP
bm25 = None                       # BM25Okapi over token lists
tokenized_docs_by_id = {}         # {note_id: [tokens, ...]}

def build_indices_from_db():
    """Bootstrap FAISS (with ids) and BM25 from current DB."""
    global faiss_index, bm25, tokenized_docs_by_id

    data = db.as_dicts()  # [{"id","title","text"}, ...]
    dim = model.get_sentence_embedding_dimension()

    base = faiss.IndexFlatIP(dim)
    faiss_index = faiss.IndexIDMap2(base)

    if not data:
        bm25 = BM25Okapi([[]])
        tokenized_docs_by_id = {}
        return

    dense_texts = [f"{n['title']}. {n['text']}" for n in data]
    vecs = model.encode(dense_texts, convert_to_numpy=True, normalize_embeddings=True)
    ids = np.array([n["id"] for n in data], dtype=np.int64)
    faiss_index.add_with_ids(vecs, ids)

    tokenized_docs_by_id = {
        n["id"]: tokenize_content_words(f"{n['title']} {n['text']}")
        for n in data
    }
    _rebuild_bm25()

def _rebuild_bm25():
    global bm25
    # BM25 expects a list-of-token-lists; use the dict's insertion order
    bm25 = BM25Okapi(list(tokenized_docs_by_id.values()))

def add_note_incremental(note_dict: dict):
    """Add a single note to both FAISS and BM25."""
    dense_text = f"{note_dict['title']}. {note_dict.get('text') or ''}"
    vec = model.encode([dense_text], convert_to_numpy=True, normalize_embeddings=True)
    faiss_index.add_with_ids(vec, np.array([note_dict["id"]], dtype=np.int64))

    tokens = tokenize_content_words(f"{note_dict['title']} {note_dict.get('text') or ''}")
    tokenized_docs_by_id[note_dict["id"]] = tokens
    _rebuild_bm25()

def delete_note_incremental(note_id: int):
    """Remove a single note from both FAISS and BM25."""
    faiss_index.remove_ids(np.array([note_id], dtype=np.int64))
    tokenized_docs_by_id.pop(note_id, None)
    _rebuild_bm25()

def edit_note_incremental(note_dict: dict):
    """Update a single note in both FAISS and BM25 (remove + add)."""
    nid = note_dict["id"]
    faiss_index.remove_ids(np.array([nid], dtype=np.int64))

    dense_text = f"{note_dict['title']}. {note_dict.get('text') or ''}"
    vec = model.encode([dense_text], convert_to_numpy=True, normalize_embeddings=True)
    faiss_index.add_with_ids(vec, np.array([nid], dtype=np.int64))

    tokenized_docs_by_id[nid] = tokenize_content_words(f"{note_dict['title']} {note_dict.get('text') or ''}")
    _rebuild_bm25()

build_indices_from_db()

# -----------------------------
# 3) Hybrid search (uses ID-mapped FAISS)
# -----------------------------
def hybrid_search(query, top_k=5, w_dense=0.70, w_lex=0.30):
    if faiss_index is None or len(tokenized_docs_by_id) == 0:
        return []

    # Lexical
    id_list = list(tokenized_docs_by_id.keys())   # BM25 corpus order
    q_tokens = tokenize_content_words(query)

    bm_scores = bm25.get_scores(q_tokens) if len(id_list) > 0 else np.zeros(0)
    bm_scores = bm_scores / (bm_scores.max() if len(bm_scores) and bm_scores.max() > 0 else 1.0)

    # Semantic (FAISS); returns ids because we used IndexIDMap2
    N = len(id_list)
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = faiss_index.search(qv, N)   # ask up to N

    id_to_pos = {nid: i for i, nid in enumerate(id_list)}
    sem_scores = np.zeros(N, dtype=np.float32)
    for score, nid in zip(D[0], I[0]):
        if int(nid) in id_to_pos:
            sem_scores[id_to_pos[int(nid)]] = score

    # Fusion (aligned by id_list order)
    final = w_dense * sem_scores + w_lex * (bm_scores if len(bm_scores) == N else np.zeros(N))

    order = np.argsort(final)[::-1][:top_k]
    results = []
    for pos in order:
        nid = id_list[pos]
        note = db.get(nid)
        if note:
            results.append({
                "id": note.id,
                "title": note.title,
                "text": note.text or "",
                "score": float(final[pos]),
            })
    return results

# -----------------------------
# 4) Simple CLI with add/del/edit/list/search (incremental)
# -----------------------------
print('Commands: add ["Title" ["Text"]] | del <id> | edit <id> ["Title"] ["Text"] | list | search <query> | quit')

while True:
    s = input("> ").strip()
    if not s:
        continue

    cmd = s.split()[0].lower()

    if cmd in {"quit", "exit", "q"}:
        break

    elif cmd == "list":
        for n in db:
            print(f'[{n.id}] {n.title}: {n.text if n.text is not None else "(null)"}')

    elif cmd == "add":
        try:
            parts = shlex.split(s)
            # forms:
            #   add
            #   add "Title"
            #   add "Title" "Text"
            title = None
            text = None
            if len(parts) >= 2:
                title = parts[1]
            if len(parts) >= 3:
                text = " ".join(parts[2:])
            nid = db.add(title=title, text=text)  # title defaults to "Untitled"; text can be None

            # incremental index update
            note = db.get(nid)
            add_note_incremental({"id": note.id, "title": note.title, "text": note.text})
            print(f"Added note id={nid}.")
        except Exception as e:
            print(f"Add failed: {e}")

    elif cmd == "del":
        try:
            _, sid = s.split(maxsplit=1)
            nid = int(sid)

            # delete from db first; if not present, no-op
            exists = db.get(nid) is not None
            db.delete(nid)
            if exists:
                delete_note_incremental(nid)
                print(f"Deleted note id={nid}.")
            else:
                print(f"No note with id={nid}.")
        except Exception as e:
            print(f"Delete failed: {e}")

    elif cmd == "edit":
        try:
            parts = shlex.split(s)
            # forms:
            #   edit 12
            #   edit 12 "New Title"
            #   edit 12 "New Title" "New Text"
            if len(parts) < 2:
                raise ValueError('Usage: edit <id> ["Title"] ["Text"]')
            nid = int(parts[1])
            title = None
            text = None
            if len(parts) >= 3:
                title = parts[2]
            if len(parts) >= 4:
                text = " ".join(parts[3:])

            ok = db.edit(nid, title=title, text=text)
            if not ok:
                print(f"No note with id={nid}.")
            else:
                note = db.get(nid)  # fetch updated
                edit_note_incremental({"id": note.id, "title": note.title, "text": note.text})
                print(f"Edited note id={nid}.")
        except Exception as e:
            print(f"Edit failed: {e}")

    else:
        # search <query> or raw query
        query = s[7:].strip() if s.lower().startswith("search ") else s
        print(f"\nQuery: {query}")
        for r in hybrid_search(query, top_k=7):
            print(f"- ({r['score']:.3f}) [{r['id']}] {r['title']}: {r['text']}")
