from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any 
from seeds import PREMADE_NOTES
from database import NoteDB
from search_index import SearchIndex

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod, set your frontend domain(s)
    allow_methods=["*"],
    allow_headers=["*"],
)

# init DB + index
db = NoteDB()
idx = SearchIndex()
idx.build_from_db(db)

def rebuild_index_from_db():
    """Full rebuild (used for bulk ops like seed/wipe/reset)."""
    idx.build_from_db(db)

# --- Pydantic models (request/response) ---
class NoteIn(BaseModel):
    title: Optional[str] = None
    text: Optional[str] = None

class NoteOut(BaseModel):
    id: int
    title: str
    text: Optional[str] = None

@app.get("/notes", response_model=List[NoteOut])
def list_notes():
    return [{"id": n.id, "title": n.title, "text": n.text} for n in db]

@app.post("/notes", response_model=NoteOut)
def add_note(body: NoteIn):
    nid = db.add(title=body.title, text=body.text)
    note = db.get(nid)
    idx.add({"id": note.id, "title": note.title, "text": note.text})
    return {"id": note.id, "title": note.title, "text": note.text}

@app.patch("/notes/{note_id}", response_model=NoteOut)
def edit_note(note_id: int, body: NoteIn):
    ok = db.edit(note_id, title=body.title, text=body.text)
    if not ok:
        raise HTTPException(404, "Note not found")
    note = db.get(note_id)
    idx.edit({"id": note.id, "title": note.title, "text": note.text})
    return {"id": note.id, "title": note.title, "text": note.text}

@app.delete("/notes/{note_id}")
def delete_note(note_id: int):
    existed = db.get(note_id) is not None
    db.delete(note_id)
    if existed:
        idx.delete(note_id)
    return {"ok": True, "deleted": existed}

@app.get("/search")
def search(q: str = Query(..., min_length=1), top_k: int = 7):
    return idx.search(q, top_k=top_k, resolver=db.get)

@app.get("/admin/stats")
def stats():
    return {"count": db.count()}

@app.post("/admin/seed")
def admin_seed(notes: Optional[List[Dict[str, Any]]] = Body(None)):
    """
    Prefill the DB with premade notes (or a provided list).
    Does not wipe first; use /admin/reset for wipe+seed.
    """
    rows = notes if notes is not None else PREMADE_NOTES
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise HTTPException(400, "Body must be a list of objects with title/text.")
    ids = db.bulk_add(rows)
    rebuild_index_from_db()
    return {"ok": True, "added": len(ids), "total": db.count()}

@app.post("/admin/wipe")
def admin_wipe():
    """Remove all notes and reset ids/index."""
    db.clear()
    rebuild_index_from_db()  # SearchIndex handles empty corpus safely (Option A)
    return {"ok": True, "total": db.count()}

@app.post("/admin/reset")
def admin_reset(notes: Optional[List[Dict[str, Any]]] = Body(None)):
    """
    Wipe the DB then seed it (default: PREMADE_NOTES or custom body).
    """
    db.clear()
    rows = notes if notes is not None else PREMADE_NOTES
    db.bulk_add(rows)
    rebuild_index_from_db()
    return {"ok": True, "total": db.count()}