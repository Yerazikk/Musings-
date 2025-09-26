# notes_db.py
from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class Note:
    id: int
    title: str = "Untitled"
    text: Optional[str] = None

class NoteDB:
    def __init__(self):
        self.notes: Dict[int, Note] = {}
        self._next_id = 1

    def add(self, title: Optional[str] = None, text: Optional[str] = None) -> int:
        nid = self._next_id
        self._next_id += 1
        self.notes[nid] = Note(id=nid, title=title or "Untitled", text=text)
        return nid

    def delete(self, note_id: int) -> None:
        self.notes.pop(note_id, None)

    def get(self, note_id: int) -> Optional[Note]:
        return self.notes.get(note_id)

    def edit(self, note_id: int, title: Optional[str] = None, text: Optional[str] = None) -> bool:
        """Edit a note. Returns True if updated, False if not found."""
        note = self.notes.get(note_id)
        if not note:
            return False
        if title is not None:
            note.title = title or "Untitled"
        if text is not None:
            note.text = text
        return True

    def as_dicts(self) -> List[dict]:
        return [
            {"id": n.id, "title": n.title, "text": n.text or ""}
            for n in self.notes.values()
        ]

    def clear(self) -> None:
        """Remove all notes and reset id counter."""
        self.notes.clear()
        self._next_id = 1

    def bulk_add(self, rows: list[dict]) -> list[int]:
        """Add many notes (list of {'title','text'}) and return their ids."""
        ids = []
        for r in rows:
            ids.append(self.add(title=r.get("title"), text=r.get("text")))
        return ids

    def count(self) -> int:
        return len(self.notes)

    def __len__(self): 
        return len(self.notes)

    def __iter__(self):
        return iter(self.notes.values())
