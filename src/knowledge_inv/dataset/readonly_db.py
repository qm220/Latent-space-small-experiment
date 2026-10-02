"""Read-only access to the neuralCAD-Edit Mongita database.

This module does not import the official DatabaseManager. That class creates
extra directories under the dataset root and includes mutating cleanup helpers.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from mongita import MongitaClientDisk

from knowledge_inv.paths import load_project_config, mongita_dir, storage_dir

STANDARD_VIEWS = (
    "front",
    "back",
    "left",
    "right",
    "top",
    "bottom",
    "toprightiso",
)
CAD_EXTENSIONS = ("step", "stl", "f3d", "smt")
BREP_OMIT = {"feature_dino": 0}


def view_label_from_path(relative_path: str) -> str:
    stem = Path(relative_path).stem
    return stem.rsplit("_", 1)[-1]


def _nonempty(value: Any) -> bool:
    return value not in (None, "", [], {})


class ReadOnlyNeuralCAD:
    def __init__(self, config: dict | None = None):
        self.config = config or load_project_config()
        self.root = storage_dir(self.config)
        self.db_path = mongita_dir(self.config)
        self.db_name = self.config["db_name"]
        if not self.db_path.is_dir():
            raise FileNotFoundError(f"Mongita directory not found: {self.db_path}")
        self.client = MongitaClientDisk(host=str(self.db_path))
        self.db = self.client[self.db_name]
        self.users = self.db["users"]
        self.breps = self.db["breps"]
        self.requests = self.db["requests"]
        self.edits = self.db["edits"]
        self.ratings = self.db["ratings"]

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "ReadOnlyNeuralCAD":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def resolve(self, relative_path: str | None) -> Path | None:
        if not relative_path:
            return None
        return (self.root / relative_path).resolve()

    def user_map(self) -> dict[str, dict]:
        return {doc["_id"]: doc for doc in self.users.find()}

    def get_request(self, request_id: str) -> dict | None:
        return self.requests.find_one({"_id": request_id})

    def get_brep(self, brep_id: str, omit_features: bool = True) -> dict | None:
        projection = BREP_OMIT if omit_features else None
        try:
            return self.breps.find_one({"_id": brep_id}, projection)
        except Exception:
            doc = self.breps.find_one({"_id": brep_id})
            if doc and omit_features:
                doc.pop("feature_dino", None)
            return doc

    def request_edits(self, request_id: str) -> list[dict]:
        return list(self.edits.find({"request": request_id}))

    def instruction_fields(self, request: dict) -> dict[str, Any]:
        text = (request.get("text") or "").strip()
        prompt = request.get("prompt")
        raw_instructions = request.get("instructions")
        transcript = request.get("transcript")
        corrected = request.get("corrected_transcript_segments")
        modality = request.get("modality")
        genuine_text = modality == "text" and bool(text)
        return {
            "modality": modality,
            "text": text or None,
            "prompt": prompt if _nonempty(prompt) else None,
            "instructions_field": raw_instructions if _nonempty(raw_instructions) else None,
            "has_transcript": isinstance(transcript, dict) and bool(transcript),
            "has_corrected_transcript": bool(corrected),
            "genuine_text_instruction": genuine_text,
            "instruction_text": text if genuine_text else None,
            "instruction_source": "requests.text" if genuine_text else None,
        }

    def brep_assets(self, brep_id: str) -> dict[str, Any]:
        brep = self.get_brep(brep_id)
        if not brep:
            return {"brep_id": brep_id, "found": False, "images": {}, "cad": {}, "missing": ["brep_record"]}

        images: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for rel in brep.get("jpg") or []:
            label = view_label_from_path(rel)
            path = self.resolve(rel)
            exists = bool(path and path.is_file())
            images[label] = {
                "relative_path": rel,
                "absolute_path": str(path) if path else None,
                "exists": exists,
            }
            if not exists:
                missing.append(f"image:{rel}")

        cad: dict[str, dict[str, Any] | None] = {}
        for ext in CAD_EXTENSIONS:
            items = brep.get(ext) or []
            if not items:
                cad[ext] = None
                missing.append(f"cad_field:{ext}")
                continue
            rel = items[0]
            path = self.resolve(rel)
            exists = bool(path and path.is_file())
            cad[ext] = {
                "relative_path": rel,
                "absolute_path": str(path) if path else None,
                "exists": exists,
            }
            if not exists:
                missing.append(f"cad_file:{rel}")

        return {
            "brep_id": brep_id,
            "found": True,
            "user": brep.get("user"),
            "images": images,
            "cad": cad,
            "missing": missing,
            "available_views": sorted(k for k, v in images.items() if v["exists"]),
        }

    def classify_edit(self, edit: dict, request: dict, users: dict[str, dict] | None = None) -> dict:
        users = users or self.user_map()
        user_id = edit.get("user")
        user = users.get(user_id, {})
        if user_id == request.get("user"):
            role = "request_author_edit"
        elif user.get("is_human"):
            role = "other_human_edit"
        else:
            role = "model_edit"
        return {
            "edit_id": edit.get("_id"),
            "user": user_id,
            "role": role,
            "is_human": bool(user.get("is_human")),
            "brep_end": edit.get("brep_end"),
            "frames_dir": edit.get("frames_dir"),
            "evaluation_only": True,
            "do_not_show_to_model": True,
        }


@contextmanager
def open_readonly_db(config: dict | None = None) -> Iterator[ReadOnlyNeuralCAD]:
    db = ReadOnlyNeuralCAD(config)
    try:
        yield db
    finally:
        db.close()
