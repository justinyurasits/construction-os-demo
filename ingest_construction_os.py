#!/usr/bin/env python3
"""
ingest_construction_os.py

Reads every .docx, .md, and .xlsx file in the Construction OS folder, extracts
the full text content of each, chunks long documents, and loads everything into
a persistent ChromaDB collection named 'construction_os' stored at
./construction_os_db.

Extraction:
- .docx  -> python-docx (paragraphs + tables)
- .md    -> plain text
- .xlsx  -> openpyxl (all sheets, all non-empty cells)

Skip rules (a file is excluded for the first reason that matches):
- Office lock files (name starts with '~$')
- 'zz' prefix
- 'Logo' anywhere in the filename
- image files (.png, .jpg, .jpeg)
- unsupported extension (not .docx / .md / .xlsx)
- superseded version: within a group of files sharing the same base name but
  different version suffixes (_v1, _v2, ...), only the highest version is kept

Chunking:
- documents > 1000 chars are split into 1000-char chunks with 200-char overlap
- each chunk is its own entry, ID '<stem>-chunk-0', '<stem>-chunk-1', ...
- documents <= 1000 chars stay as a single entry, ID '<stem>'
- metadata stores the source filename and chunk number
"""

import re
import sys
import traceback
from pathlib import Path

# --- Configuration -----------------------------------------------------------

SOURCE_FOLDER = Path(
    "/Users/justinyurasits/Documents/Business Content/"
    "Construction OS/OS Documents/Final OS"
)
DB_PATH = "./construction_os_db"
COLLECTION_NAME = "construction_os"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

SUPPORTED_EXTS = {".docx", ".md", ".xlsx"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# Matches a trailing version suffix like '_v1', '_v12' (case-insensitive).
VERSION_RE = re.compile(r"^(.*?)_v(\d+)$", re.IGNORECASE)

# --- Dependency check --------------------------------------------------------


def ensure_packages():
    """Install required packages if they are not already importable."""
    import importlib
    import subprocess

    required = {  # import name -> pip package name
        "docx": "python-docx",
        "openpyxl": "openpyxl",
        "chromadb": "chromadb",
    }
    for module_name, pip_name in required.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            print(f"Installing missing package: {pip_name} ...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name]
            )


ensure_packages()

import chromadb  # noqa: E402
from docx import Document  # noqa: E402
from openpyxl import load_workbook  # noqa: E402


# --- Extractors --------------------------------------------------------------


def extract_docx(path: Path) -> str:
    """Extract all paragraph and table text from a .docx file."""
    doc = Document(str(path))
    parts = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)

    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            cells = [c for c in cells if c]
            if cells:
                parts.append("\t".join(cells))

    return "\n".join(parts)


def extract_md(path: Path) -> str:
    """Read a .md file as plain text."""
    return path.read_text(encoding="utf-8", errors="replace")


def extract_xlsx(path: Path) -> str:
    """Extract all non-empty cell text from every sheet in a .xlsx file."""
    wb = load_workbook(filename=str(path), read_only=True, data_only=True)
    parts = []

    for sheet in wb.worksheets:
        parts.append(f"# Sheet: {sheet.title}")
        for row in sheet.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None and str(c).strip()]
            if cells:
                parts.append("\t".join(cells))

    wb.close()
    return "\n".join(parts)


EXTRACTORS = {
    ".docx": extract_docx,
    ".md": extract_md,
    ".xlsx": extract_xlsx,
}


# --- Helpers -----------------------------------------------------------------


def parse_version(stem: str):
    """Return (base_name, version_int) for a filename stem.

    'PRE-05_Proposal_Package_v2' -> ('PRE-05_Proposal_Package', 2)
    'Master_Cross_Reference_Index' -> ('Master_Cross_Reference_Index', 0)
    """
    m = VERSION_RE.match(stem)
    if m:
        return m.group(1), int(m.group(2))
    return stem, 0


def chunk_text(text: str):
    """Split text into overlapping chunks. Returns a list with at least one item."""
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP  # 800
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


# --- Main --------------------------------------------------------------------


def main():
    if not SOURCE_FOLDER.is_dir():
        print(f"ERROR: source folder not found: {SOURCE_FOLDER}")
        sys.exit(1)

    all_files = sorted(p for p in SOURCE_FOLDER.iterdir() if p.is_file())
    skipped = []  # (filename, reason)

    # --- Pass 1: per-file skip rules ----------------------------------------
    candidates = []  # files that survive the individual rules
    for path in all_files:
        name = path.name
        ext = path.suffix.lower()

        if name.startswith("~$"):
            skipped.append((name, "Office lock file (~$ prefix)"))
        elif name.startswith("zz"):
            skipped.append((name, "zz prefix"))
        elif "logo" in name.lower():
            skipped.append((name, "'Logo' in filename"))
        elif ext in IMAGE_EXTS:
            skipped.append((name, f"image file ({ext})"))
        elif ext not in SUPPORTED_EXTS:
            skipped.append((name, f"unsupported extension ({ext or 'none'})"))
        else:
            candidates.append(path)

    # --- Pass 2: keep only the highest version within each base-name group ---
    groups = {}  # base_name -> list of (version, path)
    for path in candidates:
        base, version = parse_version(path.stem)
        groups.setdefault(base, []).append((version, path))

    survivors = []
    for base, items in groups.items():
        items.sort(key=lambda t: t[0])  # ascending by version
        winner = items[-1][1]
        survivors.append(winner)
        for version, path in items[:-1]:
            skipped.append(
                (path.name, f"superseded version (kept {winner.name})")
            )

    survivors.sort(key=lambda p: p.name)

    # --- Pass 3: extract, chunk, and ingest ---------------------------------
    client = chromadb.PersistentClient(path=DB_PATH)
    # Recreate the collection for a clean, reproducible re-run.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - fine if it doesn't exist yet
        pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    indexed = 0
    total_chunks = 0
    failed = []  # (filename, error)

    for path in survivors:
        ext = path.suffix.lower()
        try:
            text = EXTRACTORS[ext](path)
            if not text.strip():
                skipped.append((path.name, "no extractable text"))
                continue

            chunks = chunk_text(text)
            stem = path.stem

            if len(chunks) == 1:
                ids = [stem]
            else:
                ids = [f"{stem}-chunk-{i}" for i in range(len(chunks))]

            metadatas = [
                {
                    "source_filename": path.name,
                    "chunk_number": i,
                    "total_chunks": len(chunks),
                    "extension": ext,
                }
                for i in range(len(chunks))
            ]

            collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
            indexed += 1
            total_chunks += len(chunks)
            print(
                f"  [OK]   {path.name}  "
                f"({len(text):,} chars -> {len(chunks)} chunk(s))"
            )
        except Exception as exc:  # noqa: BLE001
            failed.append((path.name, f"{type(exc).__name__}: {exc}"))
            print(f"  [FAIL] {path.name}  -> {type(exc).__name__}: {exc}")
            traceback.print_exc()

    # --- Summary -------------------------------------------------------------
    print("\n" + "=" * 64)
    print("INGEST SUMMARY")
    print("=" * 64)
    print(f"Collection           : '{COLLECTION_NAME}' at {DB_PATH}")
    print(f"Total source files   : {len(all_files)}")
    print(f"Files indexed        : {indexed}")
    print(f"Files skipped        : {len(skipped)}")
    print(f"Files failed         : {len(failed)}")
    print(f"Total chunks created : {total_chunks}")
    print(f"Entries in DB        : {collection.count()}")

    if skipped:
        print(f"\nSkipped files ({len(skipped)}):")
        for name, reason in sorted(skipped):
            print(f"  - {name}: {reason}")

    if failed:
        print(f"\nFailed files ({len(failed)}):")
        for name, err in failed:
            print(f"  - {name}: {err}")

    print("=" * 64)


if __name__ == "__main__":
    main()
