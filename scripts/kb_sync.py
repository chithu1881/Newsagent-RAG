"""
Move the knowledge base (data_chroma/chroma) in and out of kb.zip - the file GitHub stores on the
"kb-data" branch for the cloud app.

    venv\\Scripts\\python -m scripts.kb_sync pack kb.zip      # database folder -> kb.zip
    venv\\Scripts\\python -m scripts.kb_sync unpack kb.zip    # kb.zip -> database folder (replaces it)
    venv\\Scripts\\python -m scripts.kb_sync pull             # download the cloud copy to this computer

The GitHub Actions collector does unpack -> collect -> pack on every run.
"""

import shutil
import sys
import zipfile
from pathlib import Path

from rag.store import KB_URL, LOCAL_DIR


def pack(zip_path):
    zip_path = Path(zip_path).resolve()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in Path(LOCAL_DIR).rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(LOCAL_DIR))
    print(f"packed {LOCAL_DIR} -> {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


def unpack(zip_path):
    shutil.rmtree(LOCAL_DIR, ignore_errors=True)
    Path(LOCAL_DIR).mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(LOCAL_DIR)
    print(f"unpacked {zip_path} -> {LOCAL_DIR}")


def pull():
    import requests
    r = requests.get(KB_URL, timeout=300)
    r.raise_for_status()
    tmp = Path(LOCAL_DIR).parent / "kb_download.zip"
    tmp.write_bytes(r.content)
    unpack(tmp)
    tmp.unlink()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "pack" and len(sys.argv) == 3:
        pack(sys.argv[2])
    elif command == "unpack" and len(sys.argv) == 3:
        unpack(sys.argv[2])
    elif command == "pull":
        pull()
    else:
        sys.exit(__doc__)
