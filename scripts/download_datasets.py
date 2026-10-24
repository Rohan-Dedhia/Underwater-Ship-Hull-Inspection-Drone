"""Download the public datasets from Mendeley Data (CC BY 4.0) with SHA-256 verification.

- MaVeCoDD (Kamzelis et al., doi:10.17632/ry392rp8cj.1) -> data/raw/mavecodd/{HiRes,LoRes}/{raw,labeled}
- Underwater Crack (Gao et al., doi:10.17632/g962rbm77b.1) -> under-crack-images.zip in the repo root

Usage:
    python scripts/download_datasets.py
    python scripts/prepare_dataset.py
"""

import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://data.mendeley.com/public-api/datasets/{}"
HEADERS = {"User-Agent": "curl/8.0"}  # Mendeley rejects Python's default user agent

# Mendeley folder ids of MaVeCoDD v1 -> local sub-folder
MAVECODD_FOLDERS = {
    "ef052014-c074-45bf-9418-e6ea3b67ddba": "HiRes/raw",
    "ab22b882-59c7-4019-8184-fb6144aeea06": "HiRes/labeled",
    "56d13fa4-6232-4a5b-9e14-4d52ff7a941c": "LoRes/raw",
    "99511f67-be99-43a5-a83a-ecb237ae4087": "LoRes/labeled",
}


def _get(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=300).read()


def _fetch(file_meta: dict, dest: Path) -> str:
    want = file_meta["content_details"]["sha256_hash"]
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == want:
        return "present"
    data = _get(file_meta["content_details"]["download_url"])
    if hashlib.sha256(data).hexdigest() != want:
        raise RuntimeError(f"SHA-256 mismatch for {file_meta['filename']}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return "downloaded"


def download_mavecodd() -> None:
    meta = json.loads(_get(API.format("ry392rp8cj")))
    out = ROOT / "data" / "raw" / "mavecodd"
    jobs = [(f, out / MAVECODD_FOLDERS[f["folder_id"]] / f["filename"])
            for f in meta["files"] if f.get("folder_id") in MAVECODD_FOLDERS]
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda j: _fetch(*j), jobs))
    print(f"MaVeCoDD: {results.count('downloaded')} downloaded, {results.count('present')} already present "
          f"-> {out}")


def download_underwater_crack() -> None:
    meta = json.loads(_get(API.format("g962rbm77b")))
    f = next(f for f in meta["files"] if f["filename"] == "under-crack-images.zip")
    status = _fetch(f, ROOT / "under-crack-images.zip")
    print(f"Underwater Crack: {status} -> {ROOT / 'under-crack-images.zip'}")


if __name__ == "__main__":
    try:
        download_mavecodd()
        download_underwater_crack()
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)
    print("Done. Next: python scripts/prepare_dataset.py")
