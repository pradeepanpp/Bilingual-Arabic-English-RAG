"""
Download the English evaluation document (Attention Is All You Need, arXiv v7)
into data/docs/. The Arabic UAE labour law PDF is already in the repo.

Run:  python scripts/download_data.py
"""

import sys
import urllib.request
from pathlib import Path

URL = "https://arxiv.org/pdf/1706.03762v7"
TARGET = Path("data/docs/attention.pdf")
EXPECTED_PAGES = 15  # the page numbers in data/eval/questions.json refer to this version


def main():
    if TARGET.exists():
        print(f"{TARGET} already exists")
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    TARGET.write_bytes(urllib.request.urlopen(request, timeout=60).read())

    import pymupdf
    with pymupdf.open(TARGET) as pdf:
        if len(pdf) != EXPECTED_PAGES:
            sys.exit(f"Expected {EXPECTED_PAGES} pages, got {len(pdf)}: wrong version of the paper?")
    print(f"Saved {TARGET}")


if __name__ == "__main__":
    main()
