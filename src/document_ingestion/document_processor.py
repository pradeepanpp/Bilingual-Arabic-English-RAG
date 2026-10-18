"""
Load documents (PDFs, URLs, text files) and split them into chunks.

PDF pages are read with PyMuPDF. A page goes to OCR (EasyOCR, Arabic +
English) only when its text layer is missing or unusable; see
text_quality.ocr_reason(). OCR is slow on a CPU (tens of seconds per page),
so OCR results are cached in data/cache/ per file and page.
"""

import hashlib
import json
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.document_ingestion.text_quality import clean_text, ocr_reason

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
OCR_DPI = 200
# Sentence ends come before line breaks, so a chunk does not stop in the middle of a
# sentence (PDF lines break mid-sentence). ". " rather than "." keeps numbers like 0.1 whole.
SEPARATORS = ["\n\n", ". ", ".\n", "؟", "?", "\n", "،", ",", " ", ""]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


class OCRCache:
    """OCR text per page, stored as data/cache/<file hash>.json."""

    def __init__(self, pdf_path: Path, cache_dir: Path = CACHE_DIR):
        self.path = Path(cache_dir) / f"{file_hash(pdf_path)}.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.data.setdefault("file", pdf_path.name)
        self.data.setdefault("dpi", OCR_DPI)
        self.data.setdefault("pages", {})

    def get(self, page_index: int):
        return self.data["pages"].get(str(page_index))

    def put(self, page_index: int, text: str):
        self.data["pages"][str(page_index)] = text
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, use_ocr: bool = True,
                 cache_dir: Path = CACHE_DIR):
        self.use_ocr = use_ocr
        self.cache_dir = Path(cache_dir)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=SEPARATORS,
            keep_separator="end",
        )
        self.ocr_reader = None  # EasyOCR loads ~300 MB of models, so only when needed

    # PDFs

    def load_pdf(self, path) -> list[Document]:
        """One Document per page, with source, page (0-based) and extraction method."""
        import pymupdf

        path = Path(path)
        cache = OCRCache(path, self.cache_dir) if self.use_ocr else None
        docs = []
        with pymupdf.open(path) as pdf:
            for i, page in enumerate(pdf):
                text, method = clean_text(page.get_text()), "text"
                reason = ocr_reason(text)
                if reason and cache is not None:
                    ocr_text = self.ocr_page(page, i, cache, reason)
                    if ocr_text:
                        text, method = clean_text(ocr_text), "ocr"
                if text:
                    docs.append(Document(page_content=text,
                                         metadata={"source": path.name, "page": i, "method": method}))
        return docs

    def ocr_page(self, page, page_index: int, cache: OCRCache, reason: str) -> str:
        cached = cache.get(page_index)
        if cached is not None:
            return cached
        logger.info(f"OCR page {page_index + 1} of {cache.data['file']} ({reason})")
        try:
            import pymupdf
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(OCR_DPI / 72, OCR_DPI / 72), alpha=False)
            # EasyOCR accepts a file path, bytes or a numpy array, not a PIL image
            lines = self.get_ocr_reader().readtext(pixmap.tobytes("png"), detail=0, paragraph=True)
        except Exception as e:
            logger.warning(f"OCR failed on page {page_index + 1} of {cache.data['file']}: {e}")
            return ""
        text = "\n".join(line.strip() for line in lines if line.strip())
        cache.put(page_index, text)
        return text

    def get_ocr_reader(self):
        if self.ocr_reader is None:
            import easyocr
            self.ocr_reader = easyocr.Reader(["ar", "en"], verbose=False)  # uses a GPU if there is one
        return self.ocr_reader

    # Other sources

    def load_url(self, url: str) -> list[Document]:
        from langchain_community.document_loaders import WebBaseLoader

        docs = WebBaseLoader(url).load()
        for d in docs:
            d.page_content = clean_text(d.page_content)
            d.metadata = {"source": url, "title": d.metadata.get("title", ""), "method": "web"}
        return docs

    def load_text(self, path) -> list[Document]:
        path = Path(path)
        return [Document(page_content=clean_text(path.read_text(encoding="utf-8")),
                         metadata={"source": path.name, "method": "text"})]

    def load_sources(self, sources: list) -> list[Document]:
        """
        Sources can be URLs, PDF or .txt files, folders of PDFs, or url.txt
        files with one URL per line. A source that fails is skipped with a
        warning, so one dead link does not stop the app.
        """
        docs = []
        for src in sources:
            src = str(src)
            try:
                if src.startswith(("http://", "https://")):
                    docs += self.load_url(src)
                elif Path(src).is_dir():
                    for pdf in sorted(Path(src).glob("*.pdf")):
                        docs += self.load_pdf(pdf)
                elif Path(src).name == "url.txt":
                    docs += self.load_sources(read_url_list(src))
                elif src.lower().endswith(".pdf"):
                    docs += self.load_pdf(src)
                elif src.lower().endswith(".txt"):
                    docs += self.load_text(src)
                else:
                    logger.warning(f"Skipped unsupported source: {src}")
            except Exception as e:
                logger.warning(f"Could not load {src}: {e}")
        return docs

    def split(self, docs: list[Document]) -> list[Document]:
        return self.splitter.split_documents(docs)


def read_url_list(path) -> list[str]:
    """URLs from a text file; blank lines and # comments are ignored."""
    if not Path(path).exists():
        return []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
