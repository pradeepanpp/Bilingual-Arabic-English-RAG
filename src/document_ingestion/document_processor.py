from __future__ import annotations

import re
from pathlib import Path
from typing import List, Union, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Handles document loading and processing (URLs, TXT, PDFs with OCR fallback)."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        enable_ocr_fallback: bool = True,
        ocr_min_chars: int = 80,
        ocr_dpi: int = 200,
    ):
    
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_ocr_fallback = enable_ocr_fallback
        self.ocr_min_chars = ocr_min_chars
        self.ocr_dpi = ocr_dpi

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        
        self._easyocr_reader = None

    def _normalize_metadata(self, docs: List[Document], source: str) -> List[Document]:
        
        for d in docs:
            meta = getattr(d, "metadata", None) or {}
            meta.setdefault("source", source)
            d.metadata = meta
        return docs

    def load_url_list_file(self, file_path: Union[str, Path]) -> List[str]:
        
        path = Path(file_path)
        lines = path.read_text(encoding="utf-8").splitlines()
        urls: List[str] = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
        return urls

    def load_from_url(self, url: str) -> List[Document]:
        loaded = WebBaseLoader(url).load()
        return self._normalize_metadata(loaded, source=url)

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        loaded = TextLoader(str(file_path), encoding="utf-8").load()
        return self._normalize_metadata(loaded, source=str(file_path))

    
    def _needs_ocr(self, extracted_text: str) -> bool:
        text = (extracted_text or "").strip()

        
        if len(text) < self.ocr_min_chars:
            return True

        
        arabic_chars = re.findall(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", text)
        arabic_ratio = len(arabic_chars) / max(len(text), 1)

        
        if len(text) > 200 and arabic_ratio < 0.02:
            return True

        
        if text.count("�") > 5:
            return True

        return False

    def _get_easyocr_reader(self):
    
        if self._easyocr_reader is None:
            import easyocr  
            self._easyocr_reader = easyocr.Reader(["ar", "en"], gpu=False)
        return self._easyocr_reader

    def _ocr_pdf_page(self, pdf_path: Path, page_index: int) -> str:
    
        import fitz  
        from PIL import Image
        import io

        doc = fitz.open(str(pdf_path))
        try:
            page = doc.load_page(page_index)
            zoom = self.ocr_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            reader = self._get_easyocr_reader()
            lines = reader.readtext(image, detail=0, paragraph=True)
            text = "\n".join([l.strip() for l in lines if l and str(l).strip()])
            return text.strip()
        finally:
            doc.close()

    
    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:

        import pdfplumber

        path = Path(file_path)
        docs: List[Document] = []

        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                extracted = (page.extract_text() or "").strip()

                ocr_used = False
                text = extracted

                if self.enable_ocr_fallback and self._needs_ocr(extracted):
                    try:
                        ocr_text = self._ocr_pdf_page(path, i)
                        if ocr_text:
                            text = ocr_text
                            ocr_used = True
                    except Exception:
                        pass

                if not (text or "").strip():
                    continue

                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(path), "page": i, "ocr_used": ocr_used},
                    )
                )

        return docs

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:

        directory = Path(directory)
        docs: List[Document] = []
        for pdf_path in sorted(directory.glob("*.pdf")):
            docs.extend(self.load_from_pdf(pdf_path))
        return docs

    
    def load_documents(self, sources: List[str]) -> List[Document]:
        docs: List[Document] = []

        for src in sources:
            if src.startswith(("http://", "https://")):
                docs.extend(self.load_from_url(src))
                continue

            path = Path(src)

            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))
                continue

            if path.suffix.lower() == ".pdf":
                docs.extend(self.load_from_pdf(path))
                continue

            if path.suffix.lower() == ".txt":
                lines = self.load_url_list_file(path)
                if any(u.startswith(("http://", "https://")) for u in lines):
                    for u in lines:
                        if u.startswith(("http://", "https://")):
                            docs.extend(self.load_from_url(u))
                else:
                    docs.extend(self.load_from_txt(path))
                continue

            raise ValueError(f"Unsupported source: {src} (use URL, .pdf, .txt, or folder)")

        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)

    def process_sources(self, sources: List[str]) -> List[Document]:
        docs = self.load_documents(sources)
        return self.splitter.split_documents(docs)