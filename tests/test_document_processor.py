import json
import logging

import pytest
from langchain_core.documents import Document

from src.document_ingestion.document_processor import DocumentProcessor, read_url_list
from tests.conftest import make_pdf


class FakeReader:
    def __init__(self, text="نص من التعرف الضوئي"):
        self.text = text
        self.inputs = []

    def readtext(self, image, detail=0, paragraph=True):
        self.inputs.append(image)
        return [self.text]


class BrokenReader:
    def readtext(self, *args, **kwargs):
        raise RuntimeError("model not found")


LONG_TEXT = "This page has enough real text to be indexed without OCR. " * 2


@pytest.fixture
def processor(tmp_path):
    return DocumentProcessor(cache_dir=tmp_path / "cache")


def test_text_pages_keep_their_spaces_and_skip_ocr(tmp_path, processor):
    pdf = make_pdf(tmp_path / "en.pdf", ["The encoder is composed of a stack of six identical layers. " * 4])
    processor.ocr_reader = BrokenReader()  # would fail if OCR were used
    pages = processor.load_pdf(pdf)
    assert len(pages) == 1
    assert "stack of six identical layers" in pages[0].page_content
    assert pages[0].metadata == {"source": "en.pdf", "page": 0, "method": "text"}


def test_blank_page_is_read_with_ocr_and_cached(tmp_path, processor):
    pdf = make_pdf(tmp_path / "scan.pdf", ["", LONG_TEXT])
    reader = FakeReader()
    processor.ocr_reader = reader
    pages = processor.load_pdf(pdf)
    assert [p.metadata["method"] for p in pages] == ["ocr", "text"]
    assert pages[0].page_content == "نص من التعرف الضوئي"
    assert isinstance(reader.inputs[0], bytes)  # EasyOCR does not accept PIL images

    cache = json.loads(next((tmp_path / "cache").glob("*.json")).read_text(encoding="utf-8"))
    assert cache["file"] == "scan.pdf" and cache["pages"]["0"] == "نص من التعرف الضوئي"

    second = DocumentProcessor(cache_dir=tmp_path / "cache")
    second.ocr_reader = BrokenReader()  # the cache is used, OCR is not run again
    assert second.load_pdf(pdf)[0].page_content == "نص من التعرف الضوئي"


def test_ocr_failure_is_logged_not_hidden(tmp_path, processor, caplog):
    pdf = make_pdf(tmp_path / "scan.pdf", [""])
    processor.ocr_reader = BrokenReader()
    with caplog.at_level(logging.WARNING):
        assert processor.load_pdf(pdf) == []
    assert "OCR failed on page 1 of scan.pdf: model not found" in caplog.text


def test_no_ocr_mode(tmp_path):
    pdf = make_pdf(tmp_path / "scan.pdf", ["", LONG_TEXT])
    pages = DocumentProcessor(use_ocr=False, cache_dir=tmp_path / "cache").load_pdf(pdf)
    assert [p.metadata["page"] for p in pages] == [1]


def test_bad_sources_are_skipped(tmp_path, processor, monkeypatch, caplog):
    pdf = make_pdf(tmp_path / "en.pdf", ["A page with enough English text to be indexed without OCR. " * 2])
    urls = tmp_path / "url.txt"
    urls.write_text("# comment\nhttps://example.com/a\n\nhttps://example.com/b\n", encoding="utf-8")

    def load_url(url):
        raise ConnectionError("offline")

    monkeypatch.setattr(processor, "load_url", load_url)
    with caplog.at_level(logging.WARNING):
        pages = processor.load_sources([str(pdf), str(urls), str(tmp_path / "missing" / "url.txt"), "notes.docx"])
    assert len(pages) == 1
    assert "Could not load https://example.com/a: offline" in caplog.text
    assert "Skipped unsupported source: notes.docx" in caplog.text


def test_read_url_list(tmp_path):
    path = tmp_path / "url.txt"
    path.write_text("# my sources\nhttps://a.example\n\n  https://b.example  \n", encoding="utf-8")
    assert read_url_list(path) == ["https://a.example", "https://b.example"]
    assert read_url_list(tmp_path / "none.txt") == []


def test_chunks_keep_page_metadata(tmp_path, processor):
    pdf = make_pdf(tmp_path / "en.pdf", ["word " * 400])
    chunks = processor.split(processor.load_pdf(pdf))
    assert len(chunks) > 1
    assert all(c.metadata["page"] == 0 and len(c.page_content) <= 500 for c in chunks)


def test_chunks_end_at_sentences(processor):
    line = "The base model uses a dropout rate of 0.1 and was trained\nfor 12 hours on 8 GPUs. "
    chunks = processor.split([Document(page_content=line * 12, metadata={"page": 0})])
    assert len(chunks) > 1
    assert all(c.page_content.rstrip().endswith("GPUs.") for c in chunks)
    assert all("0.1" in c.page_content for c in chunks)
