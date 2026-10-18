from src.document_ingestion.text_quality import clean_text, is_reversed_arabic, ocr_reason

ENGLISH = "The Transformer follows this overall architecture using stacked self-attention layers. " * 5
ARABIC = "يستحق العامل إجازة سنوية بأجر كامل لا تقل عن ثلاثين يوماً عن كل سنة من سنوات الخدمة " * 3


def test_english_page_is_not_sent_to_ocr():
    """The old rule sent every English page with under 2% Arabic letters to OCR."""
    assert ocr_reason(ENGLISH) == ""


def test_normal_arabic_page_is_kept():
    assert ocr_reason(ARABIC) == ""


def test_short_page_goes_to_ocr():
    assert ocr_reason("12") == "too little text"
    assert ocr_reason("") == "too little text"


def test_reversed_arabic_goes_to_ocr():
    reversed_text = " ".join(word[::-1] for word in ARABIC.split()[::-1])
    assert is_reversed_arabic(reversed_text)
    assert ocr_reason(reversed_text) == "reversed Arabic text"


def test_broken_arabic_characters_go_to_ocr():
    broken = ARABIC.replace("س", "س\ufffd")  # how this PDF font's ش comes out of PyMuPDF
    assert ocr_reason(broken) == "broken Arabic characters"


def test_clean_text():
    assert clean_text("ﺗﻨﺎزع") == "تنازع"                    # presentation forms -> normal letters
    assert clean_text("التعريفـــات") == "التعريفات"          # tatweel removed
    assert clean_text("a  b\n\n\n\nc") == "a b\n\nc"
    assert clean_text("ﬁle") == "file"
