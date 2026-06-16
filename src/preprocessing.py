"""Text normalization shared by every matching stage.

The raw input column (`Hospital_Name (clean)`) is not actually clean — it
carries claim-system noise (review dates, document numbers, boilerplate
phrases) mixed in with the real hospital name. Centralizing the cleaning
here means TF-IDF, fuzzy and SBERT all score the same normalized string
instead of three slightly different ones, which was the source of several
disagreements in the original notebooks.
"""
import re

ABBREVIATIONS = {
    "RS": "RUMAH SAKIT",
    "RSU": "RUMAH SAKIT UMUM",
    "RSUD": "RUMAH SAKIT UMUM DAERAH",
    "RSIA": "RUMAH SAKIT IBU DAN ANAK",
    "RSUP": "RUMAH SAKIT UMUM PUSAT",
    "RSPAD": "RUMAH SAKIT PUSAT ANGKATAN DARAT",
    "RSCM": "RUMAH SAKIT CIPTO MANGUNKUSUMO",
    "RSAL": "RUMAH SAKIT ANGKATAN LAUT",
    "RSAU": "RUMAH SAKIT ANGKATAN UDARA",
    "PTE": "PRIVATE",
    "LTD": "LIMITED",
}

NOISE_PHRASES = [
    "PENDING REVIEW", "REVIEW DOR", "DOKUMEN LAPORAN", "SUDAH DIKIRIM",
    "MOHON", "NASABAH", "DAPAT", "MELAMPIRKAN", "DOKUMEN", "PROSES",
    "SEDANG", "KLAIM", "TUNDA", "CLAIM", "PENDING", "REVIEW", "DOR",
    "DITINDAKLANJUTI", "DENGAN", "PIHAK", "KONFIRMASI", "RINCIAN", "BIAYA",
    "JAWABAN", "CL", "KELUHAN", "SUHU", "HARI", "KE", "AKHIR",
    "JANUARI", "FEBRUARI", "MARET", "APRIL", "MEI", "JUNI", "JULI",
    "AGUSTUS", "SEPTEMBER", "OKTOBER", "NOVEMBER", "DESEMBER", "TOLAK",
    "KONSUL", "TERKAIT", "PEMERIKSAAN", "APA", "SAJA", "TIDAK",
    "KOLERASI", "DIAGNOSA", "ESKALASI", "TAX", "INVOICE", "ASLI",
    "TOTAL", "TAGIHAN", "YANG", "SESUAI", "EMAIL", "AIA", "TANGGAL",
    "DILAKUKAN", "BERDIRI", "INFORMASI", "MEDIS", "LANJUTAN", "SELAMA",
    "TERPISAH", "OLEH", "LAMA", "PENGISIAN", "UNTUK", "UNIT", "PEMERINTAH",
]

_NOISE_RE = re.compile(
    r"\b(" + "|".join(sorted(NOISE_PHRASES, key=len, reverse=True)) + r")\b"
)
_DATE_NUM_RE = re.compile(r"\b(20\d{2}|\d{1,2})\b")
_PUNCT_RE = re.compile(r"[^\w\s&]")
_WS_RE = re.compile(r"\s+")


def strip_noise(text: str) -> str:
    text = _NOISE_RE.sub(" ", text)
    text = _DATE_NUM_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def resolve_label_conflicts(df, text_col: str = "clean", label_col: str = "true"):
    """Collapse (text -> label) pairs to one label per unique text.

    The training data's canonical-name column is itself inconsistently
    labeled: the same cleaned input string sometimes maps to different
    canonical spellings across rows (e.g. "RS EMC ALAM SUTERA" vs
    "EMC ALAM SUTERA" for the identical input). Left alone, a matcher that
    finds a *perfect* text match can still be scored "wrong" because it
    picked whichever conflicting label happened to come first in the
    corpus. Picking the majority label per unique input (ties broken by
    the longer/more descriptive string) removes that arbitrary tie-break
    -- see docs/accuracy_analysis.md for the measured impact.
    """
    def pick(group):
        counts = group.value_counts()
        top = counts[counts == counts.max()].index
        return max(top, key=len)

    resolved = df.groupby(text_col)[label_col].agg(pick).reset_index()
    return resolved


def clean_name(text, expand_abbrev: bool = False) -> str:
    """Normalize a raw hospital name string.

    Steps: uppercase -> strip punctuation (keep '&') -> strip noise
    phrases/dates -> collapse whitespace -> optional abbreviation expansion.
    """
    if text is None or (isinstance(text, float)):
        return ""
    text = str(text).upper().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    text = strip_noise(text)

    if expand_abbrev:
        tokens = text.split()
        tokens = [ABBREVIATIONS.get(tok, tok) for tok in tokens]
        text = " ".join(tokens)

    return text
