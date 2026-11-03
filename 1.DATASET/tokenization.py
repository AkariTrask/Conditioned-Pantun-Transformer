"""
Syllable-Bounded BPE + Character Fallback untuk pantun berbahasa Indonesia.

Ide inti:
  1. Setiap kata dipecah dulu menjadi SUKU KATA (aturan fonotaktik Indonesia).
  2. Setiap suku kata diperlakukan sebagai "kata" terpisah oleh BPE, sehingga
     merge BPE TIDAK PERNAH melewati batas suku kata.
     -> token terpanjang = 1 suku kata. Akhiran baris (penentu rima) selalu
        terlihat di level token, tidak tersembunyi di dalam token kata utuh.
  3. Alfabet dasar BPE = karakter. Suku kata langka (frekuensi < min_frequency)
     otomatis terurai ke karakter -> fallback char-level tanpa kode tambahan.
  4. Penanda awal kata "▁" menempel di suku kata pertama, jadi spasi bisa
     direkonstruksi tepat saat decode.

Format dataset:
    <|start|> [RIMA: h, g]
    baris 1
    baris 2
    baris 3
    baris 4
    <|end|>
"""
import math
import random
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer

WORD_MARK = "▁"
SPECIAL_TOKENS = [
    "<|unk|>", "<|pad|>", "<|start|>", "<|end|>",
    "<|rima_a|>", "<|rima_b|>", "<|isi|>", "<|batas|>",
]
SPECIAL_RE = re.compile(r"(<\|[a-z_]+\|>)")
BASE_ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789.,;:!?'\"()-\n") + [WORD_MARK]


# =====================================================================
# 1. Syllabifier Bahasa Indonesia (rule-based)
# =====================================================================
_VOWELS = set("aiueo")
_DIGRAPHS = ("ng", "ny", "sy", "kh")  # satu fonem konsonan
_DIPHTHONGS = {"ai", "au", "oi", "ei"}  # diftong hanya di akhir kata (pan-tai, ker-bau)


def _units(word: str) -> List[str]:
    units, i = [], 0
    while i < len(word):
        if word[i:i + 2] in _DIGRAPHS:
            units.append(word[i:i + 2])
            i += 2
        else:
            units.append(word[i])
            i += 1
    return units


@lru_cache(maxsize=200_000)
def syllabify(word: str) -> Tuple[str, ...]:
    """
    V-V      -> V|V      (ma|in, tu|a)
    V-C-V    -> V|CV     (ba|ca)
    V-CC..-V -> VC|C..V  (kan|tor, sung|guh, ting|gi)
    ng/ny/sy/kh dihitung satu konsonan.
    """
    units = _units(word)
    nuclei, i = [], 0
    while i < len(units):
        if units[i] in _VOWELS:
            if i + 2 == len(units) and units[i + 1] in _VOWELS and units[i] + units[i + 1] in _DIPHTHONGS:
                nuclei.append((i, i + 2))
                break
            nuclei.append((i, i + 1))
        i += 1

    if len(nuclei) <= 1:
        return (word,)

    cuts = []
    for (_, end_k), (start_next, _) in zip(nuclei[:-1], nuclei[1:]):
        gap = start_next - end_k
        cuts.append(end_k if gap <= 1 else end_k + 1)

    bounds = [0] + cuts + [len(units)]
    return tuple("".join(units[a:b]) for a, b in zip(bounds[:-1], bounds[1:]))


_TOKEN_RE = re.compile(r"[a-z]+|\d+|\n|[^\sa-z\d]")


def prep_text(text: str) -> str:
    """
    Teks mentah -> string suku kata dipisah spasi (siap untuk BPE).
    'sungguh-sungguh\\nuang' -> '▁sung guh - sung guh \\n u ang'  (tanpa ▁ di awal segmen/baris)
    """
    text = text.lower().replace("\r", "")
    out, pos = [], 0
    for m in _TOKEN_RE.finditer(text):
        had_space = any(c in " \t" for c in text[pos:m.start()])
        pos = m.end()
        tok = m.group()
        if tok == "\n":
            out.append("\n")
            continue
        pieces = list(syllabify(tok)) if tok[0].isalpha() else [tok]
        if had_space:
            pieces[0] = WORD_MARK + pieces[0]
        out.extend(pieces)
    return " ".join(out)


# =====================================================================
# 2. Parser dataset + filter bahasa Indonesia
# =====================================================================
_PANTUN_RE = re.compile(
    r"<\|start\|>\s*\[RIMA:\s*([^,\]]*?)\s*,\s*([^\]]*?)\s*\]\s*\n(.*?)\n\s*<\|end\|>",
    re.S,
)
_EN_FUNC = {
    "the", "and", "of", "to", "is", "are", "you", "your", "with", "that", "this",
    "for", "was", "were", "my", "he", "she", "they", "we", "have", "has", "not",
    "but", "from", "will", "be",
}


def looks_indonesian(lines: List[str]) -> bool:
    """
    Filter ringan tanpa dependensi eksternal:
    - hanya karakter latin dasar + tanda baca umum (tolak aksara lain / karakter aneh)
    - rasio stopword Inggris rendah (tolak baris berbahasa Inggris)
    """
    text = " ".join(lines).lower()
    if re.search(r"[^a-z0-9\s.,;:!?'\"()\-]", text):
        return False
    words = re.findall(r"[a-z]+", text)
    if not words:
        return False
    return sum(w in _EN_FUNC for w in words) / len(words) < 0.15


def load_pantun_file(path: Union[str, Path], filter_language: bool = True, verbose: bool = True) -> List[Dict]:
    content = Path(path).read_text(encoding="utf-8")
    records, bad_fmt, bad_lang = [], 0, 0
    for m in _PANTUN_RE.finditer(content):
        lines = [l.strip().lower() for l in m.group(3).split("\n") if l.strip()]
        if len(lines) != 4:
            bad_fmt += 1
            continue
        if filter_language and not looks_indonesian(lines):
            bad_lang += 1
            continue
        records.append({
            "rima_a": m.group(1).strip().lower(),
            "rima_b": m.group(2).strip().lower(),
            "larik": lines,
        })
    if verbose:
        print(f"[Data] {Path(path).name}: {len(records):,} valid | "
              f"{bad_fmt} format salah | {bad_lang} ditolak filter bahasa")
    return records


def extract_words(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


# =====================================================================
# 3. Tokenizer
# =====================================================================
class PantunCustomTokenizer:
    def __init__(self, vocab_size: int = 4000, min_frequency: int = 2):
        # vocab_size = batas atas. Karena merge tidak boleh lintas suku kata,
        # vocab biasanya jenuh sebelum batas ini. Ukuran aktual: self.vocab_size.
        self.target_vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = SPECIAL_TOKENS
        self._special_set = set(SPECIAL_TOKENS)
        self.tokenizer = None
        self.pad_token_id = self.start_token_id = self.end_token_id = None
        self.isi_token_id = self.batas_token_id = None

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    # alias agar kompatibel dengan kode lama
    @property
    def bos_token_id(self):
        return self.start_token_id

    @property
    def eos_token_id(self):
        return self.end_token_id

    def train_from_txt(self, txt_path: Union[str, Path], save_path: Union[str, Path]):
        """Latih HANYA dari train.txt (tanpa data leakage), hanya pantun berbahasa Indonesia."""
        records = load_pantun_file(txt_path, filter_language=True)
        if not records:
            raise RuntimeError("Tidak ada pantun valid untuk melatih tokenizer.")

        self.tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        self.tokenizer.pre_tokenizer = Split(pattern=" ", behavior="removed")
        trainer = BpeTrainer(
            vocab_size=self.target_vocab_size,
            min_frequency=self.min_frequency,   # suku kata langka -> terurai ke karakter
            special_tokens=self.special_tokens,
            initial_alphabet=BASE_ALPHABET,     # fallback karakter selalu tersedia
            show_progress=True,
        )
        corpus = (prep_text("\n".join(r["larik"])) for r in records)
        print(f"[Info] Melatih syllable-bounded BPE dari {txt_path} ...")
        self.tokenizer.train_from_iterator(corpus, trainer=trainer)
        self.tokenizer.save(str(save_path))
        self._set_ids()
        print(f"[Info] Tersimpan: {save_path} | vocab aktual = {self.vocab_size}")

    def load(self, path: Union[str, Path]):
        self.tokenizer = Tokenizer.from_file(str(path))
        self._set_ids()

    def _set_ids(self):
        t = self.tokenizer.token_to_id
        self.pad_token_id = t("<|pad|>")
        self.start_token_id = t("<|start|>")
        self.end_token_id = t("<|end|>")
        self.isi_token_id = t("<|isi|>")
        self.batas_token_id = t("<|batas|>")

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for seg in SPECIAL_RE.split(text):
            if not seg:
                continue
            if seg in self._special_set:
                ids.append(self.tokenizer.token_to_id(seg))
            else:
                ids.extend(self.tokenizer.encode(prep_text(seg), add_special_tokens=False).ids)
        return ids

    def tokenize(self, text: str) -> List[str]:
        return [self.tokenizer.id_to_token(i) for i in self.encode(text)]

    def decode(self, token_ids: List[int], skip_special: bool = False) -> str:
        parts = []
        for i in token_ids:
            tok = self.tokenizer.id_to_token(int(i))
            if tok is None:
                continue
            if tok in self._special_set:
                if not skip_special:
                    parts.append(tok)
                continue
            parts.append(tok)
        text = "".join(parts).replace(WORD_MARK, " ")
        return re.sub(r"[ ]*\n[ ]*", "\n", text)


# =====================================================================
# 4. Dataset
# =====================================================================
class PantunDataset(Dataset):
    """
    Conditioning dropout: 'full' (dengan rima) vs 'unconditional' (tanpa rima).
    Loss hanya dihitung pada bagian pantun (setelah <|isi|>).
    """
    def __init__(
        self,
        txt_path: Union[str, Path],
        tokenizer: PantunCustomTokenizer,
        max_seq_len: int = 128,
        is_training: bool = True,
        p_uncond: float = 0.10,
        mask_conditioning_loss: bool = True,
        filter_language: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.is_training = is_training
        self.p_uncond = p_uncond
        self.mask_conditioning_loss = mask_conditioning_loss
        self.data: List[Dict] = load_pantun_file(txt_path, filter_language=filter_language)

    @staticmethod
    def build_prefix(rima_a: str, rima_b: str, mode: str = "full") -> str:
        if mode == "full":
            return f"<|start|><|rima_a|>{rima_a}<|rima_b|>{rima_b}<|isi|>\n"
        return "<|start|><|isi|>\n"

    @staticmethod
    def build_prompt_string(item: Dict, mode: str = "full") -> str:
        prefix = PantunDataset.build_prefix(item["rima_a"], item["rima_b"], mode)
        sampiran = "\n".join(item["larik"][:2])
        isi = "\n".join(item["larik"][2:])
        return f"{prefix}{sampiran}\n<|batas|>\n{isi}\n<|end|>"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        mode = "uncond" if (self.is_training and random.random() < self.p_uncond) else "full"
        token_ids = self.tokenizer.encode(self.build_prompt_string(item, mode))

        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len - 1] + [self.tokenizer.end_token_id]

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)

        if self.mask_conditioning_loss:
            pos = (input_ids == self.tokenizer.isi_token_id).nonzero(as_tuple=True)[0]
            if pos.numel() > 0:
                labels[: int(pos[0]) + 1] = -100  # prefix (kondisi) tidak ikut loss

        return {"input_ids": input_ids, "labels": labels}


class PantunCollator:
    def __init__(self, pad_token_id: int, ignore_index: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        inputs = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=self.ignore_index)
        return {"input_ids": inputs, "labels": labels, "attention_mask": (inputs != self.pad_token_id).long()}


# =====================================================================
# 5. Profiler panjang konteks
# =====================================================================
def analyze_context_length(txt_path: Union[str, Path], tokenizer: PantunCustomTokenizer) -> Dict[str, Union[int, float]]:
    records = load_pantun_file(txt_path, verbose=False)
    lens, n_words = [], 0
    for r in records:
        lens.append(len(tokenizer.encode(PantunDataset.build_prompt_string(r, "full"))))
        n_words += len(extract_words(" ".join(r["larik"])))
    arr = np.array(lens)
    max_len, p99 = int(arr.max()), int(np.percentile(arr, 99))
    rec = 2 ** int(math.ceil(math.log2(max_len)))
    tok_per_word = (arr.sum() / max(1, n_words))

    print("=" * 60)
    print("LAPORAN PANJANG TOKEN (Syllable-Bounded BPE)")
    print("=" * 60)
    print(f"Sampel            : {len(arr):,}")
    print(f"Mean / P99 / Max  : {arr.mean():.1f} / {p99} / {max_len}")
    print(f"Token per kata    : {tok_per_word:.2f} (termasuk special token)")
    print(f"Vocab aktual      : {tokenizer.vocab_size}")
    print(f"Rekomendasi BLOCK_SIZE : {rec}")
    print("=" * 60)
    return {"max_len": max_len, "p99": p99, "recommended_pow2": rec}


# =====================================================================
if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent
    TRAIN_TXT = HERE / "processed_data" / "train.txt"
    TOK_PATH = HERE / "pantun_syl_bpe.json"
    retrain = "--retrain" in sys.argv

    tok = PantunCustomTokenizer(vocab_size=4000, min_frequency=2)
    if retrain or not TOK_PATH.exists():
        if not TRAIN_TXT.exists():
            print(f"[Error] {TRAIN_TXT} tidak ditemukan.")
            sys.exit(1)
        tok.train_from_txt(TRAIN_TXT, TOK_PATH)
    else:
        print("[Info] Memuat tokenizer yang ada (pakai --retrain untuk melatih ulang)...")
        tok.load(TOK_PATH)

    sample = "dahan pohon ada yang teguh\nnamun dahan bengkok tak boleh ditebang"
    print("\nSyllables :", [syllabify(w) for w in extract_words(sample)])
    print("Tokens    :", tok.tokenize(sample))
    assert tok.decode(tok.encode(sample)) == sample, "Round-trip gagal!"
    print("Round-trip OK")

    stats = analyze_context_length(TRAIN_TXT, tok)
    ds = PantunDataset(TRAIN_TXT, tok, max_seq_len=stats["recommended_pow2"], is_training=True)
    dl = DataLoader(ds, batch_size=4, collate_fn=PantunCollator(tok.pad_token_id))
    print(f"\n[OK] DataLoader siap: {len(ds):,} sampel")
