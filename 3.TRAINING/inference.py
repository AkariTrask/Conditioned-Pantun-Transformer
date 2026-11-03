"""
inference.py - Generate pantun dari best.pt (lokal ATAU dari repo HuggingFace).

Taruh file ini di 3.TRAINING/ (sejajar train.py).

Contoh:
  # interaktif, ambil checkpoint + tokenizer dari HuggingFace
  python inference.py --repo_id usernamemu/pantun-mini --ckpt_file best.pt --tokenizer_file pantun_syl_bpe.json

  # interaktif, file lokal
  python inference.py --ckpt checkpoints/best.pt --tokenizer ../1.DATASET/pantun_syl_bpe.json

  # sekali jalan, 5 kandidat, diurutkan berdasarkan kecocokan rima
  python inference.py --repo_id usernamemu/pantun-mini --rima "h, g" --awalan "dahan pohon" --n 5 --rank
"""
import os
import re
import sys
import argparse
import torch
import torch.nn.functional as F

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(BASE_DIR, '2.MODEL'))
sys.path.append(os.path.join(BASE_DIR, '1.DATASET'))

from model import MiniTransformer, ModelConfig
from tokenization import PantunCustomTokenizer, PantunDataset, extract_words, load_pantun_file

# Token yang tidak boleh muncul di tengah generasi
BANNED = ["<|unk|>", "<|pad|>", "<|start|>", "<|rima_a|>", "<|rima_b|>", "<|isi|>"]


# =============================================================================
# Loading
# =============================================================================
def resolve_file(local_path, repo_id, hf_filename, label):
    if local_path:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"{label} lokal tidak ditemukan: {local_path}")
        return local_path
    if repo_id and hf_filename:
        from huggingface_hub import hf_hub_download
        print(f"[Init] Mengunduh {label} '{hf_filename}' dari HF repo '{repo_id}' ...")
        return hf_hub_download(repo_id=repo_id, filename=hf_filename)
    raise ValueError(f"Tentukan --{label.lower()} (lokal) atau --repo_id + nama file {label}.")


def load_model(ckpt_path, device):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = MiniTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    info = {k: ckpt.get(k) for k in ("step", "val_loss", "best_val")}
    return model, cfg, info


# =============================================================================
# Sampling (berhenti di <|end|>, mendukung temperature / top-k / top-p)
# =============================================================================
@torch.no_grad()
def sample_ids(model, tok, prefix_ids, max_new_tokens, temperature, top_k, top_p, greedy, device):
    block = model.cfg.block_size
    banned = [tok.tokenizer.token_to_id(t) for t in BANNED]
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits, _ = model(ids[:, -block:])
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        logits[:, banned] = float("-inf")

        if greedy:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            if top_p and top_p < 1.0:
                s_logits, s_idx = torch.sort(logits, descending=True)
                probs = F.softmax(s_logits, dim=-1)
                remove = (torch.cumsum(probs, dim=-1) - probs) > top_p
                s_logits = s_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(1, s_idx, s_logits)
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == tok.end_token_id:
            break
    return ids[0].tolist()


# =============================================================================
# Post-processing & pengecekan rima
# =============================================================================
def extract_lines(tok, ids):
    text = tok.decode(ids)
    body = text.split("<|isi|>", 1)[-1].split("<|end|>")[0].replace("<|batas|>", "\n")
    return [l.strip() for l in body.split("\n") if l.strip()]


def last_letter(line):
    letters = re.findall(r"[a-z]", line.lower())
    return letters[-1] if letters else ""


def rhyme_hits(lines, a, b):
    if not a or not b:
        return None
    return [last_letter(l) == t for l, t in zip(lines[:4], [a, b, a, b])]


def print_pantun(lines, a, b, known_words=None, idx=None):
    head = f"--- Pantun {idx} ---" if idx is not None else "--- Pantun ---"
    print(head)
    hits = rhyme_hits(lines, a, b)
    for i, line in enumerate(lines):
        mark = ""
        if hits is not None and i < 4:
            mark = f"  [{'OK' if hits[i] else 'x'} rima {[a, b, a, b][i]}]"
        print(f"  {line}{mark}")
        if i == 1 and len(lines) > 2:
            print("  ·")  # pemisah sampiran | isi
    if len(lines) != 4:
        print(f"  (!) jumlah baris = {len(lines)}, bukan 4")
    if known_words is not None:
        unk = sorted({w for w in extract_words(" ".join(lines)) if w not in known_words})
        if unk:
            print(f"  kata tak ada di korpus train (cek typo): {', '.join(unk)}")
    print()


def parse_rima(s):
    if not s or not s.strip():
        return None, None
    parts = [p.strip().lower() for p in re.split(r"[,\s]+", s.strip()) if p.strip()]
    if len(parts) != 2:
        raise ValueError("Format rima: dua huruf/akhiran, mis. 'h, g'")
    return parts[0], parts[1]


# =============================================================================
# Satu kali generate (n kandidat)
# =============================================================================
def run(model, tok, device, args, rima, awalan, known_words):
    a, b = parse_rima(rima)
    mode = "full" if a else "uncond"
    prefix = PantunDataset.build_prefix(a or "", b or "", mode)
    prefix_ids = tok.encode(prefix)
    if awalan and awalan.strip():
        prefix_ids += tok.encode(awalan.strip().lower())

    results = []
    for _ in range(args.n):
        ids = sample_ids(model, tok, prefix_ids, args.max_new_tokens,
                         args.temperature, args.top_k, args.top_p, args.greedy, device)
        lines = extract_lines(tok, ids)
        hits = rhyme_hits(lines, a, b)
        score = (sum(hits) if hits else 0) + (1 if len(lines) == 4 else 0)
        results.append((score, lines))

    if args.rank and a:
        results.sort(key=lambda r: -r[0])
    for i, (_, lines) in enumerate(results, 1):
        print_pantun(lines, a, b, known_words, idx=i)


# =============================================================================
def main():
    p = argparse.ArgumentParser()
    # sumber file
    p.add_argument("--repo_id", type=str, default=None, help="HF repo, mis. user/pantun-mini")
    p.add_argument("--ckpt_file", type=str, default="best.pt", help="Nama file ckpt di HF repo")
    p.add_argument("--tokenizer_file", type=str, default="pantun_syl_bpe.json", help="Nama file tokenizer di HF repo")
    p.add_argument("--ckpt", type=str, default=None, help="Path best.pt lokal (menimpa HF)")
    p.add_argument("--tokenizer", type=str, default=None, help="Path tokenizer lokal (menimpa HF)")
    p.add_argument("--train_txt", type=str, default=None, help="Opsional: train.txt untuk menandai kata tak dikenal (proxy typo)")
    # prompt
    p.add_argument("--rima", type=str, default=None, help="Mis. 'h, g'. Kosongkan untuk mode interaktif")
    p.add_argument("--awalan", type=str, default="", help="Awalan baris 1 (kata utuh), opsional")
    # sampling
    p.add_argument("--n", type=int, default=3, help="Jumlah pantun per prompt")
    p.add_argument("--rank", action="store_true", help="Urutkan kandidat berdasarkan kecocokan rima & struktur 4 baris")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    ckpt_path = resolve_file(args.ckpt, args.repo_id, args.ckpt_file, "ckpt")
    tok_path = resolve_file(args.tokenizer, args.repo_id, args.tokenizer_file, "tokenizer")

    tok = PantunCustomTokenizer()
    tok.load(tok_path)
    model, cfg, info = load_model(ckpt_path, device)

    if cfg.vocab_size != tok.vocab_size:
        raise RuntimeError(
            f"Vocab mismatch: checkpoint={cfg.vocab_size} vs tokenizer={tok.vocab_size}. "
            "Pakai tokenizer yang sama persis dengan saat training."
        )
    print(f"[Init] device={device} | params={model.num_params()/1e6:.2f}M | vocab={cfg.vocab_size} | "
          f"block={cfg.block_size} | step={info['step']} | val_loss={info['val_loss']}")

    known_words = None
    if args.train_txt:
        recs = load_pantun_file(args.train_txt, verbose=False)
        known_words = {w for r in recs for l in r["larik"] for w in extract_words(l)}

    if args.rima is not None:
        run(model, tok, device, args, args.rima, args.awalan, known_words)
        return

    print("\nMode interaktif. Ketik 'q' untuk keluar.")
    while True:
        try:
            rima = input("\nRima (mis. 'h, g' | kosong = tanpa rima): ").strip()
            if rima.lower() in ("q", "quit", "exit"):
                break
            awalan = input("Awalan baris 1 (opsional): ").strip()
            if awalan.lower() in ("q", "quit", "exit"):
                break
            run(model, tok, device, args, rima, awalan, known_words)
        except ValueError as e:
            print(f"[Input salah] {e}")
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()
