import os
import torch
import math
import sys

# Mengarah ke folder model dan dataset
sys.path.append('../2.MODEL')
sys.path.append('../1.DATASET')

from model import MiniTransformer

# --- TRICK: PATCHING ERROR CLASS DI LUAR FOLDER ---
# Kita tambahkan setter darurat ke kelas PantunCustomTokenizer secara dinamis 
# agar Python tidak protes saat file tokenization.py dibaca, tanpa mengubah file aslinya!
import tokenization
if hasattr(tokenization.PantunCustomTokenizer, 'vocab_size') and not isinstance(tokenization.PantunCustomTokenizer.vocab_size, property):
    pass
else:
    # Memasang setter properti vocab_size secara runtime
    original_property = tokenization.PantunCustomTokenizer.vocab_size
    tokenization.PantunCustomTokenizer.vocab_size = property(
        fget=original_property.fget,
        fset=lambda self, val: setattr(self, '_vocab_size', val)
    )
from tokenization import PantunCustomTokenizer
# --------------------------------------------------

def evaluate_all_checkpoints():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}\n")
    
    # 1. Pastikan file 'pantun_tokenizer.json' ada. Jika belum, generate secara otomatis dari train.txt
    tokenizer_path = "pantun_tokenizer.json"
    train_txt_path = "../1.DATASET/processed_data/train.txt"
    
    try:
        tokenizer = PantunCustomTokenizer(vocab_size=4000)
    except TypeError:
        # Fallback jika inisialisasi tanpa argumen
        tokenizer = PantunCustomTokenizer()
    
    if not os.path.exists(tokenizer_path):
        print(f"[Info] File '{tokenizer_path}' belum ditemukan. Melatih tokenizer dari {train_txt_path}...")
        if os.path.exists(train_txt_path):
            tokenizer.train_from_txt(train_txt_path, tokenizer_path)
        else:
            print(f"[Error] File training {train_txt_path} tidak ditemukan!")
            return
    else:
        tokenizer.load(tokenizer_path)
        print(f"Tokenizer berhasil dimuat dari: {tokenizer_path}")

    # 2. Muat Test Set
    test_file = "../1.DATASET/processed_data/test.txt"
    try:
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = f.readlines()
    except FileNotFoundError:
        print(f"File {test_file} tidak ditemukan. Periksa kembali path relatifnya.")
        return
        
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # 3. Daftar checkpoint di dalam folder best-checkpoint
    checkpoints = ["best13.pt", "best14.pt", "best15.pt"] 
    checkpoint_dir = "best-checkpoint"
    
    # 4. Iterasi setiap checkpoint untuk evaluasi
    for ckpt_name in checkpoints:
        print(f"\n=== Mengevaluasi Checkpoint: {ckpt_name} ===")
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        
        if not os.path.exists(ckpt_path):
            print(f"  -> Checkpoint {ckpt_path} tidak ditemukan, melewati...")
            continue

        try:
            model = MiniTransformer.from_checkpoint(ckpt_path, map_location=device).to(device)
        except Exception as e:
            print(f"  -> Gagal memuat model dari checkpoint {ckpt_name}: {e}")
            continue
            
        model.eval() # Matikan Dropout saat evaluasi
        
        total_loss = 0.0
        total_tokens = 0
        
        # 5. Hitung Performa pada Test Set
        with torch.no_grad():
            for pantun in test_data:
                if not pantun.strip():
                    continue
                    
                token_ids = tokenizer.encode(pantun)
                
                if len(token_ids) < 2:
                    continue
                    
                input_ids = torch.tensor(token_ids[:-1], dtype=torch.long).unsqueeze(0).to(device)
                targets = torch.tensor(token_ids[1:], dtype=torch.long).unsqueeze(0).to(device)

                outputs, _ = model(input_ids)
                
                loss = loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                
                total_loss += loss.item() * targets.size(1)
                total_tokens += targets.size(1)

        # 6. Kalkulasi Metrik Akhir (Loss & Perplexity)
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)
            
            print(f"Rata-rata Loss : {avg_loss:.4f}")
            print(f"Perplexity     : {perplexity:.4f}")
        else:
            print("Tidak ada token valid yang diproses dari data test.")

if __name__ == "__main__":
    evaluate_all_checkpoints()