import pandas as pd
import string
import json
import os


# =========================================================
# 1. INGEST DAN STANDARDISASI DATASET
# =========================================================

def ingest_and_standardize(file_txt, file_csv_2, file_csv_3, file_csv_4, file_jsonl_5):
    all_pantun = []

    # ---------------------------------------------------------
    # Dataset 1 (TXT)
    # ---------------------------------------------------------
    try:
        with open(file_txt, 'r', encoding='utf-8') as f:
            for line in f:
                clean_line = line.replace('<BOS>', '').replace('<EOS>', '').strip()
                larik = [l.strip().lower() for l in clean_line.split('<CLS>') if l.strip()]

                if len(larik) == 4:
                    all_pantun.append({
                        "tema": "umum",
                        "larik": larik
                    })

    except Exception as e:
        print(f"Gagal memproses Dataset 1: {e}")

    # ---------------------------------------------------------
    # Dataset 2 (CSV)
    # ---------------------------------------------------------
    try:
        df2 = pd.read_csv(file_csv_2)

        for text in df2['text_pantun'].dropna():
            text = text.replace('\\n', '\n')
            larik = [l.strip().lower() for l in text.split('\n') if l.strip()]

            if len(larik) == 4:
                all_pantun.append({
                    "tema": "umum",
                    "larik": larik
                })

    except Exception as e:
        print(f"Gagal memproses Dataset 2: {e}")

    # ---------------------------------------------------------
    # Dataset 3 (CSV)
    # ---------------------------------------------------------
    try:
        df3 = pd.read_csv(file_csv_3)

        for _, row in df3.iterrows():

            # Abaikan data kosong
            if pd.isna(row['teks']) or pd.isna(row['tipe']):
                continue

            text = str(row['teks'])
            tipe = str(row['tipe'])

            text = text.replace('\\n', '\n')
            larik = [l.strip().lower() for l in text.split('\n') if l.strip()]

            if len(larik) == 4:
                clean_tema = tipe.lower().replace("pantun ", "").strip()

                all_pantun.append({
                    "tema": clean_tema,
                    "larik": larik
                })

    except Exception as e:
        print(f"Gagal memproses Dataset 3: {e}")

    # ---------------------------------------------------------
    # Dataset 4 (CSV)
    # ---------------------------------------------------------
    try:
        df4 = pd.read_csv(file_csv_4)

        for text in df4['text_pantun'].dropna():
            text = text.replace('\\n', '\n')
            larik = [l.strip().lower() for l in text.split('\n') if l.strip()]

            if len(larik) == 4:
                all_pantun.append({
                    "tema": "umum",
                    "larik": larik
                })

    except Exception as e:
        print(f"Gagal memproses Dataset 4: {e}")

        # ---------------------------------------------------------
    # Dataset 5 (JSONL)
    # ---------------------------------------------------------
    try:
        with open(file_jsonl_5, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)

                    larik = [
                        l.strip().lower()
                        for l in data['larik']
                        if l.strip()
                    ]

                    if len(larik) == 4:
                        all_pantun.append({
                            "tema": "umum",
                            "larik": larik
                        })

                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    except Exception as e:
        print(f"Gagal memproses Dataset 5: {e}")

    return all_pantun


# =========================================================
# 2. NORMALISASI TEKS
# =========================================================

def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())

    return text


# =========================================================
# 3. DEDUPLIKASI DAN PENCATATAN DUPLIKAT
# =========================================================

def remove_duplicates(all_pantun):
    unique_pantun = []
    duplicate_pantun = []
    seen = set()

    for pantun in all_pantun:

        key = tuple(
            normalize_text(larik)
            for larik in pantun["larik"]
        )

        if key not in seen:
            seen.add(key)
            unique_pantun.append(pantun)

        else:
            duplicate_pantun.append(pantun)

    return unique_pantun, duplicate_pantun


# =========================================================
# 4. EKSTRAKSI AKHIRAN RIMA
# =========================================================

def get_suffix(line):
    clean_line = line.translate(
        str.maketrans('', '', string.punctuation)
    ).strip()

    return clean_line[-1:].lower() 


# =========================================================
# 5. FILTER RIMA ABAB DAN PENCATATAN DATA INVALID
# =========================================================

def filter_rhyme_and_save(all_pantun, output_path):

    valid_pantun = []
    invalid_pantun = []

    for pantun in all_pantun:

        larik = pantun["larik"]

        # Ekstraksi dua karakter terakhir
        suf1 = get_suffix(larik[0])
        suf2 = get_suffix(larik[1])
        suf3 = get_suffix(larik[2])
        suf4 = get_suffix(larik[3])

        # Pemeriksaan pola rima ABAB
        if suf1 == suf3 and suf2 == suf4 and suf1 and suf2 and suf1 != suf2:

            pantun_bersih = {
                "rima_a": suf1,
                "rima_b": suf2,
                "larik": larik
            }

            valid_pantun.append(pantun_bersih)

        else:

            invalid_pantun.append({
                "rima_terdeteksi": [suf1, suf2, suf3, suf4],
                "larik": larik
            })

    # ---------------------------------------------------------
    # Simpan pantun valid
    # ---------------------------------------------------------
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in valid_pantun:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # ---------------------------------------------------------
    # Simpan pantun yang gagal filter rima
    # ---------------------------------------------------------
    invalid_path = os.path.join(
        os.path.dirname(output_path),
        'invalid_rhyme.jsonl'
    )

    with open(invalid_path, 'w', encoding='utf-8') as f:
        for item in invalid_pantun:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return valid_pantun, invalid_pantun


# =========================================================
# 6. BLOK EKSEKUSI UTAMA
# =========================================================

if __name__ == "__main__":

    # Lokasi dataset mentah
    file_txt_1 = '1.DATASET/raw_data/dataset_1_Github.txt'
    file_csv_2 = '1.DATASET/raw_data/dataset_2_Mendeley.csv'
    file_csv_3 = '1.DATASET/raw_data/dataset_3_Kaggle.csv'
    file_csv_4 = '1.DATASET/raw_data/dataset_4_Mendeley.csv'
    file_jsonl_5 = '1.DATASET/raw_data/dataset_5_fix.jsonl'
    # Folder output
    output_directory = '1.DATASET/processed_data/'

    # Pastikan folder output tersedia
    os.makedirs(output_directory, exist_ok=True)

    # ---------------------------------------------------------
    # Tahap 1: Ingest Data
    # ---------------------------------------------------------
    master_list = ingest_and_standardize(
        file_txt_1,
        file_csv_2,
        file_csv_3,
        file_csv_4,
        file_jsonl_5
    )

    print("\n========== HASIL INGEST DATA ==========")
    print(f"Total pantun mentah: {len(master_list)} bait")

    # ---------------------------------------------------------
    # Tahap 2: Deduplication
    # ---------------------------------------------------------
    unique_list, duplicate_list = remove_duplicates(master_list)

    print("\n========== HASIL DEDUPLIKASI ==========")
    print(f"Total pantun unik: {len(unique_list)} bait")
    print(f"Total duplikat dihapus: {len(duplicate_list)} bait")

    # Simpan data duplikat
    duplicate_path = os.path.join(
        output_directory,
        'duplicate_data.jsonl'
    )

    with open(duplicate_path, 'w', encoding='utf-8') as f:
        for item in duplicate_list:

            # Hanya simpan larik, tanpa tema
            duplicate_output = {
                "larik": item["larik"]
            }

            f.write(json.dumps(duplicate_output, ensure_ascii=False) + '\n')

    # ---------------------------------------------------------
    # Tahap 3: Filter Rima ABAB
    # ---------------------------------------------------------
    output_jsonl = os.path.join(
        output_directory,
        'master_data.jsonl'
    )

    cleaned_list, invalid_list = filter_rhyme_and_save(
        unique_list,
        output_jsonl
    )

    print("\n========== HASIL FILTER RIMA ==========")
    print(f"Pantun valid ABAB: {len(cleaned_list)} bait")
    print(f"Pantun gagal filter: {len(invalid_list)} bait")

    # ---------------------------------------------------------
    # Tahap 4: Ringkasan Akhir
    # ---------------------------------------------------------
    print("\n========== RINGKASAN PREPROCESSING ==========")
    print(f"Total data awal       : {len(master_list)}")
    print(f"Duplikat dihapus      : {len(duplicate_list)}")
    print(f"Data setelah deduplikasi: {len(unique_list)}")
    print(f"Gagal filter rima     : {len(invalid_list)}")
    print(f"Data akhir            : {len(cleaned_list)}")

    print("\nFile berhasil disimpan:")
    print(f"- {output_jsonl}")
    print(f"- {duplicate_path}")
    print(f"- {os.path.join(output_directory, 'invalid_rhyme.jsonl')}")