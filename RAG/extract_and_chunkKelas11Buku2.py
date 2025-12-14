import os
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from tqdm import tqdm

DATA_DIR = r"E:\Coding\Python\RAG\Mistral_Lokal\RAG\data\Sejarah-BS-KLS-XI.pdf"
OUTPUT_FILE = "clean_chunksKelas11Buku2.json"

judul_besar = [
    "latih uji kompetensi", "latih ulangan akhir bab", "latih uji semester", "latih ulangan semester",
    "glosarium", "daftar pustaka", "kata pengantar", "daftar isi",
    "peta konsep", "profil penulis", "profil penelaah", "profil editor"
]

footer_patterns = [
    r"kelas\s*xi\s*sma/?smk",
    r"sejarah\s*untuk\s*sma/?smk\s*kelas\s*xi",
    r"semester\s*\d+",
    r"sejarah\s*indonesia",
    r"bab\s*\d+\s*[\u2022\-\–\.]?\s*.*",
]

hapus_footer_halaman = [
    "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv",
    "xv", "xvi", "2", "3","8", "9", "10", "11", "12", "13", "19", "31", "44", "45",
    "48", "47", "48","64", "77", "78", "82", "83", "84", "85", "86", "88", "127",
    "128","129", "130", "132", "133", "140", "163", "164", "165", "166", "167",
    "168", "169", "170", "171", "172", "173", "174", "175", "176", "177", "178",
    "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189",
    "190", "191", "192", "193", "194", "195", "196", "197", "198", "199", "200"
]

hapus_pdf_halaman = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24,
      25, 26, 27, 28, 29, 35, 47, 60, 61, 62, 63, 64, 65, 68, 80, 93, 94,
      98, 99, 100, 101, 102, 103, 104, 105, 106, 143, 144, 145, 146, 147,
      148, 149, 156, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
      190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
      204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216
]

def normalize_whitespace(text: str) -> str:
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'(?<=\b[A-Z])\.(?=[A-Z])', '', text)
    text = re.sub(r'\b([A-Z]{2,})\.(?=\s)', r'\1', text)
    
    return text.strip()

def is_noise_line(line):
    if not line:
        return True
    if re.search(r'\bBAB\b', line, re.IGNORECASE):
        return False
    if line.count('.') / len(line) > 0.5:
        return True
    letters_digits = sum(c.isalnum() for c in line)
    if letters_digits / len(line) < 0.3:
        return True
    return False

def is_irrelevant(text):
    text_lower = text.lower()
    if any(j.lower() in text_lower for j in judul_besar):
        return True
    return False

def is_footer_line(line):
    line_lower = line.lower()
    for pattern in footer_patterns:
        if re.search(pattern, line_lower):
            return True
    return False

def is_deleted_page(page_num, text):
    if page_num in hapus_pdf_halaman:
        return True

    lines = text.split("\n")
    for line in lines:
        clean_line = line.strip().lower()
        if clean_line in [h.lower() for h in hapus_footer_halaman]:
            return True
    return False

def extract_text_from_pdfs(data_dir):
    all_text = []
    pattern_halaman = re.compile(r'^\d+$')

    if os.path.isfile(data_dir):
        pdf_files = [data_dir]
    elif os.path.isdir(data_dir):
        pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    else:
        raise ValueError(f"Path tidak valid: {data_dir}")

    for pdf_path in pdf_files:
        print(f"Membaca {os.path.basename(pdf_path)} ...")
        reader = PdfReader(pdf_path)

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if not text:
                continue

            if is_deleted_page(page_num, text):
                print(f" Skip halaman {page_num}")
                continue

            if is_irrelevant(text):
                continue

            lines = text.split("\n")
            cleaned_lines = []

            for raw_line in lines:
                line = raw_line.strip().replace("\t", " ")
                if not line:
                    continue

                if pattern_halaman.match(line):
                    continue
                if is_footer_line(line):
                    continue
                if is_noise_line(line):
                    continue

                cleaned_lines.append(line)

            cleaned_text = " ".join(cleaned_lines).strip()
            cleaned_text = normalize_whitespace(cleaned_text)
            if cleaned_text:
                all_text.append(cleaned_text)

    return all_text

def chunk_texts(texts, chunk_size=250, chunk_overlap=60):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "],
    )

    chunks = []
    chunk_id = 1
    for doc in tqdm(texts, desc="Chunking"):
        for chunk in splitter.split_text(doc):
            chunks.append({
                "chunk_id": chunk_id,
                "content": chunk
            })
            chunk_id += 1
    return chunks

def save_chunks_to_json(chunks, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    print("Mengekstrak PDF...")
    extracted_texts = extract_text_from_pdfs(DATA_DIR)

    print(f"Teks relevan ditemukan: {len(extracted_texts)} halaman")

    print("Melakukan chunking...")
    chunks = chunk_texts(extracted_texts, chunk_size=250, chunk_overlap=60)

    print(f"Total chunk: {len(chunks)}")

    print(f"Menyimpan ke: {OUTPUT_FILE}")
    save_chunks_to_json(chunks, OUTPUT_FILE)

    print("Selesai!")
