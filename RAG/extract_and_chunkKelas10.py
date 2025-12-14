import os
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from tqdm import tqdm

DATA_DIR = r"..\data\Kelas X Sejarah BS press.pdf"
OUTPUT_FILE = "clean_chunksKelas10.json"

# Judul besar yang ingin dihapus (filter halaman)
judul_besar = [
    "glosarium", "daftar pustaka", "kata pengantar", "daftar isi",
    "profil penulis", "profil penelaah", "profil editor"
]


# Daftar halaman yang ingin dihapus berdasarkan nomor footer
hapus_footer_halaman = [
   "iii", "iv", "v", "vi", "vii", "viii",
   "2", "17", "33", "45", "53", "68", "74", "85",
   "96", "99", "150", "156", "167", "169", "170", "175",
   "234", "239", "258", "259", "260", "261", "262", "263", "264", "265", "266", "267", "268", "269",
   "270", "271", "272", "273", "274", "275", "276", "277", "278", "279", "280"
]

# Halaman dihapus berdasarkan nomor PDF (1-based)
hapus_pdf_halaman = [
   1, 2, 3, 4, 5, 6, 7, 8, 10, 25, 41, 53, 61, 76, 82, 93, 104, 107,
   158, 164, 175, 177, 178, 183, 242, 247, 264, 265, 266, 267, 268,
   269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281,
   282, 283, 284, 285, 286, 287, 288
]

# Footer yang ingin dihapus per-baris (tanpa hapus halamannya)
footer_patterns = [
    # Kelas X + kombinasi SMA/MA/SMK/MAK
    r"kelas\s*x\s*(?:sma|ma|smk|mak)(?:/?\s*(?:sma|ma|smk|mak))*",
    # Sejarah Indonesia
    r"sejarah\s*indonesia",
]

def normalize_whitespace(text: str) -> str:
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
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
        pdf_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(".pdf")
        ]
    else:
        raise ValueError(f"Path tidak valid: {data_dir}")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"Membaca {filename} ...")

        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if not text:
                continue

            if is_deleted_page(page_num, text):
                print(f"Skip halaman {page_num}")
                continue

            if is_irrelevant(text):
                continue

            lines = text.split("\n")
            cleaned_lines = []

            for line in lines:
                line = line.strip()
                line = line.replace("\t", " ")  
                
                
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

# Lakukan chunking
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

# Simpan ke JSON
def save_chunks_to_json(chunks, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

# Main pipeline
if __name__ == "__main__":
    print("📚 Mengekstrak PDF...")
    extracted_texts = extract_text_from_pdfs(DATA_DIR)

    print(f"✅ Teks relevan ditemukan: {len(extracted_texts)} halaman")

    print("🔍 Melakukan chunking...")
    chunks = chunk_texts(extracted_texts, chunk_size=250, chunk_overlap=60)

    print(f"✅ Total chunk: {len(chunks)}")

    print(f"💾 Menyimpan ke: {OUTPUT_FILE}")
    save_chunks_to_json(chunks, OUTPUT_FILE)

    print("🎉 Selesai!")
