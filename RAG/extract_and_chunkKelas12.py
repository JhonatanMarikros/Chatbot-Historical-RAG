import os
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from tqdm import tqdm

DATA_DIR = r"E:\Coding\Python\RAG\Mistral_Lokal\RAG\data\Kelas XII Sejarah BS press.pdf"

OUTPUT_FILE = "clean_chunksKelas12.json"

judul_besar = [
    "glosarium", "daftar pustaka", "kata pengantar", "daftar isi", "profil penulis", "profil penelaah", "profil editor"
]


footer_patterns = [
    r"kelas\s*xii\s*sma/?ma",              
    r"sejarah\s*indonesia",                 
    r"kelas\s*xi\s*sma/?smk",                
    r"sejarah\s*untuk\s*sma/?smk\s*kelas\s*xi",
    r"semester\s*\d+",
    r"bab\s*\d+\s*[\u2022\-\–\.]?\s*.*",    
]

hapus_footer_halaman = [
    "iii", "iv", "v", "vi", "vii", "viii", "3", 
    "4", "18", "22", "23", "24", "70", "71", "72", 
    "156", "157", "158", "169", "179", "180", "181", 
    "201", "220", "242", "243", "244", "245", "245", 
    "246", "247", "248", "249", "250", "251", "252", 
    "253", "254", "255", "256"
]

hapus_pdf_halaman = [
   1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 26, 30, 31, 32, 45, 
   74, 75, 76, 78, 79, 80, 109, 160, 161, 162, 164, 165, 166, 
   176, 177, 183, 184, 187, 188, 189, 208, 209, 227, 228, 245, 
   246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 
   258, 259, 260, 261, 262, 263, 264
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

# Main pipeline
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
