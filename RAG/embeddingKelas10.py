import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch

CHUNK_FILE = "clean_chunksKelas10.json"
CHROMA_DIR = "chroma_db"

# Load chunks dari file JSON
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load embedding model multilingual-e5
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Buat Chroma vector store
print("📦 Membuat vectorstore Chroma...")
# Ambil hanya bagian "content"
texts = [chunk["content"] for chunk in chunks]

db = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    persist_directory=CHROMA_DIR
)


# Simpan vectorstore
db.persist()
print(f"✅ Disimpan ke folder: {CHROMA_DIR}")
