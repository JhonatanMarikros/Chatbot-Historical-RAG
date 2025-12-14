import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch

CHUNK_FILE = "clean_chunksKelas11Sem2.json"
CHROMA_DIR = "chroma_db"

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Memuat vectorstore Chroma yang sudah ada...")
db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model
)
texts = [chunk["content"] for chunk in chunks]

print("Menambahkan chunk baru ke vectorstore Chroma...")
db.add_texts(texts=texts)

db.persist()
print(f"Chunk baru telah ditambahkan dan disimpan ke folder: {CHROMA_DIR}")
