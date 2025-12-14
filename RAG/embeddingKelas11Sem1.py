import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch

CHUNK_FILE = "clean_chunksKelas11Sem1.json"
CHROMA_DIR = "chroma_db"

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Membuat vectorstore Chroma...")

texts = [chunk["content"] for chunk in chunks]

db = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    persist_directory=CHROMA_DIR
)

db.persist()
print(f"Disimpan ke folder: {CHROMA_DIR}")
