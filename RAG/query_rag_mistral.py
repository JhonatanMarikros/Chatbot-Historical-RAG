import os, re, time, unicodedata
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from llama_cpp import Llama
from pathlib import Path

CUDA_BIN  = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
LLAMA_LIB = "../.venv/Lib/site-packages/llama_cpp/lib"
if Path(CUDA_BIN).exists():  os.add_dll_directory(CUDA_BIN)
if Path(LLAMA_LIB).exists(): os.add_dll_directory(LLAMA_LIB)
os.environ["LLAMA_LOG_LEVEL"]   = "info"
os.environ["GGML_CUDA_VERBOSE"] = "1"

CHROMA_DIR  = r"E:\Coding\Python\RAG\Mistral_Lokal\Projek\RAG\chroma_db"
EMBED_MODEL = "intfloat/multilingual-e5-large"
GGUF_PATH   = "../../models/ministral_8b/Ministral-8B-Instruct-2410-Q5_K_M.gguf"

TOP_K       = 20
COS_ABS     = 0.75
FINAL_TOPK  = 3

SHOW_SCORES = True

NOT_FOUND = "Tidak ditemukan dalam dokumen"

def strip_parens(text: str) -> str:
    clean = re.sub(r'\s*\([^)]*\)\s*', ' ', text)
    clean = re.sub(r'\s{2,}', ' ', clean)
    return clean.strip()

def _doc_key(d):
    content = (getattr(d, "page_content", "") or "").strip()
    return hash(content)

def _print_docs(title: str, docs, scores=None):
    print(f"\n==== {title} (total {len(docs)}) ====")
    for i, d in enumerate(docs, 1):
        head = f"[{i:02d}]"
        if SHOW_SCORES and scores is not None:
            head += f"  cos={scores[i-1]:.4f}"
        print(head)
        print(d.page_content)
        print("-" * 80)

def _build_prompt(context: str, question: str) -> list:
    system_content = (
        "Kamu asisten RAG sejarah Indonesia. Jawab hanya dari konteks. "
        "Tanpa tanda kurung, tanpa emoji, tanpa meta-komentar. "
        "Jika tidak ada di konteks, tulis persis: Tidak ditemukan dalam dokumen"
    )
    user_prompt = f"""
    Kamu adalah asisten sejarah Indonesia yang menjawab pertanyaan berdasarkan konteks yang diberikan.

    ### Aturan:
    1. Berikan jawaban sesuai keinginan **User**.
    2. Berikan jawabannya dengan **Bahasa Indonesia!**.
    3. Jangan tambahkan penjelasan panjang atau latar belakang yang tidak relevan.
    4. Jawablah **berdasarkan konteks**. Jika di dalam konteks ada kata, ide, atau makna yang masih berhubungan dengan pertanyaan, berikan satu kalimat singkat dan jelas yang sesuai dengan pertanyaan.
    5. Jika konteks memuat **kalimat tanya**. Abaikan bagian tanya itu dan **JANGAN menyalinnya**.
    6. Dilarang menulis teks dalam tanda kurung, tanpa emoji, tanpa meta-komentar, serta opini.
    7. Jika **benar-benar tidak ada informasi relevan**, jawab:
       "Tidak ditemukan dalam dokumen"

    ### Konteks:
    {context}

    ### Pertanyaan:
    {question}

    ### Jawaban:
    """.strip()
    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_prompt}
    ]

def normalize_query(question: str) -> str:
    q = unicodedata.normalize('NFKC', question or '').strip()

    # Tambahkan tanda tanya jika belum ada
    if q and not q.endswith('?'):
        q = q + '?'

    ACRONYMS = {
        'bpupki','ppki','nica','bkr','tni','peta','tkr','tri','apris','abri',
        'polri','apra','rms','di/tii','prri','permesta','g30s/pki','pki',
        'voc','nhm','budiutomo','sarekat_islam','pni','masyumi','nu',
        'pri','knip','kni','kmb','unci','kaa','gnb','3a',
        'bti','mpr','dpr','dpd','bpk','mk','ma','ky','kpu','kpk'
    }

    tokens = re.findall(r'[A-Za-z0-9/]+|[^\w\s]', q)

    normalized_parts = []
    skip_next = False

    for i, tok in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue

        low = tok.lower()

        if i + 1 < len(tokens) and tokens[i + 1].isalnum():
            pair_low = f"{low} {tokens[i + 1].lower()}"
            if pair_low in ACRONYMS:
                normalized_parts.append(pair_low.upper())
                skip_next = True
                continue

        if low in ACRONYMS:
            normalized_parts.append(low.upper())
        elif re.match(r'[^\w\s]', tok):
            normalized_parts.append(tok)
        else:
            normalized_parts.append(tok.capitalize())

    normalized = ''
    for i, tok in enumerate(normalized_parts):
        normalized += tok
        if i + 1 < len(normalized_parts):
            nxt = normalized_parts[i + 1]
            if not re.match(r'[^\w\s]', nxt):
                normalized += ' '

    normalized = re.sub(r'\s{2,}', ' ', normalized).strip()

    print(f"[DEBUG] Original: '{question}' -> Normalized: '{normalized}'")
    return normalized

print("[INFO] Loading embedding model (untuk Chroma)...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("[INFO] Loading ChromaDB...")
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)

print("[INFO] Loading Mistral LLM (GGUF)...")
llm = Llama(
    model_path=GGUF_PATH,
    n_ctx=2048,
    n_threads=os.cpu_count() or 8,
    n_batch=256,
    n_gpu_layers=-1,
    f16_kv=True,
    use_mmap=True,
    use_mlock=False,
    verbose=True,
    cache=None,
    chat_format="mistral-instruct",
    seed=42
)
print("[INFO] Semua model berhasil dimuat!\n")

def get_chatbot_response_with_metrics(question: str):
    t0 = time.perf_counter()
    print(f"\n[INPUT] Pertanyaan: {question}")

    normalized_question = normalize_query(question)

    print(f"[RETRIEVAL] query_text (normalized): '{normalized_question}'")
    docs_scores = db.similarity_search_with_relevance_scores(normalized_question, k=TOP_K)
    if not docs_scores:
        print("[INFO] 0 dokumen dari similarity_search_with_relevance_scores.")
        return {"answer": NOT_FOUND, "chosen": [], "candidates": []}

    kept = [(d, float(s)) for (d, s) in docs_scores if float(s) >= COS_ABS]
    print(f"[INFO] Chroma relevance: thr={COS_ABS:.2f} | kept={len(kept)}/{len(docs_scores)}")
    if not kept:
        print("[INFO] 0 dokumen >= threshold. Stop.")
        return {"answer": NOT_FOUND, "chosen": [], "candidates": []}

    kept.sort(key=lambda x: x[1], reverse=True)

    seen = set()
    unique_kept = []
    for d, s in kept:
        k = _doc_key(d)
        if k in seen:
            continue
        seen.add(k)
        unique_kept.append((d, s))
    kept = unique_kept

    kept_docs   = [d for d, _ in kept]
    kept_scores = [s for _, s in kept]

    _print_docs("KEPT (>= threshold, urut cosine)", kept_docs, scores=kept_scores)

    final_docs   = kept_docs[:min(FINAL_TOPK, len(kept_docs))]
    final_scores = kept_scores[:min(FINAL_TOPK, len(kept_scores))]
    _print_docs("KONTEKS AKHIR (TOP 3 berdasar cosine)", final_docs, scores=final_scores)

    ctx_blocks = []
    chosen_rows = []
    for rank, (d, s) in enumerate(zip(final_docs, final_scores), start=1):
        ctx_blocks.append(f"[{rank}]\n{d.page_content}")
        chosen_rows.append({
            "rank": rank,
            "source": None,
            "page": None,
            "cos": float(s),
            "preview": d.page_content
        })
    context_str = "\n\n---\n\n".join(ctx_blocks) if ctx_blocks else ""

    if not context_str:
        return {"answer": NOT_FOUND, "chosen": [], "candidates": []}

    messages = _build_prompt(context_str, normalized_question)
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=160, temperature=0.0, top_k=40, top_p=0.9, repeat_penalty=1.2,
    )
    answer = strip_parens(response["choices"][0]["message"]["content"].strip()) or NOT_FOUND
    answer = re.sub(r'^\s*(?:apa|kapan|mengapa|siapa|bagaimana)[^?]+\?\s*', '', answer, flags=re.I)
    answer = re.sub(rf'^\s*{re.escape(question.strip())}\s*', '', answer, flags=re.I).strip()
    if not answer:
        answer = NOT_FOUND

    final_keys = { _doc_key(d): i+1 for i, d in enumerate(final_docs) }

    candidates = []
    for pos, (d, s) in enumerate(kept, start=1):
        k = _doc_key(d)
        top_rank = final_keys.get(k)
        candidates.append({
            "rank": pos,
            "source": None,
            "page": None,
            "cos": float(s),
            "preview": d.page_content,
            "chosen": top_rank is not None,
            "top_rank": top_rank
        })

    print("[OUTPUT] Jawaban:", answer)
    print(f"[RUNTIME] Total: {(time.perf_counter()-t0)*1000:.2f} ms")

    return {
        "answer": answer,
        "chosen": chosen_rows,
        "candidates": candidates
    }

if __name__ == "__main__":
    while True:
        q = input("\nMasukkan pertanyaan (atau ketik 'exit'): ")
        if q.lower() == "exit":
            break

        print("\n--- get_chatbot_response_with_metrics() ---")
        res = get_chatbot_response_with_metrics(q)
        print("=> Jawaban:", res["answer"])
        print("=> Chosen:")
        for r in res["chosen"]:
            print(f"   - rank={r['rank']} cos={r['cos']:.4f}")
