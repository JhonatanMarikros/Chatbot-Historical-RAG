# CHATBOT HISTORICAL RAG
## Sistem Tanya Jawab Sejarah Indonesia Berbasis RAG

### Cara Menjalankan Program:
1. Clone repository ini
2. Buat virtual environment: `python -m venv .venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Download model GGUF dari HuggingFace di https://huggingface.co/bartowski/Ministral-8B-Instruct-2410-GGUF
5. Simpan model di folder: `models\ministral_8b`
6. Masuk .venv dengan cara `.\.venv\Scripts\activate`
8. Jalankan: `python RAG/Chatbot/app.py`
9. Buka browser: http://localhost:5000