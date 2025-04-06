# 🗣️ Citizen Complaint Chatbot with NLP (No OpenAI Key)

A fully-featured FastAPI-based chatbot system to handle citizen complaints and service requests. This project leverages powerful NLP tools **without using OpenAI APIs**, ensuring a cost-effective and accessible solution.

## 🚀 Features

### ✅ Complaint Handling & Categorization
- Classifies complaints into predefined categories like Electricity, Water, Waste, Road, and Others.
- Supports complaint **type detection** (e.g., Potholes, Frequent Cuts, Sewage Issues).

### 🧠 Natural Language Processing (NLP)
- **Sentiment Analysis** using TextBlob.
- **Named Entity Recognition (NER)** for location extraction using spaCy.
- **Summarization** for complaint briefs.
- **Language Detection** using `langdetect`.
- **Duplicate Detection** with SHA256 hash.
- **Similar Complaint Detection** via TF-IDF + Cosine Similarity.

### 🎙️ Voice Integration
- Convert voice complaints into text using `speech_recognition`.

### 📦 Data Storage
Two SQLite tables:
- `service` — stores citizen service requests.
- `complaint` — stores categorized, analyzed complaints.

## 📁 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/request_service/` | POST | Submit a service request |
| `/raise_complaint/` | POST | Submit and auto-analyze a complaint |
| `/speech_to_text/` | POST | Upload an audio file and get transcribed text |

## 🧪 How to Run

### Step 1: Install Requirements
```bash
pip install fastapi uvicorn spacy textblob scikit-learn pandas numpy nltk langdetect speechrecognition sentence-transformers
python -m textblob.download_corpora
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
```

### Step 2: Run FastAPI App
```bash
uvicorn main:app --reload
```

### Step 3: Open in browser
Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 📂 Database Schema

### `service` Table
- name
- address
- phone
- email
- details

### `complaint` Table
- name
- address
- phone
- email
- category
- type
- details
- sentiment
- location
- summary
- response
- complaint_hash

## 🔒 No API Keys Required
All analysis is performed locally using NLP libraries — no OpenAI or paid APIs involved.

---

## 📌 Future Enhancements
- Multilingual input handling via translation
- Admin dashboard for filtering & stats
- Alert system for urgent complaints

---

🛠️ Built with FastAPI, spaCy, TextBlob, scikit-learn, and ❤️ for public service.
