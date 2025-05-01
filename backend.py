from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
import sqlite3
import spacy
from textblob import TextBlob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
import speech_recognition as sr
import hashlib
import os
from typing import Optional, List
from googletrans import Translator
from collections import Counter

app = FastAPI()

nlp = spacy.load("en_core_web_sm")
db_path = "citizen_data.db"
translator = Translator()

# Initialize database
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS service (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT, 
    address TEXT, 
    phone TEXT, 
    email TEXT, 
    details TEXT
)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS complaint (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT, 
    address TEXT, 
    phone TEXT, 
    email TEXT, 
    category TEXT, 
    type TEXT, 
    details TEXT,
    sentiment TEXT,
    location TEXT,
    summary TEXT,
    response TEXT,
    complaint_hash TEXT,
    priority TEXT,
    language TEXT
)''')
conn.commit()

class ServiceRequest(BaseModel):
    name: str
    address: str
    phone: str
    email: str
    details: str

class ComplaintRequest(BaseModel):
    name: str
    address: str
    phone: str
    email: str
    details: str

CATEGORIES = {
    "Electricity": ["Street Lightening", "Frequent Cuts", "Frequent Low Voltage"],
    "Water": ["Poor Drainage", "Poor Quality Water", "No availability of water"],
    "Waste": ["Lack of Green Space", "Sewage Issues", "Waste collection service Issues", "Waste accumulated at a place"],
    "Road": ["Potholes", "Maintenance of Road", "Sidewalk Accessibility", "Traffic Congestion", "Public Transport"],
    "Other": ["Public Safety", "School Accessibility", "Healthcare Services", "Public Property Maintenance", "Internet Connectivity", "Air Quality", "Noise Pollution"]
}

# Helper Functions

def classify_category_type(text):
    for cat, types in CATEGORIES.items():
        for t in types:
            if t.lower() in text.lower():
                return cat, t
    return "Other", "OTH"

def extract_location(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text
    return "Unknown"

def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "Positive"
    elif polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"

def summarize_text(text):
    blob = TextBlob(text)
    return blob.sentences[0].string if blob.sentences else text

def generate_auto_response(category, type_):
    return f"Your complaint regarding {type_} under {category} category has been registered and will be addressed shortly."

def detect_language(text):
    try:
        return detect(text)
    except:
        return "Unknown"

def complaint_exists(details):
    hash_val = hashlib.sha256(details.encode()).hexdigest()
    cursor.execute("SELECT 1 FROM complaint WHERE complaint_hash=?", (hash_val,))
    return cursor.fetchone() is not None, hash_val

def get_similar_complaints(text):
    df = pd.read_sql_query("SELECT id, details FROM complaint", conn)
    if df.empty:
        return []
    vect = TfidfVectorizer()
    vectors = vect.fit_transform(df['details'].tolist() + [text])
    sims = cosine_similarity(vectors[-1:], vectors[:-1])[0]
    df['similarity'] = sims
    similar_df = df[df['similarity'] > 0.5]
    return similar_df['id'].tolist()

def detect_priority(text):
    urgent_keywords = ["urgent", "emergency", "immediately", "asap", "critical", "life threatening"]
    for keyword in urgent_keywords:
        if keyword in text.lower():
            return "High"
    return "Normal"

def translate_to_english(text):
    try:
        translation = translator.translate(text, dest='en')
        return translation.text
    except:
        return text

# API Endpoints

@app.post("/request_service/")
def request_service(request: ServiceRequest):
    cursor.execute("INSERT INTO service (name, address, phone, email, details) VALUES (?, ?, ?, ?, ?)",
                   (request.name, request.address, request.phone, request.email, request.details))
    conn.commit()
    return {"message": "Service request submitted successfully."}

@app.post("/raise_complaint/")
def raise_complaint(request: ComplaintRequest):
    translated = translate_to_english(request.details)
    is_duplicate, hash_val = complaint_exists(translated)
    if is_duplicate:
        raise HTTPException(status_code=400, detail="Duplicate complaint detected.")

    category, type_ = classify_category_type(translated)
    sentiment = analyze_sentiment(translated)
    location = extract_location(translated)
    summary = summarize_text(translated)
    response = generate_auto_response(category, type_)
    priority = detect_priority(translated)
    language = detect_language(request.details)

    cursor.execute("""
        INSERT INTO complaint (name, address, phone, email, category, type, details, sentiment, location, summary, response, complaint_hash, priority, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (request.name, request.address, request.phone, request.email, category, type_, request.details, sentiment, location, summary, response, hash_val, priority, language))
    conn.commit()

    similar = get_similar_complaints(translated)

    return {
        "message": "Complaint registered successfully.",
        "category": category,
        "type": type_,
        "sentiment": sentiment,
        "location": location,
        "summary": summary,
        "auto_response": response,
        "priority": priority,
        "language": language,
        "similar_complaints": similar
    }

@app.post("/speech_to_text/")
def speech_to_text(audio: UploadFile = File(...)):
    recognizer = sr.Recognizer()
    audio_path = f"temp_audio_{audio.filename}"
    with open(audio_path, "wb") as f:
        f.write(audio.file.read())

    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = "Could not understand the audio."
    os.remove(audio_path)
    return {"transcription": text}

@app.get("/admin/filter_complaints/")
def filter_complaints(category: Optional[str] = None, location: Optional[str] = None):
    query = "SELECT * FROM complaint WHERE 1=1"
    params = []
    if category:
        query += " AND category = ?"
        params.append(category)
    if location:
        query += " AND location = ?"
        params.append(location)
    df = pd.read_sql_query(query, conn, params=params)
    return df.to_dict(orient="records")

@app.get("/admin/analytics/")
def complaint_analytics():
    df = pd.read_sql_query("SELECT category, sentiment, location FROM complaint", conn)
    return {
        "total_complaints": len(df),
        "complaints_by_category": dict(Counter(df['category'])),
        "complaints_by_sentiment": dict(Counter(df['sentiment'])),
        "complaints_by_location": dict(Counter(df['location']))
    }

@app.get("/sample_data/")
def load_sample_data():
    sample = ComplaintRequest(
        name="John Doe",
        address="123 ABC Street",
        phone="1234567890",
        email="johndoe@example.com",
        details="There are frequent power cuts and low voltage in my area of Delhi. It is urgent."
    )
    return raise_complaint(sample)
