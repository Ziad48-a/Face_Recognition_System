import streamlit as st
import pandas as pd
import pickle
import cv2
import numpy as np
from datetime import datetime
import mysql.connector

EMBEDDINGS = "embeddings.pkl"
MODEL_PATH = "model.pkl"
IMG_SIZE = (160, 160)

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="attendance_db"
    )

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100),
            date DATE,
            time TIME,
            confidence VARCHAR(10)
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

def is_already_marked(name):
    conn = get_connection()
    cursor = conn.cursor()
    today = datetime.now().date()
    cursor.execute(
        "SELECT * FROM attendance WHERE name = %s AND date = %s",
        (name, today)
    )
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result is not None

def save_attendance(name, confidence):
    if is_already_marked(name):
        return False
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now()
    cursor.execute(
        "INSERT INTO attendance (name, date, time, confidence) VALUES (%s, %s, %s, %s)",
        (name, now.date(), now.strftime("%H:%M:%S"), f"{confidence}%")
    )
    conn.commit()
    cursor.close()
    conn.close()
    return True

def load_attendance():
    conn = get_connection()
    df = pd.read_sql("SELECT name, date, time, confidence FROM attendance ORDER BY date DESC, time DESC", conn)
    conn.close()
    return df

def load_model():
    with open(EMBEDDINGS, "rb") as f:
        embeddings = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    lda = saved["lda"]
    model = saved["svm"]
    return model, lda, embeddings

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
    return denoised

def detect_face(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(image, 1.1, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return (x, y, w, h)

def extract_face(image):
    processed = preprocess(image)
    face_coords = detect_face(processed)
    if face_coords is None:
        face = processed
    else:
        x, y, w, h = face_coords
        face = processed[y:y+h, x:x+w]
    face = cv2.resize(face, IMG_SIZE)
    return face

def get_embedding(face_img):
    from deepface import DeepFace
    try:
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        result = DeepFace.represent(face_rgb, model_name="ArcFace", enforce_detection=False)
        return result[0]["embedding"]
    except:
        return None

def recognize(embedding, model, lda, threshold=0.6):
    if embedding is None:
        return "Unknown", 0
    embedding_lda = lda.transform(np.array(embedding).reshape(1, -1))
    prediction = model.predict(embedding_lda)[0]
    proba = model.predict_proba(embedding_lda)[0]
    confidence = round(max(proba) * 100, 2)
    if max(proba) < threshold:
        return "Unknown", confidence
    return prediction, confidence

init_db()

st.title("Attendance System")

if "model" not in st.session_state:
    try:
        st.session_state.model, st.session_state.lda, st.session_state.embeddings = load_model()
        st.success("Model loaded successfully!")
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()

st.subheader("Upload Image for Attendance")
uploaded_img = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    face = extract_face(image)
    embedding = get_embedding(face)
    name, confidence = recognize(
        embedding,
        st.session_state.model,
        st.session_state.lda
    )

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image")

    if name == "Unknown":
        st.warning(f"Face not recognized. (confidence: {confidence}%)")
    else:
        saved = save_attendance(name, confidence)
        if saved:
            st.success(f"✅ Recognized & marked: **{name}** (confidence: {confidence}%)")
        else:
            st.info(f"✅ Recognized: **{name}** — already marked today.")

# ── Attendance Log ────────────────────────────────────────────────────────────
st.subheader("📋 Attendance Log")

df = load_attendance()
if not df.empty:
    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "attendance.csv", "text/csv")
else:
    st.info("No attendance marked yet.")