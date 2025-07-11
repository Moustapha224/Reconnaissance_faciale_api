from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import dlib
import numpy as np
import os
import uuid
import base64
from datetime import datetime
import pandas as pd

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("static/index.html")

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def load_known_faces(path="Known_faces"):
    encodings, names = [], []
    for name in os.listdir(path):
        person_path = os.path.join(path, name)
        for file in os.listdir(person_path):
            img_path = os.path.join(person_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dets = face_detector(rgb)
            if dets:
                shape = shape_predictor(rgb, dets[0])
                encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))
                encodings.append(encoding)
                names.append(name)
    return encodings, names

known_encodings, known_names = load_known_faces()

def log_presence(name, log_path="presence_log.csv"):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    if not os.path.exists(log_path):
        df = pd.DataFrame(columns=["Nom", "Date", "Heure"])
        df.to_csv(log_path, index=False)
    df = pd.read_csv(log_path)
    already_logged = ((df["Nom"] == name) & (df["Date"] == date_str)).any()
    if not already_logged and name != "Inconnu":
        df.loc[len(df)] = [name, date_str, time_str]
        df.to_csv(log_path, index=False)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_data = await file.read()
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_detector(rgb)
    for face in faces:
        shape = shape_predictor(rgb, face)
        encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))
        name = "Inconnu"
        confidence_display = ""
        distances = [np.linalg.norm(encoding - e) for e in known_encodings]
        if distances:
            min_dist = min(distances)
            confidence = max(0, 1.0 - min_dist)
            confidence_display = f" ({int(confidence * 100)}%)"
            if min_dist < 0.6:
                index = np.argmin(distances)
                name = known_names[index]
                log_presence(name)

        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{name}{confidence_display}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    _, jpeg = cv2.imencode('.jpg', frame)
    b64_img = base64.b64encode(jpeg.tobytes()).decode("utf-8")
    return {"image": f"data:image/jpeg;base64,{b64_img}"}

@app.post("/add_face")
async def add_face(name: str = Form(...), file: UploadFile = File(...)):
    image_data = await file.read()
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    save_dir = os.path.join("Known_faces", name)
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{uuid.uuid4()}.jpg"
    cv2.imwrite(os.path.join(save_dir, filename), frame)
    global known_encodings, known_names
    known_encodings, known_names = load_known_faces()
    return {"status": f"Visage ajouté pour {name}"}
