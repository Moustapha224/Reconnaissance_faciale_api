from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import dlib
import numpy as np
import os
from PIL import Image
import shutil
import uuid

app = FastAPI()

# Monter le dossier "static" pour servir les résultats
app.mount("/static", StaticFiles(directory="static"), name="static")

# Chargement des modèles Dlib
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Fonction pour encoder les visages connus
def load_known_faces(path="Known_faces"):
    encodings = []
    names = []
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

# Page HTML d'accueil
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head>
        <title>Reconnaissance Faciale</title>
    </head>
    <body>
        <h2>Uploader une image pour reconnaissance</h2>
        <form action="/upload" enctype="multipart/form-data" method="post">
            <input name="file" type="file" accept="image/*">
            <input type="submit" value="Envoyer">
        </form>
    </body>
    </html>
    """

# Route d'upload
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Enregistrer le fichier temporairement
    img_data = await file.read()
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_filename, "wb") as f:
        f.write(img_data)

    # Traitement de l’image
    frame = cv2.imread(temp_filename)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb)

    for face in faces:
        shape = shape_predictor(rgb, face)
        encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))
        name = "Inconnu"
        distances = [np.linalg.norm(encoding - e) for e in known_encodings]
        if distances:
            min_dist = min(distances)
            if min_dist < 0.6:
                index = np.argmin(distances)
                name = known_names[index]
        # Dessin
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Sauvegarder résultat
    result_path = f"static/results/result_{uuid.uuid4()}.jpg"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    cv2.imwrite(result_path, frame)

    # Supprimer l’image temporaire
    os.remove(temp_filename)

    return HTMLResponse(f"""
        <h3>Résultat :</h3>
        <img src='/{result_path}' width="500"><br>
        <a href="/">Revenir</a>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
