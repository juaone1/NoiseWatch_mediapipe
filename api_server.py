import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import os
import pickle
import threading
import socket
import struct
import time
import pyrebase
import atexit

from mediapipe.python.solutions import face_mesh as mp_face_mesh
from deepface import DeepFace
from deepface.commons import distance as dst
from flask import Flask, request, jsonify, Response

firebaseConfig = {
  "apiKey": "AIzaSyAKT3G3TthUZR6zULpaUZu6YrVmvT59JiU",
  "authDomain": "noisewatch-70f20.firebaseapp.com",
  "databaseURL": "https://noisewatch-70f20-default-rtdb.asia-southeast1.firebasedatabase.app",
  "projectId": "noisewatch-70f20",
  "storageBucket": "noisewatch-70f20.appspot.com",
  "messagingSenderId": "474011763806",
  "appId": "1:474011763806:web:0227f2525e82dc18f1a0cf",
  "measurementId": "G-JY0FC6GDFT"
}



firebase = pyrebase.initialize_app(firebaseConfig)

db = firebase.database()
storage = firebase.storage()

app = Flask(__name__)

stream_state = False
frames_lock = threading.Lock()
current_frame = None
# face_data_queue = Queue()
face_names = []
face_images = []

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def cleanup():
    global cap
    if cap is not None:
        cap.release()
        print("VideoCapture released")

def save_face_to_firebase(full_name, face_image):

    if face_image is None or face_image.size == 0:
        print("Empty or invalid face image, skipping upload")
        return 

    image_name = f"{full_name}_{int(time.time())}.jpg"
    storage_path = f"unknown_faces/{image_name}"
    _, encoded_image = cv2.imencode('.jpg', face_image)
    storage.child(storage_path).put(encoded_image.tobytes())
    image_url = storage.child(image_name).get_url(None)
    return image_url

def update_firebase_record(full_name):
    record_ref = db.child("Records").child(full_name)
    record = record_ref.get()
    print(record.val())
    if not record.val():
        record_ref.set({
            "name": full_name,
            "offense": 1
        })
        print("Added new record")
    else:
        offense = db.child("Records").child(full_name).child("offense").get().val()
        if offense <= 2:
            db.child("Records").child(full_name).update({"offense": (offense+1)})
        print("added offense")

def handle_recognized_faces(face_names, face_images):
    for name, face_image in zip(face_names, face_images):
        if name == "Unknown":
            # Save the unknown face to Firebase Storage
            image_url = save_face_to_firebase(name, face_image)
            print(f"Unknown face saved to Firebase Storage: {image_url}")
        else:
            # Increment the offense counter for the recognized face in Firebase Realtime Database
            update_firebase_record(name)
            print(f"Offense count updated for {name} in Firebase Realtime Database")
    

def register_face(image, full_name, face_detector):
    dataset_dir = 'dataset'
    name_dir = os.path.join(dataset_dir, full_name)

    if not os.path.exists(name_dir):
        os.makedirs(name_dir)

    # Detect faces and get the bounding box
    height, width, _ = image.shape
    resized_image = cv2.cvtColor(cv2.resize(image, (640, 480)), cv2.COLOR_BGR2RGB)
    result = face_detector.process(resized_image)

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0].landmark
        face_points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in face_landmarks]

        min_x = min(point[0] for point in face_points)
        max_x = max(point[0] for point in face_points)
        min_y = min(point[1] for point in face_points)
        max_y = max(point[1] for point in face_points)

        x, y, w, h = min_x, min_y, max_x - min_x, max_y - min_y
        
        # Crop the face
        cropped_face_image = image[y:y+h, x:x+w]

        image_name = f"{len(os.listdir(name_dir)) + 1}.jpg"
        image_path = os.path.join(name_dir, image_name)
        cv2.imwrite(image_path, cropped_face_image)
        print(f"Saved image for {full_name} at {image_path}")

    else:
        print("No face detected. Please make sure a face is visible.")


def gen_frames():
    global cap
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show the result
                   

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def capture_image_from_webcam():
    global cap
    ret, frame = cap.read()
    return frame

@app.route('/register', methods=['POST'])
def register():
    with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,    
    ) as face_detector:
        full_name = request.json['full_name']

        # Capture the image from the laptop's webcam
        image = capture_image_from_webcam()

        register_face(image, full_name, face_detector)

    return jsonify({'status': 'success'})

@app.route('/train', methods=['POST'])
def train():
    # Load existing embeddings, if available
    embeddings_path = "embeddings.pkl"
    if os.path.exists(embeddings_path):
        with open(embeddings_path, "rb") as f:
            known_face_encodings, known_face_names = pickle.load(f)
    else:
        known_face_encodings = []
        known_face_names = []

    dataset_path = "dataset"
    for person_name in os.listdir(dataset_path):
        if person_name not in known_face_names:
            person_dir = os.path.join(dataset_path, person_name)
            for image_filename in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_filename)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                
                # Check if a face is detected in the image
                if face_encodings:
                    face_encoding = face_encodings[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(person_name) 

    with open(embeddings_path, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    
    if os.path.exists(embeddings_path):
        # send_embeddings_to_rpi()
        return "Training complete"
    else:
        print("No embeddings.pkl file")
        return "Error training face recognition model.", 500
    

def send_embeddings_to_rpi():

    # Set up socket and connect to the server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_ip = "192.168.1.21"
    s.connect((server_ip, 1234))

    # Send the file type
    file_type = "pickle"
    s.send(bytes(file_type, "utf-8"))

    # Receive a response from the server to indicate that the file type has been received
    response = s.recv(1024).decode()
    print(response)

    # Send the data length prefix
    with open("embeddings.pkl", "rb") as f:
        data = f.read()
        data_length_prefix = struct.pack("!I", len(data))
        s.sendall(data_length_prefix)

    # Send the file data in chunks
    bytes_sent = 0
    while bytes_sent < len(data):
        remaining_bytes = len(data) - bytes_sent
        chunk_size = 4096 if remaining_bytes > 4096 else remaining_bytes
        s.send(data[bytes_sent:bytes_sent+chunk_size])
        bytes_sent += chunk_size

    # Receive a response from the server to indicate that the file data has been received
    response = s.recv(1024).decode()
    print(response)

    # Close the connection
    s.close()

def save_embeddings_to_file(embeddings, filename):
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings_from_file(filename):
    with open(filename, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def load_images_from_dataset(dataset_path):
    images = {}
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        for image_filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_filename)
            image = cv2.imread(image_path)
            images[image_path] = image
    return images

def compare_input_image_to_dataset(input_image, dataset_path, embeddings_filename):

    input_image_encoded = cv2.imencode('.jpg', input_image)[1]
    input_image_bytes = input_image_encoded.tobytes()

    if os.path.exists(embeddings_filename):
        embeddings_cache = load_embeddings_from_file(embeddings_filename)
    else:
        dataset_images = load_images_from_dataset(dataset_path)
        embeddings_cache = {}
        for image_path, image in dataset_images.items():
            embedding = DeepFace.represent(image, model_name='ArcFace', detector_backend='retinaface', enforce_detection=False)
            embeddings_cache[image_path] = embedding

        save_embeddings_to_file(embeddings_cache, embeddings_filename)

    input_image_array = cv2.imdecode(np.frombuffer(input_image_bytes, np.uint8), cv2.IMREAD_COLOR)
    input_embedding = DeepFace.represent(input_image_array, model_name='ArcFace', detector_backend='retinaface', enforce_detection=False)

    max_similarity = 0
    most_similar_image_path = None
    for image_path, embedding in embeddings_cache.items():
        cosine_similarity = dst.findCosineDistance(input_embedding, embedding)
        similarity_percentage = (1 - cosine_similarity) * 100
        if similarity_percentage > max_similarity:
            max_similarity = similarity_percentage
            most_similar_image_path = image_path

    # Get the subfolder (full name) of the most similar image
    most_similar_subfolder = os.path.dirname(most_similar_image_path).split(os.sep)[-1]

    return most_similar_subfolder, max_similarity

@app.route('/compare', methods=['POST'])
def compare_images():
    input_image = request.get_data()
    input_image = cv2.imdecode(np.frombuffer(input_image, np.uint8), cv2.IMREAD_COLOR)
    dataset_path = 'dataset'
    embeddings_filename = "embeddings_cache.pkl"

    most_similar_name, highest_percentage = compare_input_image_to_dataset(input_image, dataset_path, embeddings_filename)

    if highest_percentage >= 50:
        return jsonify({
            "name": most_similar_name,
            "percentage": highest_percentage
        })
    else:
        return jsonify({
            "error": "Face does not have a match..."
        })


if __name__ == "__main__":

    atexit.register(cleanup)

    # start the Flask server
    app.run(host='0.0.0.0', port= 5000, debug=False)