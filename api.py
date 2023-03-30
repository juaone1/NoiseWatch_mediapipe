import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import os
import pickle
from flask import Flask, Response
import threading
import socket
import struct
import time
import pyrebase


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

def continuous_stream():
    global stream_state, frames_lock, current_frame
    while True:
        if stream_state:
            for frame in stream():
                with frames_lock:
                    current_frame = frame
                    if not stream_state:
                        break
        else:
            time.sleep(0.1)

stream_thread = threading.Thread(target=continuous_stream)
stream_thread.start()

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def monitor_angle_file():
    global stream_state, face_names, face_images

    last_contents = None
    while True:
        try:
            with open('angle.txt', 'r') as f:
                current_contents = f.read()
                print(last_contents)
                if current_contents != last_contents and stream_state:
                    last_contents = current_contents
                    print("New content", last_contents)

                    print(face_names)
                    # Call the handle_recognized_faces function here
                    handle_recognized_faces(face_names, face_images)
        except FileNotFoundError:
            pass

        time.sleep(1)  

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
    

def extract_faces_from_image(image, face_locations):
    face_images = []
    for (top, right, bottom, left) in face_locations:
        face_image = image[top:bottom, left:right]
        face_images.append(face_image)
    return face_images

def detect_faces(image, face_detector):
    height, width, _ = image.shape
    results = face_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    face_locations = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            face_locations.append((y, x+w, y+h, x))
    return face_locations

def recognize_faces(image, known_face_encodings, known_face_names, face_locations):
    face_encodings = face_recognition.face_encodings(image, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        if not known_face_encodings:  # If no known face encodings, skip recognition
            face_names.append("Unknown")
            continue

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance= 0.5)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
    return face_names

def draw_boxes_and_labels(image, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(image, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def register_face(image, full_name, face_detector):
    dataset_dir = 'dataset'
    name_dir = os.path.join(dataset_dir, full_name)

    if not os.path.exists(name_dir):
        os.makedirs(name_dir)

    # Detect faces and get the bounding box
    height, width, _ = image.shape
    resized_image = cv2.cvtColor(cv2.resize(image, (640, 480)), cv2.COLOR_BGR2RGB)
    result = face_detector.process(resized_image)

    if result.detections:
        face_bboxC = result.detections[0].location_data.relative_bounding_box
        ih, iw, _ = image.shape
        x, y, w, h = int(face_bboxC.xmin * width), int(face_bboxC.ymin * height), int(face_bboxC.width * width), int(face_bboxC.height * height)
        
        # Crop the face
        cropped_face_image = image[y:y+h, x:x+w]

        image_name = f"{len(os.listdir(name_dir)) + 1}.jpg"
        image_path = os.path.join(name_dir, image_name)
        cv2.imwrite(image_path, cropped_face_image)
        print(f"Saved image for {full_name} at {image_path}")

    else:
        print("No face detected. Please make sure a face is visible.")

    train()
    print("Training Done.")


def train():
    known_face_encodings = []
    known_face_names = []

    dataset_path = "dataset"
    for person_name in os.listdir(dataset_path):
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
    embeddings_path = "embeddings.pkl"
    with open(embeddings_path, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)


def stream():
    global face_names, face_images
    cap = cv2.VideoCapture(0)

    embeddings_path = "embeddings.pkl"
    if os.path.exists(embeddings_path):
        with open(embeddings_path, "rb") as f:
            known_face_encodings, known_face_names = pickle.load(f)
    else:
        train()
        with open(embeddings_path, "rb") as f:
            known_face_encodings, known_face_names = pickle.load(f)


    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detector:
        registering_face = False
        full_name = ""


        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_frame)


            if not registering_face:
                face_locations = detect_faces(frame, face_detector)
                face_names = recognize_faces(frame, known_face_encodings, known_face_names, face_locations)
                face_images = extract_faces_from_image(frame, face_locations)
                print("stream face name:", face_names)
                draw_boxes_and_labels(frame, face_locations, face_names)
            else:
                cv2.putText(frame, f"Registering {full_name}, press Spacebar to capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # cv2.imshow('Real-time Face Recognition', frame)
            
            key = cv2.waitKey(5) & 0xFF

            if key == 27:  # Press 'Esc' to exit
                break
            elif key == ord('r'):  # Press 'r' to start registering a face
                registering_face = not registering_face
                if registering_face:
                    full_name = input("Enter the full name of the person: ").strip()
            elif key == ord(' ') and registering_face:  # Press Spacebar to capture the image
                if face_locations:
                    register_face(frame, full_name, face_detector)
                else:
                    print("No face detected. Please make sure a face is visible.")

@app.route('/stream_start')
def stream_start():
    global stream_state
    stream_state = True
    return "Stream started."

@app.route('/stream_stop')
def stream_stop():
    global stream_state
    stream_state = False
    return "Stream stopped."


@app.route('/video_feed')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global frames_lock, current_frame
    while True:
        with frames_lock:
            if current_frame:
                yield current_frame
        time.sleep(0.1)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # start receive_embeddings in a separate thread
    # t = threading.Thread(target=receive_embeddings)
    # t.start()
    # Start the monitor_angle_file function in a separate thread
    angle_monitor_thread = threading.Thread(target=monitor_angle_file)
    angle_monitor_thread.daemon = True
    angle_monitor_thread.start()

    # start the Flask server
    app.run(host='0.0.0.0', port= 5000, debug=False)