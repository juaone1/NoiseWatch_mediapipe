import cv2
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import os


mp_drawing = mp.solutions.drawing_utils

face_detector = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,    
    ) 

raw_data_dir = 'D:/Documents/Design2/DATASET'

for name in os.listdir(raw_data_dir):
    name_dir = os.path.join(raw_data_dir, name)
    if not os.path.isdir(name_dir):
        continue

    print(f"Processing {name}...")
    output_dir = os.path.join('dataset', name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(name_dir):
        image_path = os.path.join(name_dir, filename)
        if not os.path.isfile(image_path):
            continue

        image = cv2.imread(image_path)

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

            image_name = f"{len(os.listdir(output_dir)) + 1}.jpg"
            image_path = os.path.join(output_dir, image_name)
            cv2.imwrite(image_path, cropped_face_image)
            print(f"Saved image for {name} at {image_path}")

        else:
            print(f"No face detected in {image_path}. Please make sure a face is visible.")