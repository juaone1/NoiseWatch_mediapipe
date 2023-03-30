import cv2
import mediapipe as mp
import os


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

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

        if result.detections:
            face_bboxC = result.detections[0].location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(face_bboxC.xmin * width), int(face_bboxC.ymin * height), int(face_bboxC.width * width), int(face_bboxC.height * height)
            
            # Crop the face
            cropped_face_image = image[y:y+h, x:x+w]

            image_name = f"{len(os.listdir(output_dir)) + 1}.jpg"
            image_path = os.path.join(output_dir, image_name)
            cv2.imwrite(image_path, cropped_face_image)
            print(f"Saved image for {name} at {image_path}")

        else:
            print(f"No face detected in {image_path}. Please make sure a face is visible.")