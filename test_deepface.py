import os
import cv2
from deepface import DeepFace
from deepface.commons import distance as dst
import pickle

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

def compare_input_image_to_dataset(input_image_path, dataset_path, embeddings_filename):
    input_image = cv2.imread(input_image_path)

    if os.path.exists(embeddings_filename):
        embeddings_cache = load_embeddings_from_file(embeddings_filename)
    else:
        dataset_images = load_images_from_dataset(dataset_path)
        embeddings_cache = {}
        for image_path, image in dataset_images.items():
            embedding = DeepFace.represent(image, model_name='ArcFace', detector_backend='retinaface', enforce_detection=False)
            embeddings_cache[image_path] = embedding

        save_embeddings_to_file(embeddings_cache, embeddings_filename)

    input_embedding = DeepFace.represent(input_image, model_name='ArcFace', detector_backend='retinaface', enforce_detection=False)

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

# Usage
input_image_path = 'unknownkein.jpg'
dataset_path = 'dataset'

most_similar_name, highest_percentage = compare_input_image_to_dataset(input_image_path, dataset_path, "embeddings_cache.pkl")

if highest_percentage >= 50:
    print("Most similar name:", most_similar_name)
    print(f"Highest similarity percentage: {highest_percentage} %")

else:
    print("Face does not have a match...")