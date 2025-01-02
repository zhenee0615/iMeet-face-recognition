# import os

# import logging
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import base64
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import cv2
# from deepface import DeepFace
# import requests

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# # Preload the model
# logger.info("Preloading DeepFace model...")
# deepface_model = DeepFace.build_model("Facenet")
# logger.info("Model preloaded successfully.")

# @app.route('/')
# def main():
#     return "Welcome to the Enhanced Flask API"

# @app.route('/face_recognition', methods=['OPTIONS', 'POST'])
# def perform_face_verification():
#     if request.method == 'OPTIONS':
#         logger.info("Received OPTIONS request")
#         return jsonify({"message": "Preflight check successful"}), 200
    
#     logger.info("Received POST request")
#     try:
#         logger.info("Processing image data")
#         payload = request.get_json()
#         captured_frame = payload['image_data']
#         profile_image_path = payload['profile_image_url']

#         decoded_frame = base64.b64decode(captured_frame)
#         frame_image = Image.open(BytesIO(decoded_frame))
#         frame_array = np.array(frame_image)
#         frame_rgb = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)

#         logger.info("1")

#         if not analyze_texture(frame_rgb):
#             return jsonify({"error": "Detected potential spoofing due to blur or low texture."})
        
#         logger.info("2")

#         # detected_faces = DeepFace.extract_faces(img_path=frame_rgb, enforce_detection=True, detector_backend='opencv', anti_spoofing=True)
#         # if not detected_faces:
#         #     return jsonify({"error": "No face detected in the frame."})
#         # if not detected_faces[0].get("is_real", False):
#         #     return jsonify({"error": "Detected spoofing attack!"})
        
#         logger.info("3")

#         if profile_image_path.startswith('http'):
#             response = requests.get(profile_image_path)
#             if response.status_code != 200:
#                 return jsonify({"error": "Unable to fetch profile image from the provided URL."})
#             reference_image = np.array(Image.open(BytesIO(response.content)))
#         else:
#             reference_image = cv2.imread(profile_image_path)
#             if reference_image is None:
#                 return jsonify({"error": "Profile image not found at the specified path."})
        
#         logger.info("4")

#         reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

#         logger.info("5")

#         comparison_result = DeepFace.verify(img1_path=frame_rgb, img2_path=reference_image_rgb, enforce_detection=True)

#         logger.info("done")
#         return jsonify({
#             "match": comparison_result['verified'],
#             "anti_spoofing": True
#         })

#     except Exception as error:
#         logger.info("error", error)
#         return jsonify({"error": str(error)})
    
# def analyze_texture(frame_rgb):
#     gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
#     laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
#     return laplacian_var > 100

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
from deepface import DeepFace

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Preload the model
logger.info("Preloading DeepFace model...")
deepface_model = DeepFace.build_model("Facenet")
logger.info("Model preloaded successfully.")

def resize_image(image_array, max_size=512):
    height, width = image_array.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(image_array, new_size, interpolation=cv2.INTER_AREA)
    return image_array

@app.route('/face_recognition', methods=['OPTIONS', 'POST'])
def perform_face_verification():
    try:
        payload = request.get_json()
        captured_frame = payload['image_data']
        profile_image_path = payload['profile_image_url']

        # Decode and resize captured frame
        decoded_frame = base64.b64decode(captured_frame)
        frame_image = Image.open(BytesIO(decoded_frame))
        frame_array = np.array(frame_image)
        frame_rgb = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
        frame_rgb = resize_image(frame_rgb)

        # Load profile image and resize
        reference_image = cv2.imread(profile_image_path)
        if reference_image is None:
            return jsonify({"error": "Profile image not found"})
        reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
        reference_image_rgb = resize_image(reference_image_rgb)

        # Perform face verification
        comparison_result = DeepFace.verify(
            img1_path=frame_rgb,
            img2_path=reference_image_rgb,
            enforce_detection=True,
            model=deepface_model
        )

        return jsonify({"match": comparison_result['verified']})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)})


# import logging
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import base64
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import cv2
# import dlib
# from scipy.spatial import distance as dist
# import face_recognition
# import requests
# import os
# import tempfile
# import psutil

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize Dlib's face detector and facial landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor_path = 'shape_predictor_68_face_landmarks.dat'  # Ensure this file is in your project directory
# if not os.path.exists(predictor_path):
#     logger.error(f"Facial landmark predictor not found at {predictor_path}. Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and place it in the project directory.")
#     exit(1)
# predictor = dlib.shape_predictor(predictor_path)

# # Define EAR threshold and consecutive frames for blink detection
# EAR_THRESHOLD = 0.21
# CONSEC_FRAMES = 3

# # Initialize counters for blink detection
# COUNTER = 0
# TOTAL = 0

# def eye_aspect_ratio(eye):
#     # Compute the euclidean distances between the two sets of vertical eye landmarks
#     A = dist.euclidean(eye[1], eye[5])
#     B = dist.euclidean(eye[2], eye[4])

#     # Compute the euclidean distance between the horizontal eye landmarks
#     C = dist.euclidean(eye[0], eye[3])

#     # Compute the EAR
#     ear = (A + B) / (2.0 * C)
#     return ear

# def log_memory_usage(stage):
#     process = psutil.Process(os.getpid())
#     mem = process.memory_info().rss / (1024 * 1024)  # in MB
#     logger.info(f"Memory Usage at {stage}: {mem:.2f} MB")

# @app.route('/')
# def main():
#     return "Welcome to the Enhanced Flask API"

# @app.route('/face_recognition', methods=['OPTIONS', 'POST'])
# def perform_face_verification():
#     if request.method == 'OPTIONS':
#         logger.info("Received OPTIONS request")
#         return jsonify({"message": "Preflight check successful"}), 200

#     logger.info("Received POST request")
#     try:
#         logger.info("Processing image data")
#         payload = request.get_json()
#         captured_frame = payload['image_data']
#         profile_image_path = payload['profile_image_url']

#         # Decode the captured image from base64
#         decoded_frame = base64.b64decode(captured_frame)
#         frame_image = Image.open(BytesIO(decoded_frame)).convert('RGB')
#         frame_array = np.array(frame_image)
#         frame_rgb = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)

#         log_memory_usage("After decoding image")

#         logger.info("1: Analyzing texture")
#         if not analyze_texture(frame_rgb):
#             logger.info("Texture analysis failed")
#             return jsonify({"error": "Detected potential spoofing due to blur or low texture."})

#         log_memory_usage("After texture analysis")

#         logger.info("2: Detecting and extracting faces in captured frame")
#         gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
#         rects = detector(gray, 0)

#         if len(rects) == 0:
#             logger.info("No face detected in the frame.")
#             return jsonify({"error": "No face detected in the frame."})

#         # Assume the first detected face is the target
#         rect = rects[0]
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)

#         # Extract the coordinates of the left and right eye
#         (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#         (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]

#         # Compute EAR for both eyes
#         leftEAR = eye_aspect_ratio(leftEye)
#         rightEAR = eye_aspect_ratio(rightEye)
#         ear = (leftEAR + rightEAR) / 2.0

#         logger.info(f"EAR: {ear:.2f}")

#         # Determine if a blink was detected
#         global COUNTER, TOTAL
#         if ear < EAR_THRESHOLD:
#             COUNTER += 1
#         else:
#             if COUNTER >= CONSEC_FRAMES:
#                 TOTAL += 1
#                 logger.info(f"Blink detected. Total blinks: {TOTAL}")
#                 liveness = True
#             COUNTER = 0
#             liveness = False

#         log_memory_usage("After blink detection")

#         if not liveness:
#             logger.info("Liveness detection failed. No blink detected.")
#             return jsonify({"error": "Liveness detection failed. Please blink your eyes."})

#         # Proceed with face verification
#         logger.info("3: Verifying face with profile image")
#         # Load profile image
#         if profile_image_path.startswith('http'):
#             response = requests.get(profile_image_path)
#             if response.status_code != 200:
#                 logger.info("Unable to fetch profile image from URL")
#                 return jsonify({"error": "Unable to fetch profile image from the provided URL."})
#             reference_image = face_recognition.load_image_file(BytesIO(response.content))
#         else:
#             if not os.path.exists(profile_image_path):
#                 logger.info("Profile image not found at path")
#                 return jsonify({"error": "Profile image not found at the specified path."})
#             reference_image = face_recognition.load_image_file(profile_image_path)

#         # Encode faces
#         captured_encodings = face_recognition.face_encodings(frame_rgb)
#         reference_encodings = face_recognition.face_encodings(reference_image)

#         if not captured_encodings:
#             logger.info("No face encoding found in the captured image.")
#             return jsonify({"error": "No face encoding found in the captured image."})
#         if not reference_encodings:
#             logger.info("No face encoding found in the profile image.")
#             return jsonify({"error": "No face encoding found in the profile image."})

#         # Compare faces
#         match = face_recognition.compare_faces([reference_encodings[0]], captured_encodings[0])[0]
#         distance = face_recognition.face_distance([reference_encodings[0]], captured_encodings[0])[0]

#         logger.info(f"Face match: {match}, Distance: {distance:.4f}")

#         log_memory_usage("After face verification")

#         logger.info("done")
#         return jsonify({
#             "match": match,
#             "anti_spoofing": liveness
#         })

#     except Exception as error:
#         logger.error("Error during face verification: %s", error, exc_info=True)
#         return jsonify({"error": str(error)}), 500

# def analyze_texture(frame_rgb):
#     gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
#     laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
#     return laplacian_var > 100

# # Helper functions from imutils
# from imutils import face_utils

# if __name__ == '__main__':
#     app.run(debug=True)
