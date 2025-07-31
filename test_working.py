import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from tkinter import *
import threading
import os

# Initialize models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Global variables
known_embeddings = []
person_name = ""
training_folder = r"C:\Projects\face_recognition_attendance\TrainingImages"

def train():
    """Process all images in TrainingImages and store face embeddings."""
    global known_embeddings, person_name
    person_name = name_entry.get().strip()
    if not person_name:
        print("Error: Please enter a name.")
        return
    known_embeddings = []  # Clear previous embeddings
    print(f"Starting training for {person_name}...")
    for root, _, files in os.walk(training_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Detect faces
                boxes, _ = mtcnn.detect(image_rgb)
                if boxes is not None and len(boxes) > 0:
                    faces = mtcnn.extract(image_rgb, boxes, save_path=None)
                    face = faces[0]  # Assume one face per image
                    embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
                    known_embeddings.append(embedding)
                else:
                    print(f"No face detected in {file}")
    print(f"Training complete. {len(known_embeddings)} embeddings stored for {person_name}.")

def track():
    """Track faces in webcam feed and match against stored embeddings."""
    if not person_name:
        print("Error: No person name provided. Please train first.")
        return
    if not known_embeddings:
        print("Error: No embeddings found. Please train first.")
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Starting face tracking. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect faces
        boxes, _ = mtcnn.detect(frame_rgb)
        if boxes is not None:
            faces = mtcnn.extract(frame_rgb, boxes, save_path=None)
            for i, face in enumerate(faces):
                embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
                distances = [np.linalg.norm(embedding - known_emb) for known_emb in known_embeddings]
                min_distance = min(distances)
                name = person_name if min_distance < 0.6 else "Unknown"
                # Draw rectangle and label
                x1, y1, x2, y2 = map(int, boxes[i])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('Face Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# GUI setup
root = Tk()
root.title("Face Recognition")
Label(root, text="Enter Person's Name:").pack()
name_entry = Entry(root)
name_entry.pack()
train_button = Button(root, text="Train", command=train)
train_button.pack()
track_button = Button(root, text="Track", command=lambda: threading.Thread(target=track).start())
track_button.pack()
root.mainloop()