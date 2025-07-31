import tkinter as tk
from tkinter import messagebox, ttk
import csv
import cv2
import os
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import threading
from collections import defaultdict
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F

class AttendanceSystem:
    def __init__(self):
        self.setup_directories()
        self.setup_device()
        self.setup_facenet()
        self.setup_gui()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Enhanced parameters for better detection
        self.min_face_size = (30, 30)
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.quality_threshold = 0.3
        
    def setup_device(self):
        """Setup device for PyTorch (GPU if available, otherwise CPU)"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def setup_facenet(self):
        """Initialize FaceNet models"""
        try:
            # Initialize MTCNN for face detection
            self.mtcnn = MTCNN(
                keep_all=True, 
                device=self.device,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net thresholds
                post_process=True
            )
            
            # Initialize InceptionResnetV1 for face recognition
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            
            print("FaceNet models initialized successfully")
        except Exception as e:
            print(f"Error initializing FaceNet: {e}")
            messagebox.showerror("Error", f"Failed to initialize FaceNet models: {e}")
        
    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        directories = ['TrainingImages', 'ImagesUnknown', 'QualityCheck', 'models', 'embeddings']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
    def setup_gui(self):
        """Setup the GUI interface"""
        self.window = tk.Tk()
        self.window.title("FACENET ENHANCED STUDENT ATTENDANCE SYSTEM")
        self.window.geometry('900x600')
        self.window.configure(background='#2c3e50')
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        
        # Create main frame
        main_frame = tk.Frame(self.window, bg='#2c3e50')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text="FACENET ENHANCED FACE RECOGNITION ATTENDANCE SYSTEM", 
                              bg='#2c3e50', fg='#ecf0f1', font=('Arial', 18, 'bold'))
        title_label.pack(pady=10)
        
        # Device info
        device_label = tk.Label(main_frame, text=f"Using: {self.device}", 
                               bg='#2c3e50', fg='#f39c12', font=('Arial', 10))
        device_label.pack()
        
        # Input frame
        input_frame = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        input_frame.pack(fill='x', pady=10)
        
        # Name input
        tk.Label(input_frame, text="Full Name:", bg='#34495e', fg='#ecf0f1', 
                font=('Arial', 12)).grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.std_name = tk.Entry(input_frame, font=('Arial', 12), width=25)
        self.std_name.grid(row=0, column=1, padx=10, pady=10)
        tk.Button(input_frame, text="Clear", command=self.clear_name, 
                 bg='#e74c3c', fg='white', font=('Arial', 10)).grid(row=0, column=2, padx=5)
        
        # ID input
        tk.Label(input_frame, text="Student ID:", bg='#34495e', fg='#ecf0f1', 
                font=('Arial', 12)).grid(row=1, column=0, padx=10, pady=10, sticky='w')
        self.std_number = tk.Entry(input_frame, font=('Arial', 12), width=25)
        self.std_number.grid(row=1, column=1, padx=10, pady=10)
        tk.Button(input_frame, text="Clear", command=self.clear_id, 
                 bg='#e74c3c', fg='white', font=('Arial', 10)).grid(row=1, column=2, padx=5)
        
        # Progress bar
        self.progress_var = tk.StringVar()
        self.progress_var.set("Ready")
        progress_label = tk.Label(main_frame, textvariable=self.progress_var, 
                                 bg='#2c3e50', fg='#f39c12', font=('Arial', 12, 'bold'))
        progress_label.pack(pady=5)
        
        self.progress = ttk.Progressbar(main_frame, length=400, mode='determinate')
        self.progress.pack(pady=5)
        
        # Notification area
        tk.Label(main_frame, text="Status & Notifications", bg='#2c3e50', fg='#e74c3c', 
                font=('Arial', 14, 'bold')).pack(pady=(20, 5))
        self.notification_text = tk.Text(main_frame, height=6, width=80, font=('Arial', 10))
        self.notification_text.pack(pady=5)
        
        # Buttons frame
        buttons_frame = tk.Frame(main_frame, bg='#2c3e50')
        buttons_frame.pack(pady=20)
        
        # Buttons
        tk.Button(buttons_frame, text="CAPTURE IMAGES\n(FaceNet Detection)", command=self.start_capture_thread,
                 bg='#27ae60', fg='white', font=('Arial', 11, 'bold'), 
                 width=15, height=3).grid(row=0, column=0, padx=10)
        
        tk.Button(buttons_frame, text="TRAIN MODEL\n(FaceNet Embeddings)", command=self.start_train_thread,
                 bg='#3498db', fg='white', font=('Arial', 11, 'bold'), 
                 width=15, height=3).grid(row=0, column=1, padx=10)
        
        tk.Button(buttons_frame, text="START TRACKING\n(Real-time)", command=self.start_track_thread,
                 bg='#f39c12', fg='white', font=('Arial', 11, 'bold'), 
                 width=15, height=3).grid(row=0, column=2, padx=10)
        
        tk.Button(buttons_frame, text="VIEW ATTENDANCE", command=self.view_attendance,
                 bg='#9b59b6', fg='white', font=('Arial', 11, 'bold'), 
                 width=15, height=3).grid(row=0, column=3, padx=10)
        
    def log_message(self, message):
        """Add message to notification area"""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        self.notification_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.notification_text.see(tk.END)
        self.window.update()
        
    def clear_name(self):
        self.std_name.delete(0, 'end')
        
    def clear_id(self):
        self.std_number.delete(0, 'end')
        
    def validate_face_quality_mtcnn(self, face_tensor, confidence):
        """Validate face quality using MTCNN confidence and tensor properties"""
        if face_tensor is None:
            return False, "No face detected"
            
        # Check MTCNN confidence
        if confidence < 0.9:
            return False, f"Low detection confidence: {confidence:.2f}"
            
        # Check tensor properties
        if face_tensor.shape[0] != 3 or face_tensor.shape[1] < 160 or face_tensor.shape[2] < 160:
            return False, "Invalid face dimensions"
            
        # Check if image is too dark or bright
        mean_val = torch.mean(face_tensor).item()
        if mean_val < -0.8 or mean_val > 0.8:  # Normalized values
            return False, "Poor lighting conditions"
            
        return True, "Good quality"
        
    def detect_faces_mtcnn(self, frame):
        """Detect faces using MTCNN"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Detect faces and get bounding boxes and probabilities
            boxes, probs = self.mtcnn.detect(pil_image)
            
            if boxes is not None:
                # Convert to format compatible with OpenCV
                faces = []
                confidences = []
                for box, prob in zip(boxes, probs):
                    if prob > 0.8:  # Confidence threshold
                        x1, y1, x2, y2 = box.astype(int)
                        w = x2 - x1
                        h = y2 - y1
                        faces.append((x1, y1, w, h))
                        confidences.append(prob)
                return faces, confidences
            else:
                return [], []
                
        except Exception as e:
            self.log_message(f"MTCNN detection error: {e}")
            return [], []
        
    def extract_face_embedding(self, frame, box):
        """Extract face embedding using FaceNet"""
        try:
            # Validate input parameters
            if frame is None or len(box) != 4:
                return None
                
            x, y, w, h = box
            
            # Ensure coordinates are within frame bounds
            frame_height, frame_width = frame.shape[:2]
            x = max(0, min(x, frame_width - 1))
            y = max(0, min(y, frame_height - 1))
            w = max(1, min(w, frame_width - x))
            h = max(1, min(h, frame_height - y))
            
            # Extract face region first to validate
            face_region = frame[y:y+h, x:x+w]
            if face_region.size == 0 or face_region.shape[0] < 10 or face_region.shape[1] < 10:
                return None
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Create bounding box for MTCNN
            bbox = [x, y, x+w, y+h]
            
            # Extract and preprocess face
            face_tensor = self.mtcnn.extract(pil_image, [bbox], save_path=None)
            
            if face_tensor is not None and len(face_tensor) > 0:
                face_tensor = face_tensor[0]
                
                # Validate tensor
                if face_tensor is None or torch.isnan(face_tensor).any():
                    return None
                    
                face_tensor = face_tensor.unsqueeze(0).to(self.device)
                
                # Generate embedding
                with torch.no_grad():
                    embedding = self.resnet(face_tensor)
                    embedding = F.normalize(embedding, p=2, dim=1)
                    
                return embedding.cpu().numpy().flatten()
            else:
                return None
                
        except Exception as e:
            self.log_message(f"Embedding extraction error: {e}")
            return None
        
    def capture_images(self):
        """Enhanced image capture with MTCNN detection"""
        name = self.std_name.get().strip()
        student_id = self.std_number.get().strip()
        
        if not name or not student_id:
            messagebox.showerror("Error", "Please enter both name and student ID")
            return
            
        if not name.replace(" ", "").isalpha():
            messagebox.showerror("Error", "Name should contain only letters")
            return
            
        self.log_message(f"Starting FaceNet image capture for {name} (ID: {student_id})")
        
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            messagebox.showerror("Error", "Cannot access camera")
            return
            
        # Set camera properties for better quality
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_FPS, 30)
        
        sample_num = 0
        target_samples = 100 # Reduced since FaceNet is more efficient
        quality_samples = 0
        
        # Statistics tracking
        total_frames = 0
        faces_detected = 0
        
        self.progress['maximum'] = target_samples
        self.progress['value'] = 0
        
        while sample_num < target_samples:
            ret, frame = cam.read()
            if not ret:
                self.log_message("Failed to capture frame")
                break
                
            total_frames += 1
            faces, confidences = self.detect_faces_mtcnn(frame)
            
            for i, ((x, y, w, h), confidence) in enumerate(zip(faces, confidences)):
                faces_detected += 1
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Extract face region for saving
                y1 = max(0, y)
                y2 = min(frame.shape[0], y + h)
                x1 = max(0, x)
                x2 = min(frame.shape[1], x + w)
                face_region = frame[y1:y2, x1:x2]
                
                # Validate face quality using MTCNN confidence
                if confidence > 0.95 and face_region.size > 0 and face_region.shape[0] > 10 and face_region.shape[1] > 10:
                    sample_num += 1
                    quality_samples += 1
                    
                    # Resize and save image
                    try:
                        face_resized = cv2.resize(face_region, (160, 160))  # FaceNet input size
                        filename = f"TrainingImages/{name}.{student_id}.{sample_num}.jpg"
                        cv2.imwrite(filename, face_resized)
                    except cv2.error as e:
                        self.log_message(f"Error saving image {sample_num}: {e}")
                        sample_num -= 1  # Revert counter if save failed
                        continue
                    
                    # Update progress
                    self.progress['value'] = sample_num
                    self.progress_var.set(f"Captured: {sample_num}/{target_samples} - Conf: {confidence:.3f}")
                    
                    cv2.putText(frame, f"Captured: {sample_num}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Quality: {confidence:.2f}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
            # Display frame
            cv2.putText(frame, f"Press 'q' to quit early", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('FaceNet Enhanced Face Capture', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cam.release()
        cv2.destroyAllWindows()
        
        # Save student details
        if sample_num > 0:
            row = [student_id, name, sample_num, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            
            # Create CSV if it doesn't exist
            csv_file = 'student_details.csv'
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['ID', 'Name', 'Images_Captured', 'Date_Registered'])
                writer.writerow(row)
                
            detection_rate = (faces_detected / max(total_frames, 1)) * 100
            self.log_message(f"✓ Capture completed: {sample_num} images saved")
            self.log_message(f"Quality samples: {quality_samples}, Detection rate: {detection_rate:.1f}%")
            self.progress_var.set("Capture completed successfully!")
        else:
            self.log_message("✗ No valid images captured")
            self.progress_var.set("Capture failed - try again")
            
    def train_model(self):
        """Train model using FaceNet embeddings"""
        self.log_message("Starting FaceNet model training...")
        
        training_path = "TrainingImages"
        if not os.path.exists(training_path) or not os.listdir(training_path):
            messagebox.showerror("Error", "No training images found. Please capture images first.")
            return
            
        # Get all image files
        image_files = [f for f in os.listdir(training_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) == 0:
            messagebox.showerror("Error", "No valid training images found")
            return
            
        self.log_message(f"Processing {len(image_files)} training images...")
        
        self.progress['maximum'] = len(image_files)
        self.progress['value'] = 0
        
        embeddings_db = {}
        name_mapping = {}
        
        processed_count = 0
        
        for i, filename in enumerate(image_files):
            try:
                # Extract ID and name from filename
                parts = filename.split('.')
                if len(parts) >= 3:
                    name = parts[0]
                    student_id = int(parts[1])
                    
                    # Load image
                    image_path = os.path.join(training_path, filename)
                    frame = cv2.imread(image_path)
                    
                    if frame is not None:
                        # Use the entire image as face (since we saved cropped faces)
                        h, w = frame.shape[:2]
                        box = (0, 0, w, h)
                        
                        # Extract embedding
                        embedding = self.extract_face_embedding(frame, box)
                        
                        if embedding is not None:
                            if student_id not in embeddings_db:
                                embeddings_db[student_id] = []
                                name_mapping[student_id] = name
                                
                            embeddings_db[student_id].append(embedding)
                            processed_count += 1
                            
                # Update progress
                self.progress['value'] = i + 1
                self.progress_var.set(f"Processing: {i+1}/{len(image_files)}")
                self.window.update()
                
            except Exception as e:
                self.log_message(f"Error processing {filename}: {e}")
                continue
                
        if not embeddings_db:
            messagebox.showerror("Error", "No valid embeddings generated")
            return
            
        # Calculate average embeddings for each person
        self.progress_var.set("Calculating average embeddings...")
        average_embeddings = {}
        
        for student_id, embeddings_list in embeddings_db.items():
            if embeddings_list:
                # Calculate mean embedding
                embeddings_array = np.array(embeddings_list)
                average_embedding = np.mean(embeddings_array, axis=0)
                # Normalize
                average_embedding = average_embedding / np.linalg.norm(average_embedding)
                average_embeddings[student_id] = average_embedding
                
        # Save embeddings and name mapping
        try:
            np.save("embeddings/face_embeddings.npy", average_embeddings)
            np.save("embeddings/name_mapping.npy", name_mapping)
            
            self.progress['value'] = len(image_files)
            self.log_message("✓ FaceNet model training completed successfully")
            self.log_message(f"✓ Generated embeddings for {len(average_embeddings)} people")
            self.log_message(f"✓ Processed {processed_count}/{len(image_files)} images")
            
        except Exception as e:
            self.log_message(f"✗ Failed to save model: {str(e)}")
            return
                
        self.progress_var.set("Training completed!")
        messagebox.showinfo("Success", f"Training completed! Generated embeddings for {len(average_embeddings)} people.")
        
    def recognize_face_facenet(self, frame, box):
        """Recognize face using FaceNet embeddings"""
        try:
            # Load embeddings and name mapping
            embeddings_path = "embeddings/face_embeddings.npy"
            names_path = "embeddings/name_mapping.npy"
            
            if not os.path.exists(embeddings_path) or not os.path.exists(names_path):
                return None, 0
                
            stored_embeddings = np.load(embeddings_path, allow_pickle=True).item()
            name_mapping = np.load(names_path, allow_pickle=True).item()
            
            # Extract embedding for current face
            current_embedding = self.extract_face_embedding(frame, box)
            
            if current_embedding is None:
                return None, 0
                
            best_match_id = None
            best_similarity = 0
            
            # Compare with stored embeddings
            for student_id, stored_embedding in stored_embeddings.items():
                # Calculate cosine similarity
                similarity = np.dot(current_embedding, stored_embedding) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(stored_embedding)
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = student_id
                    
            # Convert similarity to confidence (0-100)
            confidence = best_similarity * 100
            
            return best_match_id, confidence
            
        except Exception as e:
            self.log_message(f"Recognition error: {str(e)}")
            return None, 0
        
    def track_attendance(self):
        """Enhanced real-time attendance tracking using FaceNet"""
        embeddings_path = "embeddings/face_embeddings.npy"
        names_path = "embeddings/name_mapping.npy"
        
        if not os.path.exists(embeddings_path) or not os.path.exists(names_path):
            messagebox.showerror("Error", "No trained model found. Please train the model first.")
            return
        
        # Load embeddings and name mapping
        try:
            stored_embeddings = np.load(embeddings_path, allow_pickle=True).item()
            name_mapping = np.load(names_path, allow_pickle=True).item()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
            return
        
        self.log_message("Starting FaceNet attendance tracking...")
        self.log_message(f"Loaded embeddings for {len(stored_embeddings)} people")
        
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            messagebox.showerror("Error", "Cannot access camera")
            return
            
        # Attendance tracking variables
        attendance_today = defaultdict(int)
        last_recognition = {}
        recognition_cooldown = 5  # seconds
        
        # Create attendance file with headers if it doesn't exist
        attendance_file = 'attendance.csv'
        if not os.path.exists(attendance_file):
            with open(attendance_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['ID', 'Name', 'Date', 'Time', 'Status'])
                
        while True:
            ret, frame = cam.read()
            if not ret:
                break
                
            faces, confidences = self.detect_faces_mtcnn(frame)
            current_time = time.time()
            
            for (x, y, w, h), detection_conf in zip(faces, confidences):
                if detection_conf > 0.9:  # Only process high-confidence detections
                    # Recognize face using FaceNet
                    student_id, recognition_conf = self.recognize_face_facenet(frame, (x, y, w, h))
                    
                    # Enhanced confidence threshold (85% similarity for FaceNet)
                    if student_id is not None and recognition_conf > 85:
                        name = name_mapping.get(student_id, "Unknown")
                        
                        # Check cooldown period
                        if (student_id not in last_recognition or 
                            current_time - last_recognition[student_id] > recognition_cooldown):
                            
                            # Record attendance
                            timestamp = datetime.datetime.now()
                            date_str = timestamp.strftime('%Y-%m-%d')
                            time_str = timestamp.strftime('%H:%M:%S')
                            
                            # Save to file
                            with open(attendance_file, 'a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([student_id, name, date_str, time_str, 'Present'])
                                
                            attendance_today[student_id] += 1
                            last_recognition[student_id] = current_time
                            
                            self.log_message(f"✓ Attendance: {name} (ID: {student_id}, Conf: {recognition_conf:.1f}%)")
                            
                        # Draw rectangle and label
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        label = f"{name} ({recognition_conf:.1f}%)"
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                    else:
                        # Unknown person or low confidence
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        conf_text = f"{recognition_conf:.1f}%" if student_id is not None else "Unknown"
                        cv2.putText(frame, f"Unknown ({conf_text})", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Save unknown face if confidence is very low
                        if recognition_conf < 70:
                            # Extract face region with bounds checking
                            y1 = max(0, y)
                            y2 = min(frame.shape[0], y + h)
                            x1 = max(0, x)
                            x2 = min(frame.shape[1], x + w)
                            
                            face_region = frame[y1:y2, x1:x2]
                            
                            # Only save if face region is valid
                            if face_region.size > 0 and face_region.shape[0] > 10 and face_region.shape[1] > 10:
                                unknown_count = len([f for f in os.listdir("ImagesUnknown") if f.endswith('.jpg')]) + 1
                                unknown_path = f"ImagesUnknown/unknown_{unknown_count}.jpg"
                                try:
                                    cv2.imwrite(unknown_path, face_region)
                                except cv2.error as e:
                                    self.log_message(f"Warning: Could not save unknown face: {e}")
                            
            # Display statistics
            stats_text = f"FaceNet Tracking - Today: {len(attendance_today)} people"
            cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('FaceNet Face Recognition Attendance', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cam.release()
        cv2.destroyAllWindows()
        self.log_message(f"FaceNet tracking stopped. Total attendance: {len(attendance_today)}")
        
    def view_attendance(self):
        """View attendance records"""
        attendance_file = 'attendance.csv'
        if os.path.exists(attendance_file):
            try:
                df = pd.read_csv(attendance_file)
                if not df.empty:
                    # Show recent records
                    recent_records = df.tail(20)
                    
                    # Create new window
                    attendance_window = tk.Toplevel(self.window)
                    attendance_window.title("Recent Attendance Records")
                    attendance_window.geometry("600x400")
                    
                    # Create text widget with scrollbar
                    text_frame = tk.Frame(attendance_window)
                    text_frame.pack(fill='both', expand=True, padx=10, pady=10)
                    
                    scrollbar = tk.Scrollbar(text_frame)
                    scrollbar.pack(side='right', fill='y')
                    
                    text_widget = tk.Text(text_frame, yscrollcommand=scrollbar.set)
                    text_widget.pack(side='left', fill='both', expand=True)
                    scrollbar.config(command=text_widget.yview)
                    
                    # Display records
                    text_widget.insert('1.0', recent_records.to_string(index=False))
                    text_widget.config(state='disabled')
                    
                else:
                    messagebox.showinfo("Info", "No attendance records found")
            except Exception as e:
                messagebox.showerror("Error", f"Error reading attendance file: {str(e)}")
        else:
            messagebox.showinfo("Info", "No attendance file found")
            
    def start_capture_thread(self):
        threading.Thread(target=self.capture_images, daemon=True).start()
        
    def start_train_thread(self):
        threading.Thread(target=self.train_model, daemon=True).start()
        
    def start_track_thread(self):
        threading.Thread(target=self.track_attendance, daemon=True).start()
        
    def run(self):
        self.log_message("FaceNet Enhanced Attendance System Started")
        self.log_message("Using state-of-the-art deep learning for face recognition")
        self.log_message("Steps: 1. Capture Images → 2. Train Model → 3. Start Tracking")
        self.window.mainloop()

if __name__ == "__main__":
    app = AttendanceSystem()
    app.run()