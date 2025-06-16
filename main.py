import streamlit as st
import cv2
import numpy as np
import os
import pickle
import pandas as pd
from datetime import datetime
import threading
import time
from queue import Queue

class FaceRecognitionSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Paths
        self.dataset_path, self.model_path, self.labels_path, self.attendance_path = "dataset", "face_model.yml", "labels.pkl", "attendance.csv"
        
        # Video processing
        self.video_thread, self.stop_video, self.frame_queue = None, False, Queue(maxsize=3)
        
        # Capture settings
        self.capture_mode, self.capture_active, self.captured_count = False, False, 0
        self.target_samples, self.current_person_label, self.last_capture_time, self.min_capture_interval = 30, "", 0, 0.3
        
        # Recognition & Attendance - FIXED LOGIC
        self.attendance_today, self.last_recognition_time, self.recognition_cooldown = {}, {}, 10
        self.detection_success, self.last_detected_person, self.model_loaded = False, "", False
        self.recent_detections = []  # For dashboard
        
        self._setup_directories()
        self._load_today_attendance()
        self._auto_load_model()

    def _auto_load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.labels_path):
            try:
                self.recognizer.read(self.model_path)
                with open(self.labels_path, 'rb') as f:
                    self.label_dict = pickle.load(f)
                self.model_loaded = True
            except:
                self.model_loaded = False
    
    def _setup_directories(self):
        os.makedirs(self.dataset_path, exist_ok=True)
    
    def _load_today_attendance(self):
        """Load today's attendance records - FIXED to track actual status"""
        self.attendance_today = {}  # Changed from set to dict
        if os.path.exists(self.attendance_path):
            try:
                df = pd.read_csv(self.attendance_path)
                today = datetime.now().strftime('%Y-%m-%d')
                today_records = df[df['Date'] == today]
                
                # Get the latest status for each person today
                for name in today_records['Name'].unique():
                    person_records = today_records[today_records['Name'] == name].sort_values('Time')
                    if not person_records.empty:
                        latest_status = person_records.iloc[-1]['Status']
                        self.attendance_today[name] = latest_status
            except:
                self.attendance_today = {}
    
    def log_attendance(self, name, confidence):
        """FIXED: Only log if person hasn't been marked today or needs status change"""
        now = datetime.now()
        
        # Check cooldown to prevent spam detections
        if name in self.last_recognition_time:
            if (now - self.last_recognition_time[name]).seconds < self.recognition_cooldown:
                return False
        
        self.last_recognition_time[name] = now
        date, time_str = now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')
        
        # FIXED LOGIC: Determine status based on current attendance
        current_status = self.attendance_today.get(name, None)
        
        if current_status is None:
            # First time today - mark as IN
            new_status = "IN"
        elif current_status == "IN":
            # Already IN - mark as OUT
            new_status = "OUT"
        else:
            # Already OUT - mark as IN
            new_status = "IN"
        
        # Update attendance tracking
        self.attendance_today[name] = new_status
        
        # Save to CSV
        df = pd.DataFrame([{
            'Name': name, 
            'Date': date, 
            'Time': time_str, 
            'Confidence': confidence, 
            'Status': new_status
        }])
        df.to_csv(self.attendance_path, mode='a', header=not os.path.exists(self.attendance_path), index=False)
        
        # Update dashboard
        self.detection_success = True
        self.last_detected_person = f"{name.replace('_', ' ').title()} - {new_status}"
        
        # Add to recent detections (keep last 10) - AVOID DUPLICATES
        detection_entry = {
            'name': name.replace('_', ' ').title(), 
            'time': time_str, 
            'status': new_status, 
            'confidence': confidence
        }
        
        # Remove any existing entry for this person and add new one at top
        self.recent_detections = [d for d in self.recent_detections if d['name'] != detection_entry['name']]
        self.recent_detections.insert(0, detection_entry)
        self.recent_detections = self.recent_detections[:10]
        
        return True
    
    def get_person_folders(self):
        return sorted([item for item in os.listdir(self.dataset_path) 
                      if os.path.isdir(os.path.join(self.dataset_path, item))]) if os.path.exists(self.dataset_path) else []
    
    def create_person_folder(self, person_name):
        folder_path = os.path.join(self.dataset_path, person_name.replace(" ", "_").lower())
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    def detect_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), maxSize=(300, 300))
        return faces, gray
    
    def prepare_training_data(self):
        faces, labels, label_dict = [], [], {}
        for idx, folder_name in enumerate(self.get_person_folders()):
            label_dict[idx] = folder_name
            folder_path = os.path.join(self.dataset_path, folder_name)
            
            for image_name in os.listdir(folder_path):
                if image_name.endswith(('.jpg', '.jpeg', '.png')):
                    img = cv2.imread(os.path.join(folder_path, image_name))
                    if img is not None:
                        detected_faces, gray = self.detect_faces(img)
                        for (x, y, w, h) in detected_faces:
                            faces.append(cv2.resize(gray[y:y+h, x:x+w], (200, 200)))
                            labels.append(idx)
        
        with open(self.labels_path, 'wb') as f:
            pickle.dump(label_dict, f)
        return faces, labels, label_dict
    
    def train_model(self):
        faces, labels, label_dict = self.prepare_training_data()
        if not faces:
            return False, "No faces found in dataset"
        
        self.recognizer.train(faces, np.array(labels))
        self.recognizer.save(self.model_path)
        return True, f"Model trained with {len(faces)} face samples from {len(label_dict)} people"
    
    def load_model(self):
        if not (os.path.exists(self.model_path) and os.path.exists(self.labels_path)):
            return False, "Model not found. Train the model first."
        
        try:
            self.recognizer.read(self.model_path)
            with open(self.labels_path, 'rb') as f:
                self.label_dict = pickle.load(f)
            self.model_loaded = True
            return True, "Model loaded successfully"
        except Exception as e:
            self.model_loaded = False
            return False, f"Error loading model: {e}"
    
    def recognize_face(self, img):
        faces, gray = self.detect_faces(img)
        results = []
        
        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, confidence = self.recognizer.predict(face_roi)
            
            if confidence < 70:
                name = self.label_dict.get(label, "Unknown")
                confidence_percent = round(100 - confidence, 2)
            else:
                name = "Unknown Face"
                confidence_percent = 0
            
            results.append({'name': name, 'confidence': confidence_percent, 'bbox': (x, y, w, h)})
        return results
    
    def get_video_source(self, source_type, source_path=None):
        if source_type == "Webcam":
            cap = cv2.VideoCapture(0)
        else:
            if not source_path:
                return None, "IP camera URL required"
            if not source_path.startswith(('http://', 'https://', 'rtsp://')):
                source_path = f"http://{source_path}:8080/video"
            cap = cv2.VideoCapture(source_path)
        
        return (cap, f"{source_type}: {source_path or 'Default'}") if cap.isOpened() else (None, f"Cannot connect to {source_type}")
    
    def video_capture_thread(self, source_type, source_path=None):
        cap, source_info = self.get_video_source(source_type, source_path)
        if cap is None:
            return
        
        # Optimize settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Warm up
        for _ in range(5):
            cap.read()
        
        while not self.stop_video:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            if source_type == "Webcam":
                frame = cv2.flip(frame, 1)
            
            current_time = time.time()
            
            if self.capture_mode and self.capture_active:
                self._handle_capture_mode(frame, current_time)
            elif not self.capture_mode and self.model_loaded and hasattr(self, 'label_dict'):
                self._handle_recognition_mode(frame)
            
            # Queue frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame_rgb, block=False)
            except:
                pass
            
            time.sleep(0.066)
        
        cap.release()
    
    def _handle_capture_mode(self, frame, current_time):
        faces, gray = self.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if (current_time - self.last_capture_time >= self.min_capture_interval and 
                self.captured_count < self.target_samples and w > 60 and h > 60):
                
                person_folder = self.create_person_folder(self.current_person_label)
                cv2.imwrite(os.path.join(person_folder, f"sample_{int(current_time * 1000)}_{self.captured_count}.jpg"), 
                           gray[y:y+h, x:x+w])
                
                self.captured_count += 1
                self.last_capture_time = current_time
        
        self._draw_capture_ui(frame, len(faces))
        if self.captured_count >= self.target_samples:
            self.capture_active = False
    
    def _handle_recognition_mode(self, frame):
        results = self.recognize_face(frame)
        
        for result in results:
            x, y, w, h = result['bbox']
            name, confidence = result['name'], result['confidence']
            
            if confidence > 30 and name != "Unknown Face":
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # FIXED: Show next expected status
                current_status = self.attendance_today.get(name, None)
                if current_status is None:
                    next_status = "IN"
                elif current_status == "IN":
                    next_status = "OUT"
                else:
                    next_status = "IN"
                
                label = f"{name.replace('_', ' ').title()}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                self.log_attendance(name, confidence)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def _draw_capture_ui(self, frame, face_count):
        cv2.rectangle(frame, (5, 5), (350, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Capturing: {self.current_person_label}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Progress: {self.captured_count}/{self.target_samples}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {face_count}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        progress_width = int(300 * (self.captured_count / self.target_samples))
        cv2.rectangle(frame, (10, 85), (310, 95), (100, 100, 100), -1)
        cv2.rectangle(frame, (10, 85), (10 + progress_width, 95), (0, 255, 0), -1)
    
    def start_video_stream(self, source_type, source_path=None):
        self.stop_video, self.capture_mode, self.detection_success, self.last_detected_person = False, False, False, ""
        self._start_thread(source_type, source_path)
    
    def start_dataset_capture(self, source_type, source_path=None, num_samples=30, person_label="unknown"):
        self.stop_video, self.capture_mode, self.capture_active = False, True, True
        self.captured_count, self.target_samples, self.last_capture_time = 0, num_samples, 0
        self.current_person_label = person_label.strip()
        
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        self._start_thread(source_type, source_path)
    
    def _start_thread(self, source_type, source_path):
        self.video_thread = threading.Thread(target=self.video_capture_thread, args=(source_type, source_path))
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def stop_video_stream(self):
        self.stop_video, self.capture_active = True, False
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
        
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
    
    def get_latest_frame(self):
        frame = None
        while not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
            except:
                break
        return frame
    
    def get_capture_status(self):
        return {
            'active': self.capture_active,
            'captured': self.captured_count,
            'target': self.target_samples,
            'person': self.current_person_label,
            'progress': self.captured_count / self.target_samples if self.target_samples > 0 else 0
        }
    
    def get_attendance_data(self):
        return pd.read_csv(self.attendance_path) if os.path.exists(self.attendance_path) else pd.DataFrame(columns=['Name', 'Date', 'Time', 'Confidence', 'Status'])

def main():
    st.set_page_config(page_title="Face Recognition & Attendance", layout="wide")
    
    # Initialize system
    if 'face_system' not in st.session_state:
        st.session_state.face_system = FaceRecognitionSystem()
        st.session_state.video_active = False
    
    face_system = st.session_state.face_system
    
    # Success message
    if face_system.last_detected_person:
        st.success(f"✅ Detection: {face_system.last_detected_person}")
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        face_system.recognition_cooldown = st.slider("Cooldown (seconds):", 5, 300, face_system.recognition_cooldown)
    
    st.title("🔍 Face Recognition & Attendance System")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Dataset", "🎯 Training", "📹 Recognition", "📊 Attendance"])
    
    with tab1:
        st.header("Dataset Management")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Capture Dataset")
            person_name = st.text_input("Person Name:", placeholder="Enter person's name")
            
            existing_people = face_system.get_person_folders()
            if existing_people:
                st.write("**Existing people:**")
                for person in existing_people:
                    folder_path = os.path.join(face_system.dataset_path, person)
                    image_count = len([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                    st.write(f"• {person.replace('_', ' ').title()}: {image_count} images")
            
            source_type = st.selectbox("Video Source:", ["Webcam", "IP Camera"])
            ip_source = st.text_input("IP Camera URL:", placeholder="192.168.1.100:8080") if source_type == "IP Camera" else None
            num_samples = st.slider("Samples to capture:", 10, 1000, 30)
            
            col_start, col_stop = st.columns(2)
            with col_start:
                if st.button("📸 Start Capture", disabled=st.session_state.video_active or not person_name.strip()):
                    face_system.start_dataset_capture(source_type, ip_source, num_samples, person_name.strip())
                    st.session_state.video_active = True
                    st.rerun()
            
            with col_stop:
                if st.button("⏹️ Stop Capture", disabled=not st.session_state.video_active):
                    face_system.stop_video_stream()
                    st.session_state.video_active = False
                    st.rerun()
        
        with col2:
            st.subheader("Upload Images")
            uploaded_files = st.file_uploader("Upload face images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
            
            if uploaded_files:
                upload_person_name = st.text_input("Person Name for uploads:", key="upload_name")
                
                if st.button("Upload Images") and upload_person_name.strip():
                    person_folder = face_system.create_person_folder(upload_person_name.strip())
                    for i, file in enumerate(uploaded_files):
                        with open(os.path.join(person_folder, f"upload_{int(time.time() * 1000)}_{i}.jpg"), "wb") as f:
                            f.write(file.getbuffer())
                    st.success(f"Uploaded {len(uploaded_files)} images for {upload_person_name}")
        
        # Live capture display
        if st.session_state.video_active and face_system.capture_mode:
            st.subheader("Live Capture")
            video_placeholder = st.empty()
            progress_placeholder = st.empty()
            
            while st.session_state.video_active and face_system.capture_active:
                frame = face_system.get_latest_frame()
                if frame is not None:
                    video_placeholder.image(frame, channels="RGB", width=640)
                
                status = face_system.get_capture_status()
                progress_placeholder.progress(status['progress'], f"Captured: {status['captured']}/{status['target']} samples")
                
                if status['captured'] >= status['target']:
                    face_system.stop_video_stream()
                    st.session_state.video_active = False
                    st.success(f"🎉 Completed! Captured {status['captured']} samples")
                    break
                
                time.sleep(0.1)
    
    with tab2:
        st.header("Model Training")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Train Model", type="primary"):
                with st.spinner("Training..."):
                    success, message = face_system.train_model()
                    if success:
                        st.success(message)
                        face_system._auto_load_model()
                    else:
                        st.error(message)
        
        with col2:
            if st.button("📂 Load Model"):
                success, message = face_system.load_model()
                st.success(message) if success else st.error(message)
        
        if face_system.model_loaded and hasattr(face_system, 'label_dict'):
            st.success("**Model Status**: ✅ Active")
            st.info(f"**Trained People**: {len(face_system.label_dict)}")
            
            if face_system.label_dict:
                st.write("**People in model:** " + ", ".join([name.replace('_', ' ').title() for name in face_system.label_dict.values()]))
        else:
            st.info("**Model Status**: ❌ Not loaded")
    
    with tab3:
        st.header("Real-time Recognition")
        
        if not face_system.model_loaded:
            st.warning("⚠️ Load the trained model first")
            return
        
        # Create two columns: video feed and real-time dashboard
        video_col, dashboard_col = st.columns([2, 1])
        
        with video_col:
            col1, col2 = st.columns(2)
            with col1:
                source_type = st.selectbox("Video Source:", ["Webcam", "IP Camera"], key="rec_source")
            with col2:
                if source_type == "IP Camera":
                    ip_source = st.text_input("IP Camera URL:", key="rec_ip")
                else:
                    ip_source = None
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("▶️ Start", disabled=st.session_state.video_active):
                    face_system.start_video_stream(source_type, ip_source)
                    st.session_state.video_active = True
                    st.rerun()
            
            with col2:
                if st.button("⏹️ Stop", disabled=not st.session_state.video_active):
                    face_system.stop_video_stream()
                    st.session_state.video_active = False
                    st.rerun()
            
            with col3:
                st.write("🟢 Live" if st.session_state.video_active else "🔴 Stopped")
            
            # Video display
            if st.session_state.video_active and not face_system.capture_mode:
                video_placeholder = st.empty()
        
        with dashboard_col:
            st.subheader("📊 Live Dashboard")
            
            # Current status metrics - FIXED
            col1, col2 = st.columns(2)
            with col1:
                present_count = sum(1 for status in face_system.attendance_today.values() if status == "IN")
                st.metric("Present Today", present_count)
            with col2:
                st.metric("Total People", len(face_system.label_dict) if hasattr(face_system, 'label_dict') else 0)
            
            st.subheader("Recent Detections")
            dashboard_placeholder = st.empty()
        
        # Continuous video streaming with dashboard updates
        if st.session_state.video_active and not face_system.capture_mode:
            try:
                while st.session_state.video_active:
                    frame = face_system.get_latest_frame()
                    if frame is not None:
                        video_placeholder.image(frame, channels="RGB", width=640)
                    
                    # Update dashboard
                    if face_system.recent_detections:
                        df_recent = pd.DataFrame(face_system.recent_detections)
                        df_recent['Time'] = df_recent['time']
                        df_recent['Name'] = df_recent['name']
                        df_recent['Status'] = df_recent['status'].apply(lambda x: f"{'🟢' if x == 'IN' else '🔴'} {x}")
                        # df_recent['Confidence'] = df_recent['confidence'].apply(lambda x: f"{x:.1f}%")
                        
                        dashboard_placeholder.dataframe(
                            df_recent[['Name', 'Time', 'Status']], 
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        dashboard_placeholder.info("No detections yet...")
                    
                    time.sleep(0.1)
            except:
                pass
    
    with tab4:
        st.header("Attendance Records")
        
        today = datetime.now().strftime('%Y-%m-%d')
        st.subheader(f"Today's Attendance ({today})")
        
        col1, col2 = st.columns(2)
        with col1:
            present_count = sum(1 for status in face_system.attendance_today.values() if status == "IN")
            st.metric("Present Today", present_count)
        with col2:
            if hasattr(face_system, 'label_dict'):
                st.metric("Total Registered", len(face_system.label_dict))
        
        df = face_system.get_attendance_data()
        if not df.empty:
            st.subheader("Attendance Records")
            
            col1, col2 = st.columns(2)
            with col1:
                date_filter = st.date_input("Filter by date:", value=datetime.now().date())
            with col2:
                name_filter = st.selectbox("Filter by name:", ["All"] + sorted(df['Name'].unique().tolist()))
            
            filtered_df = df.copy()
            if date_filter:
                filtered_df = filtered_df[filtered_df['Date'] == date_filter.strftime('%Y-%m-%d')]
            if name_filter != "All":
                filtered_df = filtered_df[filtered_df['Name'] == name_filter]
            
            st.dataframe(filtered_df, use_container_width=True)
            
            csv = filtered_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, f"attendance_{date_filter.strftime('%Y%m%d')}.csv", "text/csv")
        else:
            st.info("No attendance records found.")
        
        # Clear data options
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Clear Dataset"):
                import shutil
                if os.path.exists(face_system.dataset_path):
                    shutil.rmtree(face_system.dataset_path)
                    os.makedirs(face_system.dataset_path)
                st.success("Dataset cleared")
        
        with col2:
            if st.button("🗑️ Clear Model"):
                for file_path in [face_system.model_path, face_system.labels_path]:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                face_system.model_loaded = False
                st.success("Model cleared")
        
        with col3:
            if st.button("🗑️ Clear Attendance"):
                if os.path.exists(face_system.attendance_path):
                    os.remove(face_system.attendance_path)
                face_system.attendance_today.clear()
                face_system.recent_detections.clear()
                st.success("Attendance records cleared")

if __name__ == "__main__":
    main()