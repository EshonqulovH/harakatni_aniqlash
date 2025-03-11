import cv2
import av
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import threading
import os
import tempfile

# Mediapipe kutubxonalari
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Set MediaPipe model cache directory to a writable location
temp_dir = tempfile.gettempdir()
os.environ["MEDIAPIPE_MODEL_PATH"] = temp_dir

class PoseDetector(VideoProcessorBase):
    def __init__(self):
        try:
            self.holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,  # Engilroq model
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
                enable_segmentation=False  # Disable segmentation to reduce resource usage
            )
            self.previous_landmarks = {}
            self.detected_movement = "Harakat aniqlanmadi"
            self.error_message = None
        except Exception as e:
            st.error(f"Holistic model initialization error: {str(e)}")
            self.error_message = str(e)
            self.holistic = None

    def recv(self, frame):
        if self.holistic is None:
            img = frame.to_ndarray(format="bgr24")
            # Draw error message on frame
            cv2.putText(img, "Model initialization failed", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(img, self.error_message[:50], (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # Resize image to reduce processing load (optional)
            # img = cv2.resize(img, (640, 480))
            
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self.holistic.process(rgb_frame)
            
            current_landmarks = {}
            moving_parts = []
            
            if result.pose_landmarks:
                current_landmarks["pose"] = result.pose_landmarks.landmark
                mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            if result.left_hand_landmarks:
                current_landmarks["left_hand"] = result.left_hand_landmarks.landmark
                mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            if result.right_hand_landmarks:
                current_landmarks["right_hand"] = result.right_hand_landmarks.landmark
                mp_drawing.draw_landmarks(img, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Face detection is resource-intensive, you can comment this out for better performance
            if result.face_landmarks:
                current_landmarks["face"] = result.face_landmarks.landmark
                mp_drawing.draw_landmarks(img, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS)  # Less detailed mesh
            
            # Harakatni aniqlash
            for part, landmarks in current_landmarks.items():
                prev_landmarks = self.previous_landmarks.get(part, None)
                if prev_landmarks:
                    diffs = []
                    min_len = min(len(landmarks), len(prev_landmarks))
                    for i in range(min_len):
                        try:
                            diff = np.linalg.norm(
                                np.array([landmarks[i].x, landmarks[i].y, landmarks[i].z]) - 
                                np.array([prev_landmarks[i].x, prev_landmarks[i].y, prev_landmarks[i].z])
                            )
                            diffs.append(diff)
                        except (IndexError, AttributeError):
                            continue
                    
                    if diffs:  # Check if diffs is not empty
                        avg_movement = np.mean(diffs)
                        if avg_movement > 0.01:
                            moving_parts.append(part)
            
            self.detected_movement = ", ".join(moving_parts) if moving_parts else "Harakat aniqlanmadi"
            
            # Display movement info on frame
            cv2.putText(img, f"Harakat: {self.detected_movement}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Oldingi kadr ma'lumotlarini yangilash
            self.previous_landmarks.clear()
            self.previous_landmarks.update(current_landmarks)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(img, f"Error: {str(e)[:50]}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def __del__(self):
        if hasattr(self, 'holistic') and self.holistic:
            self.holistic.close()

def main():
    st.title("Mobil Kamera bilan Harakatni Aniqlash")
    
    st.warning("""
    Eslatma: 
    1. Kameranga ruxsat berishingiz kerak
    2. Agar ilovada xatolik yuz bersa, brauzeringizni yangilang
    3. Chrome yoki Edge brauzerini ishlatish tavsiya etiladi
    """)
    
    # Streamlit sidebar settings
    st.sidebar.header("Sozlamalar")
    st.sidebar.info("Kamerangiz ochilishini kuting")
    
    # Motion detection threshold (optional setting)
    threshold = st.sidebar.slider("Harakatni aniqlash sezgirligi", 0.005, 0.05, 0.01, 0.001)
    
    try:
        # WebRTC komponentini ishga tushirish
        webrtc_ctx = webrtc_streamer(
            key="pose-detection",
            video_processor_factory=PoseDetector,
            media_stream_constraints={
                "video": {
                    "frameRate": {"ideal": 10, "max": 30},  # Lower framerate
                    "width": {"ideal": 480},
                    "height": {"ideal": 360}
                },
                "audio": False
            },
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )
        
        # Harakatlarni chiqarish uchun joy
        movement_placeholder = st.empty()
        
        # Obyekt mavjudligini tekshirib harakatlarni ko'rsatish
        if webrtc_ctx.state.playing:
            st.success("Kamera muvaffaqiyatli ochildi!")
        
        if "movement" not in st.session_state:
            st.session_state["movement"] = "Harakat aniqlanmadi"
            
        # Thread to update movement text
        def update_movement():
            while webrtc_ctx.state.playing:
                try:
                    if webrtc_ctx.video_processor and hasattr(webrtc_ctx.video_processor, "detected_movement"):
                        movement = webrtc_ctx.video_processor.detected_movement
                        movement_placeholder.subheader(f"Aniqlangan harakatlar: {movement}")
                        st.session_state["movement"] = movement
                except Exception as e:
                    st.error(f"Xatolik: {e}")
                    break
        
        if webrtc_ctx.state.playing:
            threading.Thread(target=update_movement, daemon=True).start()
            
    except Exception as e:
        st.error(f"Ilovada xatolik yuz berdi: {e}")
        st.info("Brauzeringizni yangilang yoki Chrome/Edge ishlatib ko'ring")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Harakatni Aniqlash",
        page_icon="🎥",
        layout="wide"
    )
    main()