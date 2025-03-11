import mediapipe as mp
import os

# Mediapipe modellarini oldindan yuklab olish
mp.solutions.holistic._download_oss_pose_landmark_model(0)
mp.solutions.holistic._download_oss_face_landmark_model()
mp.solutions.holistic._download_oss_hand_landmark_model()



import cv2
import av
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import threading

# Mediapipe kutubxonalari
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

class PoseDetector(VideoProcessorBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,  # Engilroq model
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.previous_landmarks = {}
        self.detected_movement = "Harakat aniqlanmadi"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
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
        
        if result.face_landmarks:
            current_landmarks["face"] = result.face_landmarks.landmark
            mp_drawing.draw_landmarks(img, result.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        
        # Harakatni aniqlash
        for part, landmarks in current_landmarks.items():
            prev_landmarks = self.previous_landmarks.get(part, None)
            if prev_landmarks:
                diffs = [
                    np.linalg.norm(np.array([lm.x, lm.y, lm.z]) - np.array([prev_landmarks[i].x, prev_landmarks[i].y, prev_landmarks[i].z]))
                    for i, lm in enumerate(landmarks) if i < len(prev_landmarks)
                ]
                if diffs:  # Check if diffs is not empty
                    avg_movement = np.mean(diffs)
                    if avg_movement > 0.01:
                        moving_parts.append(part)
        
        self.detected_movement = ", ".join(moving_parts) if moving_parts else "Harakat aniqlanmadi"
        
        # Oldingi kadr ma'lumotlarini yangilash
        self.previous_landmarks.clear()
        self.previous_landmarks.update(current_landmarks)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def __del__(self):
        """Obyekt o'chirilganda Mediapipe obyektini yopish"""
        self.holistic.close()

def main():
    st.title("Mobil Kamera bilan Harakatni Aniqlash")
    
    # WebRTC komponentini ishga tushirish
    webrtc_ctx = webrtc_streamer(
        key="pose-detection",
        video_processor_factory=PoseDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,  # Change to True for better performance
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )
    
    # Harakatlarni chiqarish uchun joy
    movement_display = st.empty()
    
    # Obyekt mavjudligini tekshirib harakatlarni ko'rsatish
    if webrtc_ctx.state.playing:
        while True:
            if webrtc_ctx.video_processor:
                movement_display.write(f"**Aniqlangan harakatlar:** {webrtc_ctx.video_processor.detected_movement}")
                st.session_state['movement'] = webrtc_ctx.video_processor.detected_movement
                st.experimental_rerun()
            st.stop()

if __name__ == "__main__":
    if 'movement' not in st.session_state:
        st.session_state['movement'] = "Harakat aniqlanmadi"
    main()