import os
import cv2
import json
import threading
import queue
import time
from collections import deque

class DecisionEngine2Cam(threading.Thread):
    def __init__(self, decision_queue, termination_callback=None, infraction_callback=None):
        super().__init__()
        self.decision_queue = decision_queue
        self.termination_callback = termination_callback
        self.infraction_callback = infraction_callback
        self.running = False
        self.daemon = True
        
        # State variables
        self.latest_lstm_score = 0.1
        self.latest_primary_auth_verified = True
        self.latest_primary_auth_score = 1.0
        self.latest_secondary_auth_verified = True
        self.latest_secondary_auth_score = 1.0
        self.latest_audio_speech = False
        self.latest_audio_db = 0.0
        
        # Second camera state variables
        self.latest_second_cam_connected = True
        self.latest_second_cam_ever_connected = False
        self.latest_second_cam_phone = False
        self.latest_second_cam_person = False
        
        # Rolling anomaly state
        self.rolling_anomaly_score = 0.1
        self.anomaly_threshold = 0.7
        self.continuous_violation_duration = 3.0 # seconds
        self.breach_start_time = None
        self.violation_triggered = False
        
        # Ring buffer for raw frames (150 frames @ 30 FPS = 5 seconds)
        self.buffer_lock = threading.Lock()
        self.video_buffer = deque(maxlen=150)
        
        # Telemetry history for logging
        self.telemetry_history = []
        
        self.latest_data = {
            "rolling_anomaly": 0.1,
            "seconds_in_breach": 0.0,
            "violation_triggered": False,
            "second_cam_connected": True,
            "second_cam_phone": False,
            "second_cam_person": False,
            "timestamp": 0.0
        }
        self.data_lock = threading.Lock()
        self.session_active = False

    def push_frame_to_buffer(self, frame):
        with self.buffer_lock:
            self.video_buffer.append(frame.copy())

    def get_latest_data(self):
        with self.data_lock:
            return self.latest_data.copy()

    def set_session_active(self, active):
        with self.data_lock:
            self.session_active = active
            if not active:
                self.rolling_anomaly_score = 0.1
                self.breach_start_time = None
                self.violation_triggered = False
                self.latest_data["rolling_anomaly"] = 0.1
                self.latest_data["seconds_in_breach"] = 0.0
                self.latest_data["violation_triggered"] = False

    def start(self):
        self.running = True
        super().start()

    def stop(self):
        self.running = False

    def run(self):
        os.makedirs("infractions", exist_ok=True)

        while self.running:
            try:
                event = self.decision_queue.get(timeout=0.1)
            except queue.Empty:
                self._evaluate_breach_timing()
                continue

            event_type = event.get("type")

            if event_type == "lstm":
                self.latest_lstm_score = event.get("score", 0.0)
            elif event_type == "auth":
                source = event.get("source", "primary")
                if source == "primary":
                    self.latest_primary_auth_verified = event.get("verified", True)
                    self.latest_primary_auth_score = event.get("similarity_score", 1.0)
                elif source == "secondary":
                    self.latest_secondary_auth_verified = event.get("verified", True)
                    self.latest_secondary_auth_score = event.get("similarity_score", 1.0)
            elif event_type == "audio":
                self.latest_audio_speech = event.get("speech_detected", False)
                self.latest_audio_db = event.get("db_level", 0.0)
            elif event_type == "second_cam":
                self.latest_second_cam_connected = event.get("connected", True)
                self.latest_second_cam_ever_connected = event.get("ever_connected", False)
                self.latest_second_cam_phone = event.get("phone_detected", False)
                self.latest_second_cam_person = event.get("person_detected", False)

            # Core anomaly scoring algorithm with Second Camera:
            current_anomaly = self.latest_lstm_score
            if self.latest_audio_speech:
                current_anomaly += 0.3
            if not self.latest_primary_auth_verified:
                current_anomaly += 0.5
            if self.latest_second_cam_ever_connected and self.latest_second_cam_connected and not self.latest_secondary_auth_verified:
                current_anomaly += 0.5
                
            # Second camera checks:
            if self.latest_second_cam_ever_connected:
                if not self.latest_second_cam_connected:
                    current_anomaly += 0.4
                if self.latest_second_cam_phone:
                    current_anomaly += 0.8
                if self.latest_second_cam_person:
                    current_anomaly += 0.6

            current_anomaly = min(1.0, current_anomaly)

            # EMA rolling anomaly update
            self.rolling_anomaly_score = (0.15 * current_anomaly) + (0.85 * self.rolling_anomaly_score)

            self.telemetry_history.append({
                "timestamp": time.time(),
                "lstm_score": self.latest_lstm_score,
                "auth_verified": self.latest_primary_auth_verified,
                "auth_score": self.latest_primary_auth_score,
                "primary_auth_verified": self.latest_primary_auth_verified,
                "primary_auth_score": self.latest_primary_auth_score,
                "secondary_auth_verified": self.latest_secondary_auth_verified,
                "secondary_auth_score": self.latest_secondary_auth_score,
                "audio_speech": self.latest_audio_speech,
                "audio_db": self.latest_audio_db,
                "second_cam_connected": self.latest_second_cam_connected,
                "second_cam_phone": self.latest_second_cam_phone,
                "second_cam_person": self.latest_second_cam_person,
                "current_anomaly": current_anomaly,
                "rolling_anomaly": self.rolling_anomaly_score
            })
            if len(self.telemetry_history) > 1000:
                self.telemetry_history.pop(0)

            self._evaluate_breach_timing()
            time.sleep(0.01)

    def _evaluate_breach_timing(self):
        if not self.session_active:
            self.rolling_anomaly_score = 0.1
            self.breach_start_time = None
            self.violation_triggered = False
            with self.data_lock:
                self.latest_data = {
                    "rolling_anomaly": 0.1,
                    "seconds_in_breach": 0.0,
                    "violation_triggered": False,
                    "second_cam_connected": self.latest_second_cam_connected,
                    "second_cam_phone": self.latest_second_cam_phone,
                    "second_cam_person": self.latest_second_cam_person,
                    "timestamp": time.time()
                }
            return

        seconds_in_breach = 0.0
        
        if self.rolling_anomaly_score > self.anomaly_threshold:
            if self.breach_start_time is None:
                self.breach_start_time = time.time()
            else:
                seconds_in_breach = time.time() - self.breach_start_time
                
            if seconds_in_breach >= self.continuous_violation_duration and not self.violation_triggered:
                self.violation_triggered = True
                reasons = self._get_active_anomaly_reasons()
                print(f"\n[DecisionEngine2Cam] !!! VIOLATION DETECTED !!! Reasons: {', '.join(reasons)}")
                threading.Thread(target=self._handle_infraction, args=(reasons,), daemon=True).start()
        else:
            self.breach_start_time = None
            self.violation_triggered = False  # Reset so it can trigger again on next breach

        with self.data_lock:
            self.latest_data = {
                "rolling_anomaly": float(self.rolling_anomaly_score),
                "seconds_in_breach": float(seconds_in_breach),
                "violation_triggered": self.violation_triggered,
                "second_cam_connected": self.latest_second_cam_connected,
                "second_cam_phone": self.latest_second_cam_phone,
                "second_cam_person": self.latest_second_cam_person,
                "timestamp": time.time()
            }

    def _get_active_anomaly_reasons(self):
        reasons = []
        if self.latest_lstm_score > 0.3:
            reasons.append(f"Gaze Anomaly (Score: {self.latest_lstm_score:.2f})")
        if self.latest_audio_speech:
            reasons.append("Speech Detected")
        if not self.latest_primary_auth_verified:
            reasons.append("Identity Verification Failed on Primary Camera")
        if self.latest_second_cam_ever_connected and self.latest_second_cam_connected and not self.latest_secondary_auth_verified:
            reasons.append("Identity Verification Failed on Secondary Camera")
        if self.latest_second_cam_ever_connected and not self.latest_second_cam_connected:
            reasons.append("Second Camera Disconnected")
        if self.latest_second_cam_phone:
            reasons.append("Cell Phone/Tablet Detected on Side Camera")
        if self.latest_second_cam_person:
            reasons.append("Person Detected on Side Camera")
        if not reasons:
            reasons.append("General Anomaly Index Elevation")
        return reasons

    def _handle_infraction(self, reasons):
        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        
        with self.buffer_lock:
            frames_to_save = list(self.video_buffer)
            
        # We save the video temporarily as a temp file, then merge it with audio later
        video_path = f"infractions/temp_video_{timestamp_str}.mp4"
        audio_path = f"infractions/violation_{timestamp_str}.wav"
        final_video_path = f"infractions/violation_{timestamp_str}.mp4"

        if len(frames_to_save) > 0:
            h, w, c = frames_to_save[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
            
            for frame in frames_to_save:
                # Stamp the reasons on the frames
                annotated = frame.copy()
                cv2.rectangle(annotated, (0, 0), (w, 40), (0, 0, 150), -1)
                text = "VIOLATION: " + ", ".join(reasons)
                font_scale = 0.45
                thickness = 1
                if len(text) > 60:
                    font_scale = 0.38
                cv2.putText(annotated, text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                writer.write(annotated)
            writer.release()
            print(f"[DecisionEngine2Cam] Infraction temp video saved: {video_path} ({len(frames_to_save)} frames)")
        else:
            print("[DecisionEngine2Cam] No frames in buffer to save.")

        log_path = f"infractions/telemetry_{timestamp_str}.json"
        log_data = {
            "trigger_timestamp": time.time(),
            "trigger_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "final_rolling_anomaly": self.rolling_anomaly_score,
            "anomaly_reasons": reasons,
            "last_known_metrics": {
                "lstm_sequence_score": self.latest_lstm_score,
                "primary_auth_verified": self.latest_primary_auth_verified,
                "primary_auth_score": self.latest_primary_auth_score,
                "secondary_auth_verified": self.latest_secondary_auth_verified,
                "secondary_auth_score": self.latest_secondary_auth_score,
                "audio_speech": self.latest_audio_speech,
                "audio_db_level": self.latest_audio_db,
                "second_cam_connected": self.latest_second_cam_connected,
                "second_cam_phone": self.latest_second_cam_phone,
                "second_cam_person": self.latest_second_cam_person
            },
            "history": self.telemetry_history[-150:]
        }
        
        try:
            with open(log_path, "w") as f:
                json.dump(log_data, f, indent=4)
            print(f"[DecisionEngine2Cam] Telemetry log saved: {log_path}")
        except Exception as e:
            print(f"[DecisionEngine2Cam] Failed to write telemetry log: {e}")

        if self.infraction_callback:
            try:
                self.infraction_callback(timestamp_str)
            except Exception as e:
                print(f"[DecisionEngine2Cam] Infraction callback failed: {e}")

        # Wait briefly for file writes to finish
        time.sleep(0.5)

        # Merge video and audio into a single file with audio
        if os.path.exists(video_path) and os.path.exists(audio_path):
            import subprocess
            import imageio_ffmpeg
            try:
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                cmd = [
                    ffmpeg_exe,
                    "-y",
                    "-i", video_path,
                    "-i", audio_path,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    final_video_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # If merging was successful, remove separate files
                if os.path.exists(final_video_path) and os.path.getsize(final_video_path) > 0:
                    os.remove(video_path)
                    os.remove(audio_path)
                    print(f"[DecisionEngine2Cam] Merged infraction video & audio into single file: {final_video_path}")
            except Exception as e:
                print(f"[DecisionEngine2Cam] Error merging infraction audio and video: {e}")

        # Termination is disabled, allowing the candidate to continue the exam.
        # if self.termination_callback:
        #     print("[DecisionEngine2Cam] Invoking session termination callback...")
        #     self.termination_callback()
