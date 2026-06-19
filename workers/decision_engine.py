import os
import cv2
import json
import threading
import queue
import time
from collections import deque

class DecisionEngine(threading.Thread):
    def __init__(self, decision_queue, termination_callback=None, infraction_callback=None):
        super().__init__()
        self.decision_queue = decision_queue
        self.termination_callback = termination_callback
        self.infraction_callback = infraction_callback
        self.running = False
        self.daemon = True
        
        # State variables
        self.latest_lstm_score = 0.1
        self.latest_auth_verified = True
        self.latest_auth_score = 1.0
        self.latest_audio_speech = False
        self.latest_audio_db = 0.0
        
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
            "timestamp": 0.0
        }
        self.data_lock = threading.Lock()
        self.session_active = False

    def push_frame_to_buffer(self, frame):
        with self.buffer_lock:
            # We copy the frame to avoid reference issues across threads
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
        # Create output directory for infractions
        os.makedirs("infractions", exist_ok=True)

        while self.running:
            try:
                # Read decision event from queue
                event = self.decision_queue.get(timeout=0.1)
            except queue.Empty:
                # Still check breach timing even if queue is empty
                self._evaluate_breach_timing()
                continue

            event_type = event.get("type")

            if event_type == "lstm":
                self.latest_lstm_score = event.get("score", 0.0)
            elif event_type == "auth":
                self.latest_auth_verified = event.get("verified", True)
                self.latest_auth_score = event.get("similarity_score", 1.0)
            elif event_type == "audio":
                self.latest_audio_speech = event.get("speech_detected", False)
                self.latest_audio_db = event.get("db_level", 0.0)

            # Core anomaly scoring algorithm:
            # - Base: LSTM sequence probability
            # - Penalty for speech detection: +0.3
            # - Penalty for failed identity verification: +0.5
            current_anomaly = self.latest_lstm_score
            if self.latest_audio_speech:
                current_anomaly += 0.3
            if not self.latest_auth_verified:
                current_anomaly += 0.5
            current_anomaly = min(1.0, current_anomaly)

            # Exponential Moving Average (EMA) rolling anomaly update
            # Alpha of 0.15 provides a smooth filter over approximately 1.5 - 2 seconds
            self.rolling_anomaly_score = (0.15 * current_anomaly) + (0.85 * self.rolling_anomaly_score)

            # Append telemetry snapshot
            self.telemetry_history.append({
                "timestamp": time.time(),
                "lstm_score": self.latest_lstm_score,
                "auth_verified": self.latest_auth_verified,
                "auth_score": self.latest_auth_score,
                "audio_speech": self.latest_audio_speech,
                "audio_db": self.latest_audio_db,
                "current_anomaly": current_anomaly,
                "rolling_anomaly": self.rolling_anomaly_score
            })
            # Keep history trimmed to last 1000 items
            if len(self.telemetry_history) > 1000:
                self.telemetry_history.pop(0)

            self._evaluate_breach_timing()

            # Yield control
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
                    "timestamp": time.time()
                }
            return

        # Evaluate anomaly threshold breach
        seconds_in_breach = 0.0
        
        if self.rolling_anomaly_score > self.anomaly_threshold:
            if self.breach_start_time is None:
                self.breach_start_time = time.time()
            else:
                seconds_in_breach = time.time() - self.breach_start_time
                
            # If the violation continues beyond threshold duration, trigger action
            if seconds_in_breach >= self.continuous_violation_duration and not self.violation_triggered:
                self.violation_triggered = True
                print(f"\n[DecisionEngine] !!! VIOLATION DETECTED !!! Anomaly score exceeded {self.anomaly_threshold} for > {self.continuous_violation_duration}s.")
                # Run the infraction handling in a separate thread so we don't block the decision engine
                threading.Thread(target=self._handle_infraction, daemon=True).start()
        else:
            self.breach_start_time = None

        # Update cache for orchestrator overlay
        with self.data_lock:
            self.latest_data = {
                "rolling_anomaly": float(self.rolling_anomaly_score),
                "seconds_in_breach": float(seconds_in_breach),
                "violation_triggered": self.violation_triggered,
                "timestamp": time.time()
            }

    def _handle_infraction(self):
        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        
        # 1. Save corresponding video buffer
        with self.buffer_lock:
            frames_to_save = list(self.video_buffer)
            
        if len(frames_to_save) > 0:
            h, w, c = frames_to_save[0].shape
            video_path = f"infractions/violation_{timestamp_str}.mp4"
            # 30 FPS, mp4v encoding
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
            
            for frame in frames_to_save:
                writer.write(frame)
            writer.release()
            print(f"[DecisionEngine] Infraction video saved: {video_path} ({len(frames_to_save)} frames)")
        else:
            print("[DecisionEngine] No frames in buffer to save.")

        # 2. Save detailed telemetry logs
        log_path = f"infractions/telemetry_{timestamp_str}.json"
        log_data = {
            "trigger_timestamp": time.time(),
            "trigger_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "final_rolling_anomaly": self.rolling_anomaly_score,
            "last_known_metrics": {
                "lstm_sequence_score": self.latest_lstm_score,
                "auth_verified": self.latest_auth_verified,
                "auth_score": self.latest_auth_score,
                "audio_speech": self.latest_audio_speech,
                "audio_db_level": self.latest_audio_db
            },
            "history": self.telemetry_history[-150:] # save corresponding window history
        }
        
        try:
            with open(log_path, "w") as f:
                json.dump(log_data, f, indent=4)
            print(f"[DecisionEngine] Telemetry log saved: {log_path}")
        except Exception as e:
            print(f"[DecisionEngine] Failed to write telemetry log: {e}")

        # 3. Save audio infraction via callback
        if self.infraction_callback:
            try:
                self.infraction_callback(timestamp_str)
            except Exception as e:
                print(f"[DecisionEngine] Infraction callback failed: {e}")

        # 4. Trigger Session Termination
        if self.termination_callback:
            print("[DecisionEngine] Invoking session termination callback...")
            self.termination_callback()
