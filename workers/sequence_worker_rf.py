import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import threading
import queue
import time
from collections import deque

class SequenceWorkerRF(threading.Thread):
    def __init__(self, sequence_queue, decision_queue):
        super().__init__()
        self.sequence_queue = sequence_queue
        self.decision_queue = decision_queue
        self.running = False
        self.daemon = True
        
        # Feature Buffer Node: 30-frame rolling window for pitch, yaw, gaze_x, gaze_y
        self.window_size = 30
        self.history = deque(maxlen=self.window_size)
        
        # Latest features state
        self.current_state = {
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "left_ratio_h": 0.5,
            "left_ratio_v": 0.5,
            "right_ratio_h": 0.5,
            "left_open": 1.0,
            "right_open": 1.0,
            "screen_gaze_x": 0.5,
            "screen_gaze_y": 0.5,
            "person_count": 1.0,
            "phone_present": 0.0,
            "device_present": 0.0,
            "face_detected": 1.0,
            "gaze_deviation": 0.0,
            "avg_eye_ratio_h": 0.5,
            "avg_eye_ratio_v": 0.5
        }

        # Timer for continuous look-away tracking
        self.look_away_start_time = None
        self.look_away_duration = 0.0
        self.session_active = False
        
        # Prediction interval timer (1-second)
        self.last_predict_time = 0.0
        
        self.latest_data = {
            "anomaly_probability": 0.0,
            "features_snapshot": {},
            "look_away_duration": 0.0,
            "timestamp": 0.0
        }
        self.data_lock = threading.Lock()

        # Initialize and train RandomForest behavioral classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self._train_synthetic_model()

    def _train_synthetic_model(self):
        # Build a synthetic dataset representing standard normal and abnormal metrics
        # Features: [pitch_var, yaw_var, gaze_x_var, gaze_y_var, pitch_mean, yaw_mean, gaze_x_mean, gaze_y_mean, look_away_duration]
        
        # 1. Normal behavior samples (Class 0)
        # Allows normal variations in gaze coordinates and head pose since user looks across the entire screen
        X_normal = []
        for _ in range(150):
            pitch_var = np.random.uniform(0.05, 3.5)
            yaw_var = np.random.uniform(0.05, 3.5)
            gaze_x_var = np.random.uniform(0.0001, 0.02)
            gaze_y_var = np.random.uniform(0.0001, 0.02)
            
            # Normal comfortable head pose limits when looking at screen
            pitch_mean = np.random.uniform(-15.0, 15.0)
            yaw_mean = np.random.uniform(-20.0, 20.0)
            # Normal eye gaze covers the entire screen from left to right, top to bottom
            gaze_x_mean = np.random.uniform(0.05, 0.95)
            gaze_y_mean = np.random.uniform(0.05, 0.95)
            
            duration = 0.0
            
            X_normal.append([
                pitch_var, yaw_var, gaze_x_var, gaze_y_var,
                pitch_mean, yaw_mean, gaze_x_mean, gaze_y_mean,
                duration
            ])

        # 2. Anomalous behavior samples (Class 1)
        # High variance (cheating signs/restlessness) OR extreme means (looking away) OR high duration
        X_anomalous = []
        for _ in range(150):
            case = np.random.choice(["variance", "means", "duration"])
            
            if case == "variance":
                pitch_var = np.random.uniform(15.0, 45.0)
                yaw_var = np.random.uniform(20.0, 55.0)
                gaze_x_var = np.random.uniform(0.08, 0.25)
                gaze_y_var = np.random.uniform(0.08, 0.25)
                pitch_mean = np.random.uniform(-15.0, 15.0)
                yaw_mean = np.random.uniform(-20.0, 20.0)
                gaze_x_mean = np.random.uniform(0.05, 0.95)
                gaze_y_mean = np.random.uniform(0.05, 0.95)
                duration = np.random.uniform(0.0, 1.0)
            elif case == "means":
                pitch_var = np.random.uniform(0.05, 3.5)
                yaw_var = np.random.uniform(0.05, 3.5)
                gaze_x_var = np.random.uniform(0.0001, 0.02)
                gaze_y_var = np.random.uniform(0.0001, 0.02)
                # Looking off-screen: extreme head pose or gaze coordinate
                pitch_mean = np.random.choice([np.random.uniform(-35.0, -22.0), np.random.uniform(22.0, 35.0)])
                yaw_mean = np.random.choice([np.random.uniform(-45.0, -28.0), np.random.uniform(28.0, 45.0)])
                gaze_x_mean = np.random.choice([np.random.uniform(-0.35, -0.15), np.random.uniform(1.15, 1.35)])
                gaze_y_mean = np.random.choice([np.random.uniform(-0.35, -0.15), np.random.uniform(1.15, 1.35)])
                duration = np.random.uniform(0.0, 1.5)
            else: # duration
                pitch_var = np.random.uniform(0.05, 3.5)
                yaw_var = np.random.uniform(0.05, 3.5)
                gaze_x_var = np.random.uniform(0.0001, 0.02)
                gaze_y_var = np.random.uniform(0.0001, 0.02)
                pitch_mean = np.random.uniform(-15.0, 15.0)
                yaw_mean = np.random.uniform(-20.0, 20.0)
                gaze_x_mean = np.random.uniform(0.05, 0.95)
                gaze_y_mean = np.random.uniform(0.05, 0.95)
                duration = np.random.uniform(3.5, 10.0) # Look away duration exceeds threshold

            X_anomalous.append([
                pitch_var, yaw_var, gaze_x_var, gaze_y_var,
                pitch_mean, yaw_mean, gaze_x_mean, gaze_y_mean,
                duration
            ])

        X = np.vstack([X_normal, X_anomalous])
        y = np.hstack([np.zeros(150), np.ones(150)])
        
        self.model.fit(X, y)
        print("[SequenceWorkerRF] Scikit-Learn RandomForestClassifier pre-fitted successfully on realistic behavioral bounds.")

    def get_latest_data(self):
        with self.data_lock:
            return self.latest_data.copy()

    def set_session_active(self, active):
        with self.data_lock:
            self.session_active = active
            if not active:
                self.look_away_start_time = None
                self.look_away_duration = 0.0
                self.latest_data["look_away_duration"] = 0.0
                self.latest_data["anomaly_probability"] = 0.0

    def start(self):
        self.running = True
        super().start()

    def stop(self):
        self.running = False

    def run(self):
        self.last_predict_time = time.time()

        while self.running:
            try:
                event = self.sequence_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            event_type = event.get("type")
            features = event.get("features", [])

            if event_type == "gaze":
                # Ingest gaze features
                self.current_state["pitch"] = features[0]
                self.current_state["yaw"] = features[1]
                self.current_state["roll"] = features[2]
                self.current_state["left_ratio_h"] = features[3]
                self.current_state["left_ratio_v"] = features[4]
                self.current_state["right_ratio_h"] = features[5]
                self.current_state["left_open"] = features[6]
                self.current_state["right_open"] = features[7]
                self.current_state["screen_gaze_x"] = features[8]
                self.current_state["screen_gaze_y"] = features[9]
                self.current_state["face_detected"] = features[12]
                self.current_state["gaze_deviation"] = features[13]
                self.current_state["avg_eye_ratio_h"] = features[14]
                self.current_state["avg_eye_ratio_v"] = features[15]

                # 3-second continuous look-away tracking:
                is_looking_away = (self.current_state["face_detected"] == 0.0) or (self.current_state["gaze_deviation"] > 0.8)
                
                if is_looking_away and self.session_active:
                    if self.look_away_start_time is None:
                        self.look_away_start_time = time.time()
                    self.look_away_duration = time.time() - self.look_away_start_time
                else:
                    self.look_away_start_time = None
                    self.look_away_duration = 0.0

                # Feature Buffer Node: Append to rolling history
                self.history.append([
                    self.current_state["pitch"],
                    self.current_state["yaw"],
                    self.current_state["screen_gaze_x"],
                    self.current_state["screen_gaze_y"]
                ])

            elif event_type == "object":
                # Ingest object features
                self.current_state["person_count"] = features[0]
                self.current_state["phone_present"] = features[1]
                self.current_state["device_present"] = features[2]

            # Model prediction: check if 1 second has elapsed and history is ready
            current_time = time.time()
            if current_time - self.last_predict_time >= 1.0:
                self.last_predict_time = current_time

                if len(self.history) == self.window_size:
                    # 1. Extract rolling statistics from Feature Buffer Node
                    history_arr = np.array(self.history)
                    means = np.mean(history_arr, axis=0) # [pitch_mean, yaw_mean, gaze_x_mean, gaze_y_mean]
                    vars = np.var(history_arr, axis=0)   # [pitch_var, yaw_var, gaze_x_var, gaze_y_var]

                    # 2. Construct 9-D input feature vector
                    rf_features = [
                        float(vars[0]), float(vars[1]), float(vars[2]), float(vars[3]),
                        float(means[0]), float(means[1]), float(means[2]), float(means[3]),
                        float(self.look_away_duration)
                    ]

                    # 3. Inference: predict anomaly probability via RandomForest predict_proba
                    try:
                        probs = self.model.predict_proba([rf_features])[0]
                        anomaly_prob = float(probs[1]) # Class 1 represents anomalous behavior
                    except Exception as e:
                        print(f"[SequenceWorkerRF] Inference error: {e}")
                        anomaly_prob = 0.0

                    # 4. Integrate object rules overrides
                    rule_anomaly = 0.0
                    # Phone or device presence immediately triggers critical flag
                    if self.current_state["phone_present"] > 0 or self.current_state["device_present"] > 0:
                        rule_anomaly += 0.80
                    # Person count invalid (multiple people or no person on screen)
                    if self.current_state["person_count"] != 1.0:
                        rule_anomaly += 0.50

                    # Blend model probability with rule checks
                    final_anomaly_prob = np.clip(0.5 * anomaly_prob + 0.5 * rule_anomaly, 0.0, 1.0)

                    # Update latest data cache
                    with self.data_lock:
                        self.latest_data = {
                            "anomaly_probability": float(final_anomaly_prob),
                            "features_snapshot": self.current_state.copy(),
                            "look_away_duration": float(self.look_away_duration),
                            "timestamp": time.time()
                        }

                    # Push anomaly score to Decision Engine queue
                    event_to_decision = {
                        "type": "lstm", # Keep type as lstm to maintain compatibility with DecisionEngine queue parser
                        "score": float(final_anomaly_prob),
                        "look_away_duration": float(self.look_away_duration),
                        "timestamp": time.time()
                    }
                    
                    try:
                        self.decision_queue.put_nowait(event_to_decision)
                    except queue.Full:
                        pass

            # Relinquish CPU slice
            time.sleep(0.01)
