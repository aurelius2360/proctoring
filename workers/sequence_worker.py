import torch
import torch.nn as nn
import numpy as np
import threading
import queue
import time
from collections import deque

class BehaviorLSTM(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self._initialize_heuristics()

    def _initialize_heuristics(self):
        with torch.no_grad():
            nn.init.constant_(self.fc.bias, -1.5)
            nn.init.normal_(self.fc.weight, mean=0.2, std=0.1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        prob = self.sigmoid(self.fc(last_out))
        return prob

class SequenceWorker(threading.Thread):
    def __init__(self, sequence_queue, decision_queue):
        super().__init__()
        self.sequence_queue = sequence_queue
        self.decision_queue = decision_queue
        self.running = False
        self.daemon = True
        
        # 16-D feature vector rolling window (30 frames)
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
        
        self.latest_data = {
            "anomaly_probability": 0.0,
            "features_snapshot": {},
            "look_away_duration": 0.0,
            "timestamp": 0.0
        }
        self.data_lock = threading.Lock()

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
        # Initialize PyTorch LSTM model
        model = BehaviorLSTM(input_dim=16, hidden_dim=32, num_layers=2)
        model.eval()

        while self.running:
            try:
                event = self.sequence_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            event_type = event.get("type")
            features = event.get("features", [])

            if event_type == "gaze":
                # Ingest gaze features (16 values)
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
                # Index 10, 11 are person_count, phone_present (skipped here, updated by object event)
                self.current_state["face_detected"] = features[12]
                self.current_state["gaze_deviation"] = features[13]
                self.current_state["avg_eye_ratio_h"] = features[14]
                self.current_state["avg_eye_ratio_v"] = features[15]

                # 10-second continuous look-away tracking:
                # Look-away is defined as face not detected OR gaze deviation exceeding 0.8
                is_looking_away = (self.current_state["face_detected"] == 0.0) or (self.current_state["gaze_deviation"] > 0.8)
                
                if is_looking_away and self.session_active:
                    if self.look_away_start_time is None:
                        self.look_away_start_time = time.time()
                    self.look_away_duration = time.time() - self.look_away_start_time
                else:
                    self.look_away_start_time = None
                    self.look_away_duration = 0.0

            elif event_type == "object":
                # Ingest object features
                self.current_state["person_count"] = features[0]
                self.current_state["phone_present"] = features[1]
                self.current_state["device_present"] = features[2]

            # Reconstruct 16-D feature vector with updated object metrics
            feat_vector = [
                self.current_state["pitch"],
                self.current_state["yaw"],
                self.current_state["roll"],
                self.current_state["left_ratio_h"],
                self.current_state["left_ratio_v"],
                self.current_state["right_ratio_h"],
                self.current_state["left_open"],
                self.current_state["right_open"],
                self.current_state["screen_gaze_x"],
                self.current_state["screen_gaze_y"],
                self.current_state["person_count"],
                # Blend phone and device presence into a single slot for 16-D feature compatibility
                float(max(self.current_state["phone_present"], self.current_state["device_present"])),
                self.current_state["face_detected"],
                self.current_state["gaze_deviation"],
                self.current_state["avg_eye_ratio_h"],
                self.current_state["avg_eye_ratio_v"]
            ]

            # Sequence evaluation triggered by gaze events
            if event_type == "gaze":
                self.history.append(feat_vector)

                if len(self.history) == self.window_size:
                    seq_array = np.array(self.history, dtype=np.float32)
                    seq_tensor = torch.tensor(seq_array).unsqueeze(0)

                    with torch.no_grad():
                        prob_tensor = model(seq_tensor)
                        anomaly_prob = float(prob_tensor.item())

                    # Calibration metrics
                    calibration = 0.0
                    
                    # 1. Phone or electronic device presence (immediate critical flag)
                    if self.current_state["phone_present"] > 0 or self.current_state["device_present"] > 0:
                        calibration += 0.80
                    
                    # 2. Person count discrepancies (multiple people or no one)
                    if self.current_state["person_count"] != 1.0:
                        calibration += 0.50

                    # Blend LSTM with rules
                    final_anomaly_prob = np.clip(0.5 * anomaly_prob + 0.5 * calibration, 0.0, 1.0)

                    # Update thread-safe stats
                    with self.data_lock:
                        self.latest_data = {
                            "anomaly_probability": float(final_anomaly_prob),
                            "features_snapshot": self.current_state.copy(),
                            "look_away_duration": float(self.look_away_duration),
                            "timestamp": time.time()
                        }

                    # Push anomaly score to Decision Engine
                    event_to_decision = {
                        "type": "lstm",
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
