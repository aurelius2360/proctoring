import os
import cv2
import numpy as np
import threading
import queue
import time
import urllib.request
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
from eyetrax import GazeEstimator

class CustomGazeEstimator(GazeEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_face_landmarks = None

    def extract_features(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = np.ascontiguousarray(image_rgb)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=image_rgb,
        )
        ts_ms = int(time.time() * 1000)
        if ts_ms <= self._mp_last_ts_ms:
            ts_ms = self._mp_last_ts_ms + 1
        self._mp_last_ts_ms = ts_ms

        result = self._face_landmarker.detect_for_video(mp_image, ts_ms)
        if not result.face_landmarks:
            self.last_face_landmarks = None
            return None, False

        landmarks = result.face_landmarks[0]
        self.last_face_landmarks = landmarks

        all_points = np.array(
            [(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32
        )
        left_corner = all_points[33]
        right_corner = all_points[263]
        top_of_head = all_points[10]

        eye_center = (left_corner + right_corner) / 2.0
        shifted_points = all_points - eye_center
        x_axis = right_corner - left_corner
        x_axis /= np.linalg.norm(x_axis) + 1e-9
        y_approx = top_of_head - eye_center
        y_approx /= np.linalg.norm(y_approx) + 1e-9
        y_axis = y_approx - np.dot(y_approx, x_axis) * x_axis
        y_axis /= np.linalg.norm(y_axis) + 1e-9
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-9
        R = np.column_stack((x_axis, y_axis, z_axis))
        rotated_points = (R.T @ shifted_points.T).T

        left_corner_rot = R.T @ (left_corner - eye_center)
        right_corner_rot = R.T @ (right_corner - eye_center)
        inter_eye_dist = np.linalg.norm(right_corner_rot - left_corner_rot)
        if inter_eye_dist > 1e-7:
            rotated_points /= inter_eye_dist

        from eyetrax.constants import LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MUTUAL_INDICES
        subset_indices = LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MUTUAL_INDICES
        eye_landmarks = rotated_points[subset_indices]
        features = eye_landmarks.flatten()

        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        roll = np.arctan2(R[2, 1], R[2, 2])
        features = np.concatenate([features, [yaw, pitch, roll]])

        # Blink detection
        left_eye_inner = np.array([landmarks[133].x, landmarks[133].y])
        left_eye_outer = np.array([landmarks[33].x, landmarks[33].y])
        left_eye_top = np.array([landmarks[159].x, landmarks[159].y])
        left_eye_bottom = np.array([landmarks[145].x, landmarks[145].y])

        right_eye_inner = np.array([landmarks[362].x, landmarks[362].y])
        right_eye_outer = np.array([landmarks[263].x, landmarks[263].y])
        right_eye_top = np.array([landmarks[386].x, landmarks[386].y])
        right_eye_bottom = np.array([landmarks[374].x, landmarks[374].y])

        left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
        left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
        left_EAR = left_eye_height / (left_eye_width + 1e-9)

        right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)
        right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
        right_EAR = right_eye_height / (right_eye_width + 1e-9)

        EAR = (left_EAR + right_EAR) / 2

        self._ear_history.append(EAR)
        if len(self._ear_history) >= self._min_history:
            thr = float(np.mean(self._ear_history)) * self._blink_ratio
        else:
            thr = 0.2
        blink_detected = EAR < thr

        return features, blink_detected

class GazeWorker(threading.Thread):
    def __init__(self, frame_queue, sequence_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.sequence_queue = sequence_queue
        self.running = False
        self.daemon = True
        
        # Define model task file destination
        self.model_path = os.path.join("models", "face_landmarker.task")
        self._ensure_model_exists()

        # 3D Head model points for solvePnP
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye corner (outer)
            (225.0, 170.0, -135.0),      # Right eye corner (outer)
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float32)

        # 5-point screen calibration targets (4 corners + 1 center)
        self.calibration_targets = [
            (0.5, 0.5),   # Center
            (0.03, 0.03), # Top-Left
            (0.97, 0.03), # Top-Right
            (0.03, 0.97), # Bottom-Left
            (0.97, 0.97)  # Bottom-Right
        ]
        
        # Calibration state machine
        self.calibrated = False
        self.calibration_started = False
        self.current_target_index = 0
        self.target_start_time = None
        self.target_duration = 2.6 # seconds per target point (gives user time to look)
        self.collected_samples = [] # list of [target_x, target_y, pitch, yaw, left_h, left_v, right_h, right_v]
        self.collected_features = [] # list of 1D feature arrays for EyeTrax
        self.collected_targets = []  # list of [target_x, target_y] coordinates for EyeTrax
        self.calib_min_x = 0.03
        self.calib_max_x = 0.97
        self.calib_min_y = 0.03
        self.calib_max_y = 0.97

        # Calibration weights (least squares projection)
        self.W_x = None
        self.W_y = None

        # Feature boundary limits (min/max envelopes)
        self.min_pitch = -10.0
        self.max_pitch = 10.0
        self.min_yaw = -15.0
        self.max_yaw = 15.0
        self.min_left_h = 0.35
        self.max_left_h = 0.65
        self.min_left_v = 0.35
        self.max_left_v = 0.65
        self.min_right_h = 0.35
        self.max_right_h = 0.65
        self.min_right_v = 0.35
        self.max_right_v = 0.65

        # Shared latest data cache
        self.latest_data = {
            "face_detected": False,
            "calibrated": False,
            "calibration_active": True,
            "calibration_target": (0.5, 0.5),
            "calibration_progress_point": 0.0,
            "current_target_index": 0,
            "total_targets": len(self.calibration_targets),
            "head_pose": [0.0, 0.0, 0.0],
            "rvec": None,
            "tvec": None,
            "left_gaze": [0.5, 0.5, 0.0],
            "right_gaze": [0.5, 0.5, 0.0],
            "eye_apertures": [1.0, 1.0],
            "screen_gaze": [0.5, 0.5],
            "landmarks_2d": [],
            "gaze_deviation": 0.0,
            "timestamp": 0.0
        }
        self.data_lock = threading.Lock()

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            print(f"[GazeWorker] Model '{self.model_path}' not found. Downloading...")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                print(f"[GazeWorker] Model downloaded successfully to '{self.model_path}'")
            except Exception as e:
                print(f"[GazeWorker] Error downloading model file: {e}")

    def get_latest_data(self):
        with self.data_lock:
            return self.latest_data.copy()

    def start_calibration(self):
        with self.data_lock:
            self.calibration_started = True
            self.target_start_time = None
            self.current_target_index = 0
            self.collected_samples = []
            self.collected_features = [] # Clear EyeTrax lists
            self.collected_targets = []
            self.calibrated = False
            self.latest_data["calibrated"] = False
            self.latest_data["current_target_index"] = 0
            self.latest_data["calibration_active"] = True
            print("[GazeWorker] Gaze calibration started.")

    def start(self):
        self.running = True
        super().start()

    def stop(self):
        self.running = False

    def run(self):
        # Initialize EyeTrax CustomGazeEstimator
        try:
            self.estimator = CustomGazeEstimator(face_landmarker_model=self.model_path)
            print("[GazeWorker] EyeTrax CustomGazeEstimator initialized successfully using local face landmarker.")
        except Exception as e:
            print(f"[GazeWorker] Failed to initialize CustomGazeEstimator: {e}")
            self.running = False
            return

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            h, w, c = frame.shape
            
            try:
                features, blink = self.estimator.extract_features(frame)
            except Exception as e:
                print(f"[GazeWorker] Inference exception: {e}")
                continue

            if features is not None:
                face_landmarks = self.estimator.last_face_landmarks
                
                # 1. Head Pose (extracted from features array)
                yaw_rad = features[-3]
                pitch_rad = features[-2]
                roll_rad = features[-1]
                
                pitch = float(np.degrees(pitch_rad))
                yaw = float(np.degrees(yaw_rad))
                roll = float(np.degrees(roll_rad))

                # Head Pose solvePnP for rvec/tvec HUD overlays
                indices_solve_pnp = [1, 152, 33, 263, 61, 291]
                image_points = []
                for idx in indices_solve_pnp:
                    lm = face_landmarks[idx]
                    image_points.append((lm.x * w, lm.y * h))
                image_points = np.array(image_points, dtype=np.float32)

                focal_length = w
                center = (w / 2, h / 2)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype=np.float32)
                dist_coeffs = np.zeros((4, 1))

                success, rvec, tvec = cv2.solvePnP(
                    self.model_points, 
                    image_points, 
                    camera_matrix, 
                    dist_coeffs, 
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                rvec_list = rvec.tolist() if success else None
                tvec_list = tvec.tolist() if success else None

                # 2. Eye aperture
                def dist_3d(idx1, idx2):
                    p1 = face_landmarks[idx1]
                    p2 = face_landmarks[idx2]
                    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

                left_aperture = dist_3d(159, 145) / max(dist_3d(33, 133), 1e-6)
                right_aperture = dist_3d(386, 374) / max(dist_3d(362, 263), 1e-6)
                left_open = np.clip(left_aperture / 0.4, 0.0, 1.0)
                right_open = np.clip(right_aperture / 0.4, 0.0, 1.0)

                # 3. Eye gaze horizontal/vertical self-normalized ratios
                left_outer = face_landmarks[33]
                left_inner = face_landmarks[133]
                left_iris = face_landmarks[468] if len(face_landmarks) > 468 else left_outer

                right_outer = face_landmarks[263]
                right_inner = face_landmarks[362]
                right_iris = face_landmarks[473] if len(face_landmarks) > 473 else right_outer

                left_dx = left_inner.x - left_outer.x
                left_ratio_h = (left_iris.x - left_outer.x) / left_dx if abs(left_dx) > 1e-6 else 0.5

                right_dx = right_outer.x - right_inner.x
                right_ratio_h = (right_iris.x - right_inner.x) / right_dx if abs(right_dx) > 1e-6 else 0.5

                left_dy = face_landmarks[145].y - face_landmarks[159].y
                left_ratio_v = (left_iris.y - face_landmarks[159].y) / left_dy if abs(left_dy) > 1e-6 else 0.5

                right_dy = face_landmarks[374].y - face_landmarks[386].y
                right_ratio_v = (right_iris.y - face_landmarks[386].y) / right_dy if abs(right_dy) > 1e-6 else 0.5

                # 4. Multi-point calibration state machine (5 points)
                progress = 0.0
                current_target = (0.5, 0.5)
                
                if not self.calibrated:
                    if not self.calibration_started:
                        current_target = (0.5, 0.5)
                        progress = 0.0
                    else:
                        current_target = self.calibration_targets[self.current_target_index]
                        
                        if self.target_start_time is None:
                            self.target_start_time = time.time()
                        
                        elapsed = time.time() - self.target_start_time
                        progress = min(1.0, elapsed / self.target_duration)
                        
                        # Collect samples for EyeTrax
                        if elapsed > 1.0 and not blink:
                            self.collected_features.append(features)
                            self.collected_targets.append([current_target[0], current_target[1]])
                            self.collected_samples.append([
                                current_target[0], current_target[1],
                                1.0, pitch, yaw, left_ratio_h, left_ratio_v, right_ratio_h, right_ratio_v
                            ])

                        if elapsed >= self.target_duration:
                            self.current_target_index += 1
                            self.target_start_time = None
                            
                            # Once all 5 targets are complete
                            if self.current_target_index >= len(self.calibration_targets):
                                self._fit_calibration()

                # 5. Project gaze using EyeTrax & check boundary envelope violations
                if self.calibrated:
                    # Bounding envelope checks (highly relaxed for free access looking coordinates)
                    pose_buffer = 20.0
                    ratio_buffer = 0.20
                    
                    out_of_pose = (pitch < self.min_pitch - pose_buffer or pitch > self.max_pitch + pose_buffer or
                                   yaw < self.min_yaw - pose_buffer or yaw > self.max_yaw + pose_buffer)
                    
                    out_of_eye = (left_ratio_h < self.min_left_h - ratio_buffer or left_ratio_h > self.max_left_h + ratio_buffer or
                                  left_ratio_v < self.min_left_v - ratio_buffer or left_ratio_v > self.max_left_v + ratio_buffer or
                                  right_ratio_h < self.min_right_h - ratio_buffer or right_ratio_h > self.max_right_h + ratio_buffer or
                                  right_ratio_v < self.min_right_v - ratio_buffer or right_ratio_v > self.max_right_v + ratio_buffer)

                    # Estimate gaze position on screen using EyeTrax
                    pred = self.estimator.predict([features])[0]
                    pred_x = float(pred[0])
                    pred_y = float(pred[1])
                    
                    gaze_screen_x = np.clip(pred_x, 0.0, 1.0)
                    gaze_screen_y = np.clip(pred_y, 0.0, 1.0)
                    
                    out_of_regression = (pred_x < -1.25 or pred_x > 2.25 or pred_y < -1.25 or pred_y > 2.25)
                    
                    # Bounding box of calibration targets with a buffer
                    gaze_buffer = 0.25
                    inside_calibrated_area = (
                        (self.calib_min_x - gaze_buffer <= pred_x <= self.calib_max_x + gaze_buffer) and
                        (self.calib_min_y - gaze_buffer <= pred_y <= self.calib_max_y + gaze_buffer)
                    )
                    
                    if inside_calibrated_area:
                        # User has free access of looking in these coordinates!
                        is_outside = False
                        gaze_deviation = 0.0
                    else:
                        # Flag as looking away if regression says so OR if features exceed calibration envelope
                        is_outside = out_of_pose or out_of_eye or out_of_regression
                        gaze_deviation = 1.0 if is_outside else 0.0
                else:
                    gaze_screen_x = 0.5
                    gaze_screen_y = 0.5
                    gaze_deviation = 0.0

                # Collect landmarks for HUD
                landmarks_2d = []
                for lm in face_landmarks:
                    landmarks_2d.append((int(lm.x * w), int(lm.y * h)))

                gaze_data = {
                    "face_detected": True,
                    "calibrated": self.calibrated,
                    "calibration_active": not self.calibrated,
                    "calibration_target": current_target,
                    "calibration_progress_point": progress,
                    "current_target_index": self.current_target_index,
                    "total_targets": len(self.calibration_targets),
                    "head_pose": [float(pitch), float(yaw), float(roll)],
                    "rvec": rvec_list,
                    "tvec": tvec_list,
                    "left_gaze": [float(left_ratio_h), float(left_ratio_v), 0.0],
                    "right_gaze": [float(right_ratio_h), float(right_ratio_v), 0.0],
                    "eye_apertures": [float(left_open), float(right_open)],
                    "screen_gaze": [float(gaze_screen_x), float(gaze_screen_y)],
                    "landmarks_2d": landmarks_2d,
                    "gaze_deviation": gaze_deviation,
                    "timestamp": time.time()
                }

                with self.data_lock:
                    self.latest_data = gaze_data

                avg_eye_h = float((left_ratio_h + right_ratio_h) / 2.0)
                avg_eye_v = float((left_ratio_v + right_ratio_v) / 2.0)
                
                event = {
                    "type": "gaze",
                    "face_detected": True,
                    "features": [
                        float(pitch), float(yaw), float(roll),
                        float(left_ratio_h), float(left_ratio_v), 0.0,
                        float(left_open), float(right_open),
                        float(gaze_screen_x), float(gaze_screen_y),
                        1.0, 0.0,
                        1.0,
                        float(gaze_deviation),
                        avg_eye_h,
                        avg_eye_v
                    ],
                    "timestamp": time.time()
                }
                
                try:
                    self.sequence_queue.put_nowait(event)
                except queue.Full:
                    pass

            else:
                # No face detected
                current_target_val = (0.5, 0.5)
                progress_val = 0.0
                if not self.calibrated and self.calibration_started and self.current_target_index < len(self.calibration_targets):
                    current_target_val = self.calibration_targets[self.current_target_index]
                    if self.target_start_time is not None:
                        elapsed = time.time() - self.target_start_time
                        progress_val = min(1.0, elapsed / self.target_duration)

                gaze_data = {
                    "face_detected": False,
                    "calibrated": self.calibrated,
                    "calibration_active": not self.calibrated,
                    "calibration_target": current_target_val,
                    "calibration_progress_point": progress_val,
                    "current_target_index": self.current_target_index,
                    "total_targets": len(self.calibration_targets),
                    "head_pose": [0.0, 0.0, 0.0],
                    "left_gaze": [0.5, 0.5, 0.0],
                    "right_gaze": [0.5, 0.5, 0.0],
                    "eye_apertures": [0.0, 0.0],
                    "screen_gaze": [0.5, 0.5],
                    "landmarks_2d": [],
                    "gaze_deviation": 2.0,
                    "timestamp": time.time()
                }

                with self.data_lock:
                    self.latest_data = gaze_data

                event = {
                    "type": "gaze",
                    "face_detected": False,
                    "features": [
                        0.0, 0.0, 0.0,
                        0.5, 0.5, 0.0,
                        0.0, 0.0,
                        0.5, 0.5,
                        1.0, 0.0,
                        0.0,
                        2.0,
                        0.5,
                        0.5
                    ],
                    "timestamp": time.time()
                }

                try:
                    self.sequence_queue.put_nowait(event)
                except queue.Full:
                    pass

            time.sleep(0.01)

        # Cleanup CustomGazeEstimator
        if hasattr(self, 'estimator') and self.estimator is not None:
            self.estimator.close()

    def _fit_calibration(self):
        if len(self.collected_features) < 5:
            # Fallback default calibration weights
            self.W_x = np.array([0.5, 0.0, -0.015, 2.0, 0.0, 2.0, 0.0], dtype=np.float32)
            self.W_y = np.array([0.5, -0.02, 0.0, 0.0, -2.0, 0.0, -2.0], dtype=np.float32)
            self.calibrated = True
            print("[GazeWorker] Calibration features deficient. Standard regression weights loaded.")
            return

        try:
            # Train the EyeTrax GazeEstimator Ridge model
            X = np.array(self.collected_features)
            y = np.array(self.collected_targets)
            self.estimator.train(X, y)
            self.calibrated = True
            
            # Store the bounding box of the calibration targets
            self.calib_min_x = float(np.min(y[:, 0]))
            self.calib_max_x = float(np.max(y[:, 0]))
            self.calib_min_y = float(np.min(y[:, 1]))
            self.calib_max_y = float(np.max(y[:, 1]))
            
            print("[GazeWorker] EyeTrax GazeEstimator calibration model trained successfully.")
        except Exception as e:
            print(f"[GazeWorker] EyeTrax calibration solving error: {e}. Fallback to standard weights.")
            self.W_x = np.array([0.5, 0.0, -0.015, 2.0, 0.0, 2.0, 0.0], dtype=np.float32)
            self.W_y = np.array([0.5, -0.02, 0.0, 0.0, -2.0, 0.0, -2.0], dtype=np.float32)
            self.calibrated = True

        # Extract feature boundary envelope limits from calibration data
        if len(self.collected_samples) >= 5:
            samples = np.array(self.collected_samples)
            self.min_pitch = min(float(np.min(samples[:, 3])), -10.0)
            self.max_pitch = max(float(np.max(samples[:, 3])), 10.0)
            self.min_yaw = min(float(np.min(samples[:, 4])), -15.0)
            self.max_yaw = max(float(np.max(samples[:, 4])), 15.0)
            
            self.min_left_h = min(float(np.min(samples[:, 5])), 0.40)
            self.max_left_h = max(float(np.max(samples[:, 5])), 0.60)
            self.min_left_v = min(float(np.min(samples[:, 6])), 0.40)
            self.max_left_v = max(float(np.max(samples[:, 6])), 0.60)
            
            self.min_right_h = min(float(np.min(samples[:, 7])), 0.40)
            self.max_right_h = max(float(np.max(samples[:, 7])), 0.60)
            self.min_right_v = min(float(np.min(samples[:, 8])), 0.40)
            self.max_right_v = max(float(np.max(samples[:, 8])), 0.60)

        print(f"[GazeWorker] Feature envelope boundaries loaded:")
        print(f"  - Pitch range: [{self.min_pitch:.2f}, {self.max_pitch:.2f}]")
        print(f"  - Yaw range:   [{self.min_yaw:.2f}, {self.max_yaw:.2f}]")
        print(f"  - Eye H range: [{self.min_left_h:.3f}, {self.max_left_h:.3f}]")
