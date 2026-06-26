import cv2
import threading
import queue
import time
from deepface import DeepFace

class AuthWorker(threading.Thread):
    def __init__(self, frame_queue, decision_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.decision_queue = decision_queue
        self.running = False
        self.daemon = True
        
        self.baseline_frame = None
        self.baseline_lock = threading.Lock()
        
        self.latest_data = {
            "verified": True,
            "similarity_score": 1.0,
            "baseline_set": False,
            "timestamp": 0.0
        }
        self.data_lock = threading.Lock()

    def set_baseline_frame(self, frame):
        with self.baseline_lock:
            # DeepFace works best with RGB or standard numpy arrays
            self.baseline_frame = frame.copy()
        with self.data_lock:
            self.latest_data["baseline_set"] = True
            self.latest_data["timestamp"] = time.time()

    def get_latest_data(self):
        with self.data_lock:
            return self.latest_data.copy()

    def start(self):
        self.running = True
        super().start()

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            try:
                # Get frame with timeout
                item = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if isinstance(item, dict):
                source = item.get("source", "primary")
                frame = item.get("frame")
            else:
                source = "primary"
                frame = item

            if frame is None:
                continue

            # Check if we have a baseline frame. If not, set this as baseline.
            with self.baseline_lock:
                if self.baseline_frame is None:
                    self.baseline_frame = frame.copy()
                    print("[AuthWorker] Baseline frame captured and set.")
                    with self.data_lock:
                        self.latest_data["baseline_set"] = True
                    continue
                else:
                    baseline = self.baseline_frame.copy()

            # Perform verification
            try:
                # DeepFace.verify takes BGR or RGB numpy arrays directly
                # We use cosine distance, which typically has a threshold of 0.40 for VGG-Face
                result = DeepFace.verify(
                    img1_path=baseline,
                    img2_path=frame,
                    model_name="VGG-Face",
                    distance_metric="cosine",
                    enforce_detection=False
                )
                
                # In DeepFace cosine distance:
                # 0.0 means identical, 1.0 or higher means completely different.
                # similarity_score can be defined as 1.0 - cosine_distance.
                cosine_dist = float(result.get("distance", 1.0))
                verified = bool(result.get("verified", False))
                similarity_score = max(0.0, 1.0 - cosine_dist)

            except Exception as e:
                print(f"[AuthWorker] DeepFace exception for {source} camera: {e}")
                verified = False
                similarity_score = 0.0

            # Update cache
            with self.data_lock:
                if source == "primary":
                    self.latest_data["verified"] = verified
                    self.latest_data["similarity_score"] = similarity_score
                
                self.latest_data["baseline_set"] = True
                self.latest_data["timestamp"] = time.time()
                self.latest_data[source] = {
                    "verified": verified,
                    "similarity_score": similarity_score
                }

            # Send event to Decision Engine:
            # Type: "auth"
            # verified: boolean flag
            # similarity: score
            event = {
                "type": "auth",
                "source": source,
                "verified": verified,
                "similarity_score": similarity_score,
                "timestamp": time.time()
            }
            
            try:
                self.decision_queue.put_nowait(event)
            except queue.Full:
                pass

            # Relinquish CPU briefly
            time.sleep(0.01)
