import cv2
import threading
import queue
import time
from ultralytics import YOLO

class ObjectWorker(threading.Thread):
    def __init__(self, frame_queue, sequence_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.sequence_queue = sequence_queue
        self.running = False
        self.daemon = True
        
        self.latest_data = {
            "boxes": [],  # list of {"box": [x1, y1, x2, y2], "class": "person"/"phone"/"device", "conf": 0.8}
            "person_count": 0,
            "phone_present": False,
            "device_present": False,
            "timestamp": 0.0
        }
        self.data_lock = threading.Lock()

    def get_latest_data(self):
        with self.data_lock:
            return self.latest_data.copy()

    def start(self):
        self.running = True
        super().start()

    def stop(self):
        self.running = False

    def run(self):
        # Load the lightweight YOLOv8 nano model
        # YOLO downloaded automatically upon first instantiation
        model = YOLO("yolov8n.pt")

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Perform inference filtering classes: 0 (person), 62 (tv), 63 (laptop), 64 (mouse), 65 (remote), 66 (keyboard), 67 (cell phone)
            results = model(frame, classes=[0, 62, 63, 64, 65, 66, 67], verbose=False)
            
            person_count = 0
            phone_present = False
            device_present = False
            boxes_data = []

            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    if cls_id == 0:
                        cls_name = "person"
                        person_count += 1
                    elif cls_id == 67:
                        cls_name = "phone"
                        phone_present = True
                    elif cls_id in [62, 63, 64, 65, 66]:
                        cls_name = "device"
                        device_present = True
                    else:
                        continue
                        
                    boxes_data.append({
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "class": cls_name,
                        "conf": conf
                    })

            # Update thread-safe latest data cache
            with self.data_lock:
                self.latest_data = {
                    "boxes": boxes_data,
                    "person_count": person_count,
                    "phone_present": phone_present,
                    "device_present": device_present,
                    "timestamp": time.time()
                }

            # Send event features to sequence queue:
            # Format: [person_count, float(phone_present), float(device_present)]
            event = {
                "type": "object",
                "features": [float(person_count), float(phone_present), float(device_present)],
                "timestamp": time.time()
            }
            
            try:
                self.sequence_queue.put_nowait(event)
            except queue.Full:
                pass

            # Relinquish CPU slice
            time.sleep(0.01)
