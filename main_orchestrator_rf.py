import cv2
import numpy as np
import queue
import time
import threading
import os

from workers.gaze_worker import GazeWorker
from workers.object_worker import ObjectWorker
from workers.auth_worker import AuthWorker
from workers.audio_worker import AudioWorker
from workers.sequence_worker_rf import SequenceWorkerRF
from workers.decision_engine import DecisionEngine

class MainOrchestrator:
    def __init__(self, camera_index=0, video_path=None):
        self.camera_index = camera_index
        self.video_path = video_path
        self.running = False
        self.termination_requested = False
        self.termination_lock = threading.Lock()
        
        # UI/Calibration States
        self.session_state = "START_SCREEN" # "START_SCREEN", "CALIBRATION", "PROCTORING"
        self.start_btn_coords = (0, 0, 0, 0)
        self.whole_test_writer = None
        
        # Thread-safe Queues
        self.gaze_queue = queue.Queue(maxsize=1)
        self.object_queue = queue.Queue(maxsize=1)
        self.auth_queue = queue.Queue(maxsize=1)
        self.sequence_queue = queue.Queue()
        self.decision_queue = queue.Queue()

        # Initialize Workers
        self.decision_engine = DecisionEngine(
            self.decision_queue, 
            termination_callback=self.trigger_termination,
            infraction_callback=self.handle_infraction_saving
        )
        self.gaze_worker = GazeWorker(self.gaze_queue, self.sequence_queue)
        self.object_worker = ObjectWorker(self.object_queue, self.sequence_queue)
        self.auth_worker = AuthWorker(self.auth_queue, self.decision_queue)
        self.audio_worker = AudioWorker(self.decision_queue)
        self.sequence_worker = SequenceWorkerRF(self.sequence_queue, self.decision_queue)

        # Predefined sampling rates (FPS)
        self.gaze_fps = 12.0
        self.object_fps = 1.0
        self.auth_fps = 0.1 # 10s interval

        # Timers to manage frequencies
        self.last_sent = {
            "gaze": 0.0,
            "object": 0.0,
            "auth": 0.0
        }

    def trigger_termination(self):
        with self.termination_lock:
            self.termination_requested = True

    def handle_infraction_saving(self, timestamp_str):
        # Trigger infraction audio saving inside audio worker
        self.audio_worker.save_infraction_audio(timestamp_str)

    def start_pipeline(self):
        print("[Orchestrator] Starting all model worker threads...")
        self.running = True
        
        # Start all workers
        self.decision_engine.start()
        self.sequence_worker.start()
        self.gaze_worker.start()
        self.object_worker.start()
        self.auth_worker.start()
        self.audio_worker.start()
        
        # Start frames capture loop
        self._run_capture_loop()

    def stop_pipeline(self):
        print("\n[Orchestrator] Shutting down worker threads...")
        self.running = False
        
        # Release whole test recording if any
        if hasattr(self, 'whole_test_writer') and self.whole_test_writer is not None:
            self.whole_test_writer.release()
            self.whole_test_writer = None
            print("[Orchestrator] Saved whole session video recording.")
        
        # Stop all workers
        self.gaze_worker.stop()
        self.object_worker.stop()
        self.auth_worker.stop()
        self.audio_worker.stop()
        self.sequence_worker.stop()
        self.decision_engine.stop()
        
        # Join threads
        self.gaze_worker.join(timeout=1.0)
        self.object_worker.join(timeout=1.0)
        self.auth_worker.join(timeout=1.0)
        self.audio_worker.join(timeout=1.0)
        self.sequence_worker.join(timeout=1.0)
        self.decision_engine.join(timeout=1.0)
        
        cv2.destroyAllWindows()
        print("[Orchestrator] Pipeline shutdown complete.")

    def _run_capture_loop(self):
        # Initialize video source
        if self.video_path is not None and os.path.exists(self.video_path):
            cap = cv2.VideoCapture(self.video_path)
            print(f"[Orchestrator] Video capture stream initialized from: {self.video_path}")
        else:
            cap = cv2.VideoCapture(self.camera_index)
            print(f"[Orchestrator] Webcam capture stream initialized (Camera index: {self.camera_index})")

        if not cap.isOpened():
            print("[Orchestrator] Error: Could not open video source.")
            self.stop_pipeline()
            return

        # Wait a short duration to capture a stable frame for baseline auth
        time.sleep(1.0)
        ret, frame = cap.read()
        if ret:
            self.auth_worker.set_baseline_frame(frame)
            print("[Orchestrator] Baseline frame set successfully.")
            # Start whole session recording immediately upon initialization to capture entire test
            self._start_whole_test_recording(frame)

        feed_window = "Live Feed - Proctored Session"
        dash_window = "Proctoring Telemetry Dashboard"
        det_window = "Detections & Tracking Viewer"
        
        cv2.namedWindow(feed_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(dash_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(det_window, cv2.WINDOW_NORMAL)
        
        # Start in Fullscreen mode for calibration (covers entire monitor screen)
        cv2.setWindowProperty(feed_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow(dash_window, 300, 580)
        cv2.resizeWindow(det_window, 640, 480)

        # Register mouse callback on the webcam feed window
        cv2.setMouseCallback(feed_window, self._on_mouse_click)

        # Frame rate regulation (target 30 FPS capture)
        frame_time = 1.0 / 30.0

        while self.running:
            start_time = time.time()

            # Read frame
            ret, frame = cap.read()
            if not ret:
                # If video file loop back to start, otherwise break
                if self.video_path is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[Orchestrator] Webcam stream disconnected.")
                    break

            # Push frame to decision engine ring buffer
            self.decision_engine.push_frame_to_buffer(frame)

            # Distribute frames to worker queues based on clock frequencies
            current_time = time.time()
            
            # Gaze: 10-15 FPS (target 12 FPS)
            if current_time - self.last_sent["gaze"] >= 1.0 / self.gaze_fps:
                try:
                    # Put latest frame in maxsize=1 queue (non-blocking overwrite)
                    if self.gaze_queue.full():
                        self.gaze_queue.get_nowait()
                    self.gaze_queue.put_nowait(frame)
                    self.last_sent["gaze"] = current_time
                except queue.Full:
                    pass

            # YOLOv8 Object: 1 FPS
            if current_time - self.last_sent["object"] >= 1.0 / self.object_fps:
                try:
                    if self.object_queue.full():
                        self.object_queue.get_nowait()
                    self.object_queue.put_nowait(frame)
                    self.last_sent["object"] = current_time
                except queue.Full:
                    pass

            # DeepFace Auth: 0.1 FPS (once every 10 seconds)
            if current_time - self.last_sent["auth"] >= 1.0 / self.auth_fps:
                try:
                    if self.auth_queue.full():
                        self.auth_queue.get_nowait()
                    self.auth_queue.put_nowait(frame)
                    self.last_sent["auth"] = current_time
                except queue.Full:
                    pass

            # Retrieve telemetry data from all workers for HUD display
            gaze_data = self.gaze_worker.get_latest_data()
            obj_data = self.object_worker.get_latest_data()
            auth_data = self.auth_worker.get_latest_data()
            audio_data = self.audio_worker.get_latest_data()
            dec_data = self.decision_engine.get_latest_data()

            # Handle calibration completion transition
            if self.session_state == "CALIBRATION" and gaze_data.get("calibrated", False):
                print("[Orchestrator] Calibration solved! Transitioning to active proctoring...")
                self.session_state = "PROCTORING"
                # Return feed window back to normal size
                cv2.setWindowProperty(feed_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(feed_window, 640, 480)
                # Activate workers
                self.sequence_worker.set_session_active(True)
                self.decision_engine.set_session_active(True)

            # Record frame to whole session video writer if active
            if hasattr(self, 'whole_test_writer') and self.whole_test_writer is not None:
                self.whole_test_writer.write(frame)

            # Render HUD, Telemetry Dashboard, and Detections frame
            feed_img, dash_img, det_img = self._draw_hud_and_dashboard(frame, gaze_data, obj_data, auth_data, audio_data, dec_data)

            # Display windows
            cv2.imshow(feed_window, feed_img)
            cv2.imshow(dash_window, dash_img)
            cv2.imshow(det_window, det_img)

            # Check termination requests
            with self.termination_lock:
                if self.termination_requested:
                    self._draw_termination_alert(feed_img)
                    cv2.imshow(feed_window, feed_img)
                    cv2.waitKey(2000)
                    break

            # Handle keystrokes (Press 'q' to exit)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Orchestrator] User requested termination via keyboard.")
                break

            # Limit capture loop rate to ~30 FPS
            elapsed = time.time() - start_time
            sleep_duration = max(0.001, frame_time - elapsed)
            time.sleep(sleep_duration)

        cap.release()
        self.stop_pipeline()

    def _on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.session_state == "START_SCREEN":
                btn_x1, btn_y1, btn_x2, btn_y2 = self.start_btn_coords
                if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
                    print("[Orchestrator] Start button clicked! Initializing gaze calibration...")
                    self.session_state = "CALIBRATION"
                    self.gaze_worker.start_calibration()

    def _start_whole_test_recording(self, frame):
        os.makedirs("session_recordings", exist_ok=True)
        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        h, w, c = frame.shape
        video_path = f"session_recordings/whole_test_{timestamp_str}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.whole_test_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
        print(f"[Orchestrator] Started whole session video recording: {video_path}")
        # Notify audio worker to start recording whole session audio
        self.audio_worker.start_whole_test_recording(timestamp_str)

    def _draw_hud_and_dashboard(self, frame, gaze_data, obj_data, auth_data, audio_data, dec_data):
        # 1. Clean feed image
        feed = frame.copy()
        h, w, c = feed.shape

        # 3. Build Detections & Tracking Viewer Frame
        det = frame.copy()
        landmarks = gaze_data.get("landmarks_2d", [])

        # Draw the boundary loop mapping the screen
        # We increase the boundary region to 3% inset (0.03 to 0.97) so it aligns near the screen edges
        tl_x, tl_y = int(0.03 * w), int(0.03 * h)
        br_x, br_y = int(0.97 * w), int(0.97 * h)
        
        calibrated = gaze_data.get("calibrated", False)
        look_away_dur = self.sequence_worker.get_latest_data().get("look_away_duration", 0.0)
        seconds_in_breach = dec_data.get("seconds_in_breach", 0.0)
        
        if self.session_state == "START_SCREEN":
            box_color = (0, 165, 255) # Orange border on start screen
            thickness = 2
        elif self.session_state == "CALIBRATION":
            box_color = (0, 255, 255) # Cyan during calibration
            thickness = 2
        else: # PROCTORING
            if look_away_dur >= 10.0:
                box_color = (0, 0, 255) # Red warning
                thickness = 3
            elif look_away_dur > 0.0:
                box_color = (0, 165, 255) # Orange warning
                thickness = 2
            else:
                box_color = (0, 255, 0) # Green OK
                thickness = 1
                
        cv2.rectangle(feed, (tl_x, tl_y), (br_x, br_y), box_color, thickness)
        cv2.putText(feed, "MONITOR AREA BOUNDS", (tl_x + 10, tl_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA)

        # Draw predicted gaze pointer inside HUD if calibrated
        if calibrated and self.session_state == "PROCTORING":
            pred_x, pred_y = gaze_data.get("screen_gaze", (0.5, 0.5))
            gx = int(pred_x * w)
            gy = int(pred_y * h)
            cv2.circle(feed, (gx, gy), 4, (255, 255, 0), -1)
            cv2.circle(feed, (gx, gy), 6, (255, 255, 0), 1)

        # Draw active calibration target ball
        if self.session_state == "CALIBRATION":
            # Draw a black semi-transparent strip at the top of the screen for instructions
            overlay_calib = feed.copy()
            cv2.rectangle(overlay_calib, (0, 0), (w, 55), (15, 15, 15), -1)
            cv2.addWeighted(overlay_calib, 0.8, feed, 0.2, 0, feed)
            
            # Print instruction text
            cv2.putText(feed, "GAZE CALIBRATION: LOOK AT THE TARGET BALLS", (w // 2 - 220, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(feed, "Keep your eyes on the orange center as the white ring shrinks", (w // 2 - 215, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            # Get target info from gaze_data
            tx, ty = gaze_data.get("calibration_target", (0.5, 0.5))
            point_progress = gaze_data.get("calibration_progress_point", 0.0)
            target_idx = gaze_data.get("current_target_index", 0)
            total_targets = gaze_data.get("total_targets", 5)
            
            px = int(tx * w)
            py = int(ty * h)
            
            cv2.circle(feed, (px, py), 10, (0, 165, 255), -1)  # Solid orange center
            cv2.circle(feed, (px, py), 10, (0, 255, 255), 2)   # Yellow border
            cv2.circle(feed, (px, py), 2, (0, 0, 255), -1)      # Red center dot
            
            max_ring_radius = 35
            min_ring_radius = 12
            ring_radius = int(max_ring_radius - point_progress * (max_ring_radius - min_ring_radius))
            cv2.circle(feed, (px, py), ring_radius, (255, 255, 255), 2)
            cv2.putText(feed, f"{target_idx + 1}/{total_targets}", (px + 15, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            if not gaze_data.get("face_detected", False):
                # Draw a dark warning banner at the bottom
                cv2.putText(feed, "FACE NOT DETECTED - PLEASE LOOK AT THE TARGET BALL", (w // 2 - 210, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw Start Screen Button Overlay
        if self.session_state == "START_SCREEN":
            # Draw a dark semi-transparent overlay
            overlay = feed.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.65, feed, 0.35, 0, feed)

            # Button coords (stored so click handler can access them)
            btn_w, btn_h = 240, 60
            btn_x1 = w // 2 - btn_w // 2
            btn_y1 = h // 2 - btn_h // 2
            btn_x2 = w // 2 + btn_w // 2
            btn_y2 = h // 2 + btn_h // 2
            self.start_btn_coords = (btn_x1, btn_y1, btn_x2, btn_y2)

            # Draw start button
            cv2.rectangle(feed, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 165, 255), -1)
            cv2.rectangle(feed, (btn_x1, btn_y1), (btn_x2, btn_y2), (255, 255, 255), 2)

            text = "START SESSION"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            tx = w // 2 - text_size[0] // 2
            ty = h // 2 + text_size[1] // 2
            cv2.putText(feed, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # Draw instructions
            inst1 = "PROCTORING TRACKING ACTIVE"
            inst1_size = cv2.getTextSize(inst1, font, 0.75, 2)[0]
            cv2.putText(feed, inst1, (w // 2 - inst1_size[0] // 2, btn_y1 - 60), font, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

            inst2 = "Position yourself in front of the camera, then click Start."
            inst2_size = cv2.getTextSize(inst2, font, 0.45, 1)[0]
            cv2.putText(feed, inst2, (w // 2 - inst2_size[0] // 2, btn_y1 - 25), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Draw YOLO Bounding Boxes
        if self.session_state == "PROCTORING":
            for box_info in obj_data.get("boxes", []):
                x1, y1, x2, y2 = box_info["box"]
                cls_name = box_info["class"]
                conf = box_info["conf"]
                
                color = (255, 0, 0) if cls_name == "person" else (0, 0, 255)
                cv2.rectangle(feed, (x1, y1), (x2, y2), color, 2)
                length = min(15, int(abs(x2 - x1) * 0.2))
                cv2.line(feed, (x1, y1), (x1 + length, y1), color, 4)
                cv2.line(feed, (x1, y1), (x1, y1 + length), color, 4)
                cv2.line(feed, (x2, y1), (x2 - length, y1), color, 4)
                cv2.line(feed, (x2, y1), (x2, y1 + length), color, 4)
                cv2.line(feed, (x1, y2), (x1 + length, y2), color, 4)
                cv2.line(feed, (x1, y2), (x1, y2 - length), color, 4)
                cv2.line(feed, (x2, y2), (x2 - length, y2), color, 4)
                cv2.line(feed, (x2, y2), (x2, y2 - length), color, 4)
                
                label = f"{cls_name.upper()} {conf:.2f}"
                cv2.putText(feed, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Draw "PLEASE LOOK AT THE SCREEN!" alert on camera feed
        if look_away_dur >= 10.0:
            box_w = 400
            box_h = 70
            bx = (w - box_w) // 2
            by = (h - box_h) // 2
            
            overlay_warn = feed.copy()
            cv2.rectangle(overlay_warn, (bx, by), (bx + box_w, by + box_h), (0, 0, 150), -1)
            cv2.addWeighted(overlay_warn, 0.7, feed, 0.3, 0, feed)
            
            cv2.rectangle(feed, (bx, by), (bx + box_w, by + box_h), (0, 255, 255), 2)
            cv2.putText(feed, "PLEASE LOOK AT THE SCREEN!", (bx + 15, by + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw Global Overlay Status Bar (Top Right)
        status_box_w = 180
        status_box_h = 40
        overlay_status = feed.copy()
        cv2.rectangle(overlay_status, (w - status_box_w - 10, 10), (w - 10, 10 + status_box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay_status, 0.75, feed, 0.25, 0, feed)
        
        if self.session_state == "START_SCREEN":
            state_txt = "SETUP STATE"
            state_col = (0, 255, 255)
        elif self.session_state == "CALIBRATION":
            state_txt = "CALIBRATING..."
            state_col = (0, 165, 255)
        else:
            state_txt = "SESSION ACTIVE"
            state_col = (0, 255, 0)
            if look_away_dur > 0.0:
                state_txt = f"LOOK AWAY: {look_away_dur:.1f}s"
                state_col = (0, 165, 255) if look_away_dur < 5.0 else (0, 0, 255)
            elif seconds_in_breach > 0:
                state_txt = "INTEGRITY ALERT"
                state_col = (0, 0, 255)
            
        cv2.putText(feed, state_txt, (w - status_box_w + 5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_col, 2, cv2.LINE_AA)

        # 2. Build Telemetry Dashboard Window (300 x 580)
        dash = np.zeros((580, 300, 3), dtype=np.uint8)
        dash[:] = (30, 30, 30) # Dark gray background

        start_y = 40
        panel_w = 300
        
        cv2.putText(dash, "SYSTEM TELEMETRY", (25, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(dash, (25, start_y + 10), (panel_w - 25, start_y + 10), (80, 80, 80), 1)

        # 2a. Gaze Tracker State
        pose = gaze_data.get("head_pose", [0.0, 0.0, 0.0])
        face_det = gaze_data.get("face_detected", False)
        gaze_dev = gaze_data.get("gaze_deviation", 0.0)
        calib = gaze_data.get("calibrated", False)
        target_idx = gaze_data.get("current_target_index", 0)
        total_targets = gaze_data.get("total_targets", 5)

        y = start_y + 35
        if self.session_state == "START_SCREEN":
            calib_txt = "WAITING..."
            calib_col = (0, 165, 255)
        else:
            calib_txt = "READY" if calib else f"CALIB ({target_idx}/{total_targets})"
            calib_col = (0, 255, 0) if calib else (0, 165, 255)
        cv2.putText(dash, "Gaze Tracker: ", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(dash, calib_txt, (140, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, calib_col, 1, cv2.LINE_AA)

        y += 20
        face_txt = "DETECTED" if face_det else "NO FACE"
        face_col = (0, 255, 0) if face_det else (0, 0, 255)
        cv2.putText(dash, "Face State  : ", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(dash, face_txt, (140, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, face_col, 1, cv2.LINE_AA)
        
        y += 20
        cv2.putText(dash, f"Head Pitch  : {pose[0]:.1f}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y += 20
        cv2.putText(dash, f"Head Yaw    : {pose[1]:.1f}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y += 20
        cv2.putText(dash, f"Gaze Dev    : {gaze_dev:.2f}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        if look_away_dur > 0.0:
            y += 20
            look_col = (0, 0, 255) if look_away_dur > 5.0 else (0, 165, 255)
            cv2.putText(dash, f"Look Away   : {look_away_dur:.1f}s", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, look_col, 1, cv2.LINE_AA)

        # 2b. Object detections
        y += 35
        cv2.putText(dash, f"Presence State", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(dash, (25, y + 5), (panel_w - 25, y + 5), (80, 80, 80), 1)
        
        y += 25
        p_count = obj_data.get("person_count", 0)
        phone = obj_data.get("phone_present", False)
        cv2.putText(dash, f"Person Count : {p_count}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y += 20
        phone_txt = "DETECTED" if phone else "ABSENT"
        phone_col = (0, 0, 255) if phone else (0, 255, 0)
        cv2.putText(dash, "Cell Phone   : ", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(dash, phone_txt, (140, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, phone_col, 1, cv2.LINE_AA)

        # 2c. Identity Verification Status
        y += 35
        cv2.putText(dash, f"Authentication", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(dash, (25, y + 5), (panel_w - 25, y + 5), (80, 80, 80), 1)
        
        y += 25
        auth_set = auth_data.get("baseline_set", False)
        verified = auth_data.get("verified", True)
        score = auth_data.get("similarity_score", 1.0)
        
        if not auth_set:
            cv2.putText(dash, "Status: CAPTURING...", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
        else:
            auth_txt = "VERIFIED" if verified else "MATCH FAILED"
            auth_col = (0, 255, 0) if verified else (0, 0, 255)
            cv2.putText(dash, "Identity  : ", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(dash, auth_txt, (110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, auth_col, 1, cv2.LINE_AA)
            y += 20
            cv2.putText(dash, f"Similarity: {score:.3f}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # 2d. Audio levels & VAD speech flag
        y += 35
        cv2.putText(dash, f"Audio Stream", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(dash, (25, y + 5), (panel_w - 25, y + 5), (80, 80, 80), 1)
        
        y += 25
        db = audio_data.get("db_level", 0.0)
        speech = audio_data.get("speech_detected", False)
        
        cv2.putText(dash, "Level: ", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        bar_w = int((db / 100.0) * 120)
        cv2.rectangle(dash, (90, y - 10), (210, y), (50, 50, 50), -1)
        cv2.rectangle(dash, (90, y - 10), (90 + bar_w, y), (0, 255, 0), -1)
        
        y += 25
        speech_txt = "SPEECH DETECTED" if speech else "SILENT / AMBIENT"
        speech_col = (0, 0, 255) if speech else (0, 255, 0)
        cv2.putText(dash, speech_txt, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, speech_col, 1, cv2.LINE_AA)

        # 2e. Session anomaly score
        y += 35
        cv2.putText(dash, f"Session Anomaly Index", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(dash, (25, y + 5), (panel_w - 25, y + 5), (80, 80, 80), 1)
        
        y += 25
        rolling_anomaly = dec_data.get("rolling_anomaly", 0.1)
        seconds_in_breach = dec_data.get("seconds_in_breach", 0.0)
        
        if rolling_anomaly < 0.4:
            anom_color = (0, 255, 0)
        elif rolling_anomaly < 0.7:
            anom_color = (0, 165, 255)
        else:
            anom_color = (0, 0, 255)
            
        cv2.putText(dash, f"Index: {rolling_anomaly:.2f}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, anom_color, 1, cv2.LINE_AA)
        
        y += 15
        bar_anom_w = int(rolling_anomaly * (panel_w - 50))
        cv2.rectangle(dash, (25, y), (panel_w - 25, y + 15), (50, 50, 50), -1)
        cv2.rectangle(dash, (25, y), (25 + bar_anom_w, y + 15), anom_color, -1)
        
        if seconds_in_breach > 0:
            y += 35
            remaining = max(0.0, 5.0 - seconds_in_breach)
            cv2.putText(dash, "BREACH COUNTDOWN", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            y += 20
            cv2.putText(dash, f"TERMINATION IN: {remaining:.1f}s", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Draw face landmarks (green mesh dots) on det frame
        if len(landmarks) > 0:
            for lm in landmarks[::6]:
                cv2.circle(det, lm, 1, (0, 255, 0), -1)

            # Draw gaze crosshairs on pupils (iris centers) on det frame
            if len(landmarks) > 473:
                left_iris = landmarks[468]
                right_iris = landmarks[473]
                
                cv2.circle(det, left_iris, 2, (0, 0, 255), -1) # Red center dot
                cv2.circle(det, right_iris, 2, (0, 0, 255), -1)
                
                l = 8
                # Blue crosshair lines
                cv2.line(det, (left_iris[0] - l, left_iris[1]), (left_iris[0] + l, left_iris[1]), (255, 100, 0), 1)
                cv2.line(det, (left_iris[0], left_iris[1] - l), (left_iris[0], left_iris[1] + l), (255, 100, 0), 1)
                
                cv2.line(det, (right_iris[0] - l, right_iris[1]), (right_iris[0] + l, right_iris[1]), (255, 100, 0), 1)
                cv2.line(det, (right_iris[0], right_iris[1] - l), (right_iris[0], right_iris[1] + l), (255, 100, 0), 1)

                # Draw Gaze Vector lines extending from pupils to the screen gaze coordinate
                if calibrated:
                    pred_x, pred_y = gaze_data.get("screen_gaze", (0.5, 0.5))
                    gx = int(pred_x * w)
                    gy = int(pred_y * h)
                    # Cyan gaze vector lines
                    cv2.line(det, left_iris, (gx, gy), (255, 255, 0), 2)
                    cv2.line(det, right_iris, (gx, gy), (255, 255, 0), 2)
                    # Gaze focus circle
                    cv2.circle(det, (gx, gy), 6, (255, 255, 0), -1)
                    cv2.circle(det, (gx, gy), 10, (255, 255, 0), 1)

        # Draw 3D Head Pose Axes projected on nose
        rvec_list = gaze_data.get("rvec", None)
        tvec_list = gaze_data.get("tvec", None)
        if rvec_list is not None and tvec_list is not None and len(landmarks) > 1:
            rvec = np.array(rvec_list, dtype=np.float32)
            tvec = np.array(tvec_list, dtype=np.float32)
            
            # Reconstruct camera matrix based on frame dimensions
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float32)
            dist_coeffs = np.zeros((4, 1))

            # 3D points for head pose axes: nose tip origin, X (pitch), Y (yaw), Z (roll)
            axis_points = np.array([
                (0.0, 0.0, 0.0),       # Nose tip
                (120.0, 0.0, 0.0),     # X axis (pitch - Red)
                (0.0, 120.0, 0.0),     # Y axis (yaw - Green)
                (0.0, 0.0, 120.0)      # Z axis (roll - Blue)
            ], dtype=np.float32)

            imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts = imgpts.astype(int)
            origin = tuple(imgpts[0].ravel())
            x_end = tuple(imgpts[1].ravel())
            y_end = tuple(imgpts[2].ravel())
            z_end = tuple(imgpts[3].ravel())

            # Draw axes on det frame
            cv2.line(det, origin, x_end, (0, 0, 255), 2)  # Pitch axis: Red
            cv2.line(det, origin, y_end, (0, 255, 0), 2)  # Yaw axis: Green
            cv2.line(det, origin, z_end, (255, 0, 0), 2)  # Roll axis: Blue
            
        # Draw YOLO Bounding Boxes on det frame
        for box_info in obj_data.get("boxes", []):
            x1, y1, x2, y2 = box_info["box"]
            cls_name = box_info["class"]
            conf = box_info["conf"]
            
            color = (255, 0, 0) if cls_name == "person" else (0, 0, 255)
            cv2.rectangle(det, (x1, y1), (x2, y2), color, 2)
            length = min(15, int(abs(x2 - x1) * 0.2))
            cv2.line(det, (x1, y1), (x1 + length, y1), color, 4)
            cv2.line(det, (x1, y1), (x1, y1 + length), color, 4)
            cv2.line(det, (x2, y1), (x2 - length, y1), color, 4)
            cv2.line(det, (x2, y1), (x2, y1 + length), color, 4)
            cv2.line(det, (x1, y2), (x1 + length, y2), color, 4)
            cv2.line(det, (x1, y2), (x1, y2 - length), color, 4)
            cv2.line(det, (x2, y2), (x2 - length, y2), color, 4)
            cv2.line(det, (x2, y2), (x2, y2 - length), color, 4)
            
            label = f"{cls_name.upper()} {conf:.2f}"
            cv2.putText(det, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Draw Detections HUD overlay on det window
        hud_overlay = det.copy()
        cv2.rectangle(hud_overlay, (10, 10), (220, 100), (20, 20, 20), -1)
        cv2.addWeighted(hud_overlay, 0.7, det, 0.3, 0, det)
        
        cv2.putText(det, "DETECTION STATUS", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        
        p_count = obj_data.get("person_count", 0)
        phone = obj_data.get("phone_present", False)
        device = obj_data.get("device_present", False)
        
        cv2.putText(det, f"Person Count: {p_count}", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        phone_txt = "DETECTED" if phone else "ABSENT"
        phone_col = (0, 0, 255) if phone else (0, 255, 0)
        cv2.putText(det, f"Cell Phone: {phone_txt}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, phone_col, 1, cv2.LINE_AA)
        dev_txt = "DETECTED" if device else "ABSENT"
        dev_col = (0, 0, 255) if device else (0, 255, 0)
        cv2.putText(det, f"Other Devices: {dev_txt}", (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.4, dev_col, 1, cv2.LINE_AA)

        return feed, dash, det

    def _draw_termination_alert(self, frame):
        h, w, c = frame.shape
        overlay = frame.copy()
        
        # Red fill over screen
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 150), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Alert box
        box_w = 500
        box_h = 160
        bx = (w - box_w) // 2
        by = (h - box_h) // 2
        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (10, 10, 10), -1)
        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (0, 0, 255), 3)
        
        # Text
        cv2.putText(frame, "PROCTORING SESSION TERMINATED", (bx + 20, by + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Infraction Threshold Exceeded Continuously.", (bx + 20, by + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Saving video buffer and exiting...", (bx + 20, by + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
