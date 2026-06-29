import cv2
import numpy as np
import queue
import time
import threading
import os
from collections import deque

from workers.gaze_worker import GazeWorker
from workers.object_worker import ObjectWorker
from workers.auth_worker import AuthWorker
from workers.audio_worker import AudioWorker
from workers.sequence_worker_rf import SequenceWorkerRF
from workers.second_camera_worker import SecondCameraWorker
from workers.decision_engine_2cam import DecisionEngine2Cam

class MainOrchestrator:
    def __init__(self, camera_index=0, video_path=None, baseline_photo_path=None):
        self.camera_index = camera_index
        self.video_path = video_path
        self.baseline_photo_path = baseline_photo_path
        self.running = False
        self.termination_requested = False
        self.termination_lock = threading.Lock()
        
        # UI/Calibration States
        self.session_state = "START_SCREEN" # "START_SCREEN", "CALIBRATION", "PROCTORING"
        self.start_btn_coords = (0, 0, 0, 0)
        self.cancel_btn_coords = (0, 0, 0, 0)
        self.whole_test_writer = None
        self.gaze_history = deque(maxlen=90)  # ~3 seconds of history at 30 FPS
        
        # Thread-safe Queues
        self.gaze_queue = queue.Queue(maxsize=1)
        self.object_queue = queue.Queue(maxsize=1)
        self.auth_queue = queue.Queue(maxsize=1)
        self.sequence_queue = queue.Queue()
        self.decision_queue = queue.Queue()
 
        # Initialize Workers
        self.decision_engine = DecisionEngine2Cam(
            self.decision_queue, 
            termination_callback=self.trigger_termination,
            infraction_callback=self.handle_infraction_saving
        )
        self.gaze_worker = GazeWorker(self.gaze_queue, self.sequence_queue)
        self.object_worker = ObjectWorker(self.object_queue, self.sequence_queue)
        self.auth_worker = AuthWorker(self.auth_queue, self.decision_queue)
        self.audio_worker = AudioWorker(self.decision_queue)
        self.sequence_worker = SequenceWorkerRF(self.sequence_queue, self.decision_queue)
        self.second_camera_worker = SecondCameraWorker(self.decision_queue, self.auth_queue)

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
        self.second_camera_worker.start()
        
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
        self.second_camera_worker.stop()
        self.decision_engine.stop()
        
        # Join threads
        self.gaze_worker.join(timeout=1.0)
        self.object_worker.join(timeout=1.0)
        self.auth_worker.join(timeout=1.0)
        self.audio_worker.join(timeout=1.0)
        self.sequence_worker.join(timeout=1.0)
        self.second_camera_worker.join(timeout=1.0)
        self.decision_engine.join(timeout=1.0)
        
        cv2.destroyAllWindows()
        print("[Orchestrator] Pipeline shutdown complete.")

        # Merge whole session video and audio
        if hasattr(self, 'whole_test_timestamp'):
            timestamp_str = self.whole_test_timestamp
            video_path = f"session_recordings/temp_whole_test_{timestamp_str}.mp4"
            audio_path = f"session_recordings/whole_test_{timestamp_str}.wav"
            final_video_path = f"session_recordings/whole_test_{timestamp_str}.mp4"
            
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
                    print("[Orchestrator] Merging whole session video and audio...")
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if os.path.exists(final_video_path) and os.path.getsize(final_video_path) > 0:
                        os.remove(video_path)
                        os.remove(audio_path)
                        print(f"[Orchestrator] Saved whole session video & audio merged into: {final_video_path}")
                except Exception as e:
                    print(f"[Orchestrator] Error merging whole session audio and video: {e}")

    def _run_capture_loop(self):
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

        # Initialize baseline frame from uploaded photo or webcam snapshot
        baseline_set = False
        if self.baseline_photo_path and os.path.exists(self.baseline_photo_path):
            baseline_img = cv2.imread(self.baseline_photo_path)
            if baseline_img is not None:
                self.auth_worker.set_baseline_frame(baseline_img)
                print(f"[Orchestrator] Baseline identity frame loaded from file: {self.baseline_photo_path}")
                baseline_set = True
            else:
                print(f"[Orchestrator] Error: Could not read baseline photo '{self.baseline_photo_path}'")
                
        time.sleep(1.0)
        ret, frame = cap.read()
        if ret:
            if not baseline_set:
                self.auth_worker.set_baseline_frame(frame)
                print("[Orchestrator] Baseline frame set from webcam snapshot.")
            self._start_whole_test_recording(frame)

        feed_window = "Live Feed - Proctored Session"
        dash_window = "Proctoring Telemetry Dashboard"
        det_window = "Detections & Tracking Viewer"
        sec_window = "Second Camera Feed - Side Angle"
        
        cv2.namedWindow(feed_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(dash_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(det_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(sec_window, cv2.WINDOW_NORMAL)
        
        cv2.setWindowProperty(feed_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow(dash_window, 300, 580)
        cv2.resizeWindow(det_window, 640, 480)
        cv2.resizeWindow(sec_window, 640, 480)

        cv2.setMouseCallback(feed_window, self._on_mouse_click)
        frame_time = 1.0 / 30.0

        while self.running:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                if self.video_path is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[Orchestrator] Webcam stream disconnected.")
                    break

            current_time = time.time()
            
            # Gaze
            if current_time - self.last_sent["gaze"] >= 1.0 / self.gaze_fps:
                try:
                    if self.gaze_queue.full():
                        self.gaze_queue.get_nowait()
                    self.gaze_queue.put_nowait(frame)
                    self.last_sent["gaze"] = current_time
                except queue.Full:
                    pass

            # YOLOv8 Object
            if current_time - self.last_sent["object"] >= 1.0 / self.object_fps:
                try:
                    if self.object_queue.full():
                        self.object_queue.get_nowait()
                    self.object_queue.put_nowait(frame)
                    self.last_sent["object"] = current_time
                except queue.Full:
                    pass

            # DeepFace Auth
            auth_interval = 10.0
            if self.session_state == "START_SCREEN":
                auth_interval = 2.0  # Verify faster on start screen
            
            if current_time - self.last_sent["auth"] >= auth_interval:
                try:
                    if self.auth_queue.full():
                        self.auth_queue.get_nowait()
                    self.auth_queue.put_nowait({"source": "primary", "frame": frame})
                    self.last_sent["auth"] = current_time
                except queue.Full:
                    pass

            # Retrieve data from all workers
            gaze_data = self.gaze_worker.get_latest_data()
            obj_data = self.object_worker.get_latest_data()
            auth_data = self.auth_worker.get_latest_data()
            audio_data = self.audio_worker.get_latest_data()
            dec_data = self.decision_engine.get_latest_data()
            sec_data = self.second_camera_worker.get_latest_data()

            # Handle calibration transition
            if self.session_state == "CALIBRATION" and gaze_data.get("calibrated", False):
                print("[Orchestrator] Calibration solved! Transitioning to active proctoring...")
                self.session_state = "PROCTORING"
                cv2.setWindowProperty(feed_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(feed_window, 640, 480)
                
                # Activate proctoring checks
                self.sequence_worker.set_session_active(True)
                self.decision_engine.set_session_active(True)
                self.second_camera_worker.active_proctoring = True

            # Retrieve second camera frame
            sec_frame = self.second_camera_worker.get_annotated_frame()
            if sec_frame is None:
                # Draw placeholder instructions with QR code
                sec_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                sec_frame[:] = (20, 20, 20)
                qr_img = self.second_camera_worker.get_qr_image()
                if qr_img is not None:
                    qh, qw, _ = qr_img.shape
                    qy1 = (480 - qh) // 2 - 20
                    qy2 = qy1 + qh
                    qx1 = (640 - qw) // 2
                    qx2 = qx1 + qw
                    sec_frame[qy1:qy2, qx1:qx2] = qr_img
                    cv2.rectangle(sec_frame, (qx1 - 2, qy1 - 2), (qx2 + 2, qy2 + 2), (255, 255, 255), 2)
                    
                cv2.putText(sec_frame, "SECOND CAMERA - PHONE STREAM", (160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(sec_frame, "Scan the QR code to connect your phone", (145, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(sec_frame, f"URL: {self.second_camera_worker.server_url}", (155, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                sec_frame = cv2.resize(sec_frame, (640, 480))

            # Render HUD and Dashboards
            feed_img, dash_img, det_img = self._draw_hud_and_dashboard(frame, gaze_data, obj_data, auth_data, audio_data, dec_data, sec_data)

            # Create side-by-side combined frame
            dh, dw, dc = det_img.shape
            sec_resized = cv2.resize(sec_frame, (dw, dh))
            combined_frame = np.hstack((det_img, sec_resized))

            # Record frame
            if hasattr(self, 'whole_test_writer') and self.whole_test_writer is not None:
                self.whole_test_writer.write(combined_frame)

            # Push the annotated combined frame to the decision engine ring buffer
            self.decision_engine.push_frame_to_buffer(combined_frame)

            cv2.imshow(feed_window, feed_img)
            cv2.imshow(dash_window, dash_img)
            cv2.imshow(det_window, det_img)
            cv2.imshow(sec_window, sec_frame)

            with self.termination_lock:
                if self.termination_requested:
                    self._draw_termination_alert(feed_img)
                    cv2.imshow(feed_window, feed_img)
                    cv2.waitKey(2000)
                    break

            # Handle keystrokes (Press 'q' or ESC key 27 to cancel/exit)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[Orchestrator] User requested cancellation/termination via keyboard.")
                break

            elapsed = time.time() - start_time
            sleep_duration = max(0.001, frame_time - elapsed)
            time.sleep(sleep_duration)

        cap.release()
        self.stop_pipeline()

    def _on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.session_state == "START_SCREEN":
                # Start Button
                btn_x1, btn_y1, btn_x2, btn_y2 = self.start_btn_coords
                if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
                    auth_data = self.auth_worker.get_latest_data()
                    if auth_data.get("verified", False):
                        print("[Orchestrator] Start button clicked! Identity verified. Initializing gaze calibration...")
                        self.session_state = "CALIBRATION"
                        self.gaze_worker.start_calibration()
                    else:
                        print("[Orchestrator] Start button clicked but identity is not verified yet. Please look at the camera.")
                
                # Cancel Button
                c_x1, c_y1, c_x2, c_y2 = self.cancel_btn_coords
                if c_x1 <= x <= c_x2 and c_y1 <= y <= c_y2:
                    print("[Orchestrator] Cancel button clicked. Exiting...")
                    self.trigger_termination()

    def _start_whole_test_recording(self, frame):
        os.makedirs("session_recordings", exist_ok=True)
        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        self.whole_test_timestamp = timestamp_str
        h, w, c = frame.shape
        video_path = f"session_recordings/temp_whole_test_{timestamp_str}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.whole_test_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (2 * w, h))
        print(f"[Orchestrator] Started whole session video recording: {video_path}")
        self.audio_worker.start_whole_test_recording(timestamp_str)

    def _draw_hud_and_dashboard(self, frame, gaze_data, obj_data, auth_data, audio_data, dec_data, sec_data):
        feed = frame.copy()
        h, w, c = feed.shape
        det = frame.copy()
        landmarks = gaze_data.get("landmarks_2d", [])

        if self.session_state == "PROCTORING":
            feed = np.zeros_like(frame)

        tl_x, tl_y = int(0.03 * w), int(0.03 * h)
        br_x, br_y = int(0.97 * w), int(0.97 * h)
        
        calibrated = gaze_data.get("calibrated", False)
        look_away_dur = self.sequence_worker.get_latest_data().get("look_away_duration", 0.0)
        seconds_in_breach = dec_data.get("seconds_in_breach", 0.0)
        
        if self.session_state == "START_SCREEN":
            box_color = (0, 165, 255)
            thickness = 2
        elif self.session_state == "CALIBRATION":
            box_color = (0, 255, 255)
            thickness = 2
        else:
            if look_away_dur >= 10.0:
                box_color = (0, 0, 255)
                thickness = 3
            elif look_away_dur > 0.0:
                box_color = (0, 165, 255)
                thickness = 2
            else:
                box_color = (0, 255, 0)
                thickness = 1
                
        cv2.rectangle(feed, (tl_x, tl_y), (br_x, br_y), box_color, thickness)
        cv2.putText(feed, "MONITOR AREA BOUNDS", (tl_x + 10, tl_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA)

        if self.session_state == "PROCTORING":
            if calibrated and gaze_data.get("face_detected", False):
                pred_x, pred_y = gaze_data.get("screen_gaze", (0.5, 0.5))
                gx = int(pred_x * w)
                gy = int(pred_y * h)
                
                # Constrain coordinates to screen boundaries
                gx = max(0, min(w - 1, gx))
                gy = max(0, min(h - 1, gy))
                
                # Append to history
                self.gaze_history.append((gx, gy))
                
                # Determine where user is looking
                if look_away_dur > 0.0:
                    gaze_desc = "LOOKING AWAY"
                    gaze_col = (0, 0, 255) # Red
                else:
                    # Map to screen region
                    horiz = "CENTER"
                    if gx < 0.33 * w:
                        horiz = "LEFT"
                    elif gx > 0.67 * w:
                        horiz = "RIGHT"
                        
                    vert = "MIDDLE"
                    if gy < 0.33 * h:
                        vert = "TOP"
                    elif gy > 0.67 * h:
                        vert = "BOTTOM"
                        
                    if horiz == "CENTER" and vert == "MIDDLE":
                        gaze_desc = "LOOKING CENTER"
                    else:
                        gaze_desc = f"LOOKING {horiz} {vert}".replace(" MIDDLE", "").replace("CENTER ", "")
                    gaze_col = (0, 255, 0) # Green
                
                # Draw the gaze trajectory line (fading trail)
                if len(self.gaze_history) > 1:
                    for i in range(len(self.gaze_history) - 1):
                        pt1 = self.gaze_history[i]
                        pt2 = self.gaze_history[i+1]
                        if pt1 is not None and pt2 is not None:
                            # Fading color (bright green to dark green)
                            ratio = i / len(self.gaze_history)
                            intensity = int(60 + ratio * 195)
                            thickness = int(1 + ratio * 3)
                            cv2.line(feed, pt1, pt2, (0, intensity, 0), thickness)
                
                # Draw final target gaze dot
                cv2.circle(feed, (gx, gy), 5, (0, 255, 255), -1) # Glowing yellow dot at tip
                cv2.circle(feed, (gx, gy), 9, (0, 255, 255), 1)
                
                # Draw dynamic coordinate label next to the gaze point
                lbl_x = gx + 15 if gx < w - 180 else gx - 180
                lbl_y = gy + 5
                cv2.putText(feed, f"{gaze_desc} ({gx}, {gy})", (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                
                # Write where user is looking on the black screen
                cv2.putText(feed, "GAZE REGION :", (tl_x + 20, tl_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(feed, gaze_desc, (tl_x + 140, tl_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gaze_col, 2, cv2.LINE_AA)
                cv2.putText(feed, f"COORDINATES : X={gx}, Y={gy}", (tl_x + 20, tl_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            else:
                self.gaze_history.clear()
                status_msg = "LOOKING AWAY (NO FACE)" if not gaze_data.get("face_detected", False) else "NOT CALIBRATED"
                cv2.putText(feed, "GAZE REGION :", (tl_x + 20, tl_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(feed, status_msg, (tl_x + 140, tl_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # Calibration targets
        if self.session_state == "CALIBRATION":
            overlay_calib = feed.copy()
            cv2.rectangle(overlay_calib, (0, 0), (w, 55), (15, 15, 15), -1)
            cv2.addWeighted(overlay_calib, 0.8, feed, 0.2, 0, feed)
            
            cv2.putText(feed, "GAZE CALIBRATION: LOOK AT THE TARGET BALLS", (w // 2 - 220, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(feed, "Keep your eyes on the orange center as the white ring shrinks", (w // 2 - 215, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            tx, ty = gaze_data.get("calibration_target", (0.5, 0.5))
            point_progress = gaze_data.get("calibration_progress_point", 0.0)
            target_idx = gaze_data.get("current_target_index", 0)
            total_targets = gaze_data.get("total_targets", 5)
            
            px = int(tx * w)
            py = int(ty * h)
            
            cv2.circle(feed, (px, py), 10, (0, 165, 255), -1)
            cv2.circle(feed, (px, py), 10, (0, 255, 255), 2)
            cv2.circle(feed, (px, py), 2, (0, 0, 255), -1)
            
            max_ring_radius = 35
            min_ring_radius = 12
            ring_radius = int(max_ring_radius - point_progress * (max_ring_radius - min_ring_radius))
            cv2.circle(feed, (px, py), ring_radius, (255, 255, 255), 2)
            cv2.putText(feed, f"{target_idx + 1}/{total_targets}", (px + 15, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            if not gaze_data.get("face_detected", False):
                cv2.putText(feed, "FACE NOT DETECTED - PLEASE LOOK AT THE TARGET BALL", (w // 2 - 210, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)

        # Start Screen with responsive layout (QR on left, Start button on right)
        if self.session_state == "START_SCREEN":
            overlay = feed.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.70, feed, 0.30, 0, feed)

            # Responsive coords
            qr_w, qr_h = 200, 200
            qr_x1 = int(0.12 * w)
            qr_y1 = int(h // 2 - qr_h // 2)
            qr_x2 = qr_x1 + qr_w
            qr_y2 = qr_y1 + qr_h
            
            # Start button on the right
            btn_w, btn_h = 220, 45
            btn1_x1 = int(w * 0.88 - btn_w)
            btn1_y1 = int(h // 2 - 55)
            btn1_x2 = btn1_x1 + btn_w
            btn1_y2 = btn1_y1 + btn_h
            self.start_btn_coords = (btn1_x1, btn1_y1, btn1_x2, btn1_y2)

            # Cancel button on the right
            btn2_x1 = btn1_x1
            btn2_y1 = int(h // 2 + 10)
            btn2_x2 = btn1_x2
            btn2_y2 = btn2_y1 + btn_h
            self.cancel_btn_coords = (btn2_x1, btn2_y1, btn2_x2, btn2_y2)

            # Draw QR Code image
            qr_img = self.second_camera_worker.get_qr_image()
            if qr_img is not None:
                feed[qr_y1:qr_y2, qr_x1:qr_x2] = qr_img
                cv2.rectangle(feed, (qr_x1 - 2, qr_y1 - 2), (qr_x2 + 2, qr_y2 + 2), (255, 255, 255), 2)
                cv2.putText(feed, "CONNECT SIDE PHONE CAM", (qr_x1 - 10, qr_y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                
                # Draw all detected adapter IPs for fallback options
                ip_y = qr_y2 + 40
                for idx, ip in enumerate(self.second_camera_worker.all_detected_ips[:3]):
                    cv2.putText(feed, f"IP {idx+1}: {ip}:{self.second_camera_worker.port}", (qr_x1 - 5, ip_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
                    ip_y += 15

            # Identity Verification status
            is_verified = auth_data.get("verified", False)
            sim_score = auth_data.get("similarity_score", 0.0)
            if is_verified:
                auth_status_txt = f"IDENTITY: VERIFIED ({sim_score:.2f})"
                auth_status_col = (0, 255, 0)
                btn_color = (0, 165, 255) # Orange
                btn_text_color = (255, 255, 255)
            else:
                auth_status_txt = f"IDENTITY: NOT VERIFIED ({sim_score:.2f})"
                auth_status_col = (0, 0, 255)
                btn_color = (80, 80, 80) # Gray
                btn_text_color = (160, 160, 160)

            cv2.putText(feed, auth_status_txt, (btn1_x1 + 10, btn1_y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, auth_status_col, 2, cv2.LINE_AA)

            # Connection status
            if sec_data.get("connected", False):
                status_txt = "SIDE CAMERA: READY"
                status_col = (0, 255, 0)
            else:
                status_txt = "SIDE CAMERA: OPTIONAL"
                status_col = (0, 165, 255)
            cv2.putText(feed, status_txt, (btn1_x1 + 10, btn1_y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_col, 2, cv2.LINE_AA)

            # Draw start button
            cv2.rectangle(feed, (btn1_x1, btn1_y1), (btn1_x2, btn1_y2), btn_color, -1)
            cv2.rectangle(feed, (btn1_x1, btn1_y1), (btn1_x2, btn1_y2), (255, 255, 255), 1)
            text1 = "START SESSION"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text1_size = cv2.getTextSize(text1, font, 0.55, 2)[0]
            tx1 = btn1_x1 + (btn1_x2 - btn1_x1) // 2 - text1_size[0] // 2
            ty1 = btn1_y1 + (btn1_y2 - btn1_y1) // 2 + text1_size[1] // 2
            cv2.putText(feed, text1, (tx1, ty1), font, 0.55, btn_text_color, 2, cv2.LINE_AA)

            # Draw Cancel button
            cv2.rectangle(feed, (btn2_x1, btn2_y1), (btn2_x2, btn2_y2), (0, 0, 200), -1)
            cv2.rectangle(feed, (btn2_x1, btn2_y1), (btn2_x2, btn2_y2), (255, 255, 255), 1)
            text2 = "CANCEL TEST"
            text2_size = cv2.getTextSize(text2, font, 0.55, 2)[0]
            tx2 = btn2_x1 + (btn2_x2 - btn2_x1) // 2 - text2_size[0] // 2
            ty2 = btn2_y1 + (btn2_y2 - btn2_y1) // 2 + text2_size[1] // 2
            cv2.putText(feed, text2, (tx2, ty2), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            inst1 = "PROCTORING TRACKING ACTIVE"
            inst1_size = cv2.getTextSize(inst1, font, 0.70, 2)[0]
            cv2.putText(feed, inst1, (w // 2 - inst1_size[0] // 2, 40), font, 0.70, (0, 255, 255), 2, cv2.LINE_AA)

            if not is_verified:
                inst_text = "Identity Verification Pending - Look at Webcam"
                inst_size = cv2.getTextSize(inst_text, font, 0.42, 1)[0]
                cv2.putText(feed, inst_text, (w // 2 - inst_size[0] // 2, h - 30), font, 0.42, (0, 165, 255), 1, cv2.LINE_AA)

        # Draw YOLO Boxes (Main Webacm)
        if self.session_state == "PROCTORING":
            for box_info in obj_data.get("boxes", []):
                x1, y1, x2, y2 = box_info["box"]
                cls_name = box_info["class"]
                conf = box_info["conf"]
                
                color = (255, 0, 0) if cls_name == "person" else (0, 0, 255)
                cv2.rectangle(feed, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name.upper()} {conf:.2f}"
                cv2.putText(feed, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if look_away_dur >= 10.0:
            box_w, box_h = 400, 70
            bx, by = (w - box_w) // 2, (h - box_h) // 2
            overlay_warn = feed.copy()
            cv2.rectangle(overlay_warn, (bx, by), (bx + box_w, by + box_h), (0, 0, 150), -1)
            cv2.addWeighted(overlay_warn, 0.7, feed, 0.3, 0, feed)
            cv2.rectangle(feed, (bx, by), (bx + box_w, by + box_h), (0, 255, 255), 2)
            cv2.putText(feed, "PLEASE LOOK AT THE SCREEN!", (bx + 15, by + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        # Status Bar Overlay
        status_box_w, status_box_h = 180, 40
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
                state_col = (0, 0, 255) if look_away_dur >= 5.0 else (0, 165, 255)
            elif seconds_in_breach > 0:
                state_txt = "INTEGRITY ALERT"
                state_col = (0, 0, 255)
            
        cv2.putText(feed, state_txt, (w - status_box_w + 5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_col, 2, cv2.LINE_AA)

        # Telemetry Dashboard
        dash = np.zeros((580, 300, 3), dtype=np.uint8)
        dash[:] = (30, 30, 30)

        start_y = 35
        panel_w = 300
        cv2.putText(dash, "SYSTEM TELEMETRY", (25, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(dash, (25, start_y + 10), (panel_w - 25, start_y + 10), (80, 80, 80), 1)

        # Gaze State
        pose = gaze_data.get("head_pose", [0.0, 0.0, 0.0])
        face_det = gaze_data.get("face_detected", False)
        gaze_dev = gaze_data.get("gaze_deviation", 0.0)
        calib = gaze_data.get("calibrated", False)
        target_idx = gaze_data.get("current_target_index", 0)
        total_targets = gaze_data.get("total_targets", 5)

        y = start_y + 30
        calib_txt = "READY" if calib else f"CALIB ({target_idx}/{total_targets})"
        calib_col = (0, 255, 0) if calib else (0, 165, 255)
        cv2.putText(dash, "Gaze Tracker: ", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(dash, calib_txt, (140, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, calib_col, 1, cv2.LINE_AA)

        y += 18
        face_txt = "DETECTED" if face_det else "NO FACE"
        face_col = (0, 255, 0) if face_det else (0, 0, 255)
        cv2.putText(dash, "Face State  : ", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(dash, face_txt, (140, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, face_col, 1, cv2.LINE_AA)
        
        y += 18
        cv2.putText(dash, f"Head Pitch  : {pose[0]:.1f}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y += 18
        cv2.putText(dash, f"Head Yaw    : {pose[1]:.1f}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y += 18
        cv2.putText(dash, f"Gaze Dev    : {gaze_dev:.2f}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Presence State
        y += 30
        cv2.putText(dash, "Presence State", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(dash, (25, y + 5), (panel_w - 25, y + 5), (80, 80, 80), 1)
        
        y += 22
        p_count = obj_data.get("person_count", 0)
        phone = obj_data.get("phone_present", False)
        cv2.putText(dash, f"Person Count : {p_count}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y += 18
        phone_txt = "DETECTED" if phone else "ABSENT"
        phone_col = (0, 0, 255) if phone else (0, 255, 0)
        cv2.putText(dash, "Cell Phone   : ", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(dash, phone_txt, (140, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, phone_col, 1, cv2.LINE_AA)

        # Identity Verification
        y += 30
        cv2.putText(dash, "Authentication", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(dash, (25, y + 5), (panel_w - 25, y + 5), (80, 80, 80), 1)
        
        y += 22
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
            y += 18
            cv2.putText(dash, f"Similarity: {score:.3f}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Audio Stream
        y += 30
        cv2.putText(dash, "Audio Stream", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(dash, (25, y + 5), (panel_w - 25, y + 5), (80, 80, 80), 1)
        
        y += 22
        db = audio_data.get("db_level", 0.0)
        speech = audio_data.get("speech_detected", False)
        cv2.putText(dash, "Level: ", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        bar_w = int((db / 100.0) * 120)
        cv2.rectangle(dash, (90, y - 10), (210, y), (50, 50, 50), -1)
        cv2.rectangle(dash, (90, y - 10), (90 + bar_w, y), (0, 255, 0), -1)
        y += 22
        speech_txt = "SPEECH DETECTED" if speech else "SILENT / AMBIENT"
        speech_col = (0, 0, 255) if speech else (0, 255, 0)
        cv2.putText(dash, speech_txt, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, speech_col, 1, cv2.LINE_AA)

        # Second Camera Dashboard telemetry
        y += 30
        cv2.putText(dash, "Side Camera (Phone)", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(dash, (25, y + 5), (panel_w - 25, y + 5), (80, 80, 80), 1)
        
        y += 22
        sec_connected = dec_data.get("second_cam_connected", False)
        sec_phone = dec_data.get("second_cam_phone", False)
        sec_person = dec_data.get("second_cam_person", False)
        
        sec_conn_txt = "CONNECTED" if sec_connected else "DISCONNECTED"
        sec_conn_col = (0, 255, 0) if sec_connected else (0, 0, 255)
        cv2.putText(dash, "Connection: ", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(dash, sec_conn_txt, (140, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sec_conn_col, 1, cv2.LINE_AA)
        
        y += 18
        sec_phone_txt = "DETECTED" if sec_phone else "ABSENT"
        sec_phone_col = (0, 0, 255) if sec_phone else (0, 255, 0)
        cv2.putText(dash, "Phone (Side): ", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(dash, sec_phone_txt, (140, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sec_phone_col, 1, cv2.LINE_AA)
        
        y += 18
        sec_pers_txt = "DETECTED" if sec_person else "ABSENT"
        sec_pers_col = (0, 0, 255) if sec_person else (0, 255, 0)
        cv2.putText(dash, "Person (Side): ", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(dash, sec_pers_txt, (140, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sec_pers_col, 1, cv2.LINE_AA)

        # Anomaly Index
        y += 30
        cv2.putText(dash, "Session Anomaly Index", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(dash, (25, y + 5), (panel_w - 25, y + 5), (80, 80, 80), 1)
        
        y += 20
        rolling_anomaly = dec_data.get("rolling_anomaly", 0.1)
        
        if rolling_anomaly < 0.4:
            anom_color = (0, 255, 0)
        elif rolling_anomaly < 0.7:
            anom_color = (0, 165, 255)
        else:
            anom_color = (0, 0, 255)
            
        cv2.putText(dash, f"Index: {rolling_anomaly:.2f}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, anom_color, 1, cv2.LINE_AA)
        
        y += 10
        bar_anom_w = int(rolling_anomaly * (panel_w - 50))
        cv2.rectangle(dash, (25, y), (panel_w - 25, y + 10), (50, 50, 50), -1)
        cv2.rectangle(dash, (25, y), (25 + bar_anom_w, y + 10), anom_color, -1)
        
        if seconds_in_breach > 0:
            y += 28
            remaining = max(0.0, 3.0 - seconds_in_breach)
            cv2.putText(dash, "BREACH COUNTDOWN", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            y += 18
            cv2.putText(dash, f"TERMINATION IN: {remaining:.1f}s", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # 3. Draw CV tracking overlays on det window
        if len(landmarks) > 0:
            for lm in landmarks[::6]:
                cv2.circle(det, lm, 1, (0, 255, 0), -1)

            if len(landmarks) > 473:
                left_iris = landmarks[468]
                right_iris = landmarks[473]
                cv2.circle(det, left_iris, 2, (0, 0, 255), -1)
                cv2.circle(det, right_iris, 2, (0, 0, 255), -1)
                
                l = 8
                cv2.line(det, (left_iris[0] - l, left_iris[1]), (left_iris[0] + l, left_iris[1]), (255, 100, 0), 1)
                cv2.line(det, (left_iris[0], left_iris[1] - l), (left_iris[0], left_iris[1] + l), (255, 100, 0), 1)
                cv2.line(det, (right_iris[0] - l, right_iris[1]), (right_iris[0] + l, right_iris[1]), (255, 100, 0), 1)
                cv2.line(det, (right_iris[0], right_iris[1] - l), (right_iris[0], right_iris[1] + l), (255, 100, 0), 1)

                if calibrated:
                    pred_x, pred_y = gaze_data.get("screen_gaze", (0.5, 0.5))
                    gx = int(pred_x * w)
                    gy = int(pred_y * h)
                    
                    # Draw EyeTrax target cursor style instead of drawing cyan gaze vector lines
                    from eyetrax.utils.draw import draw_cursor
                    draw_cursor(det, gx, gy, 1.0, radius_outer=12, radius_inner=8, color_outer=(0, 0, 255), color_inner=(255, 255, 255))

        rvec_list = gaze_data.get("rvec", None)
        tvec_list = gaze_data.get("tvec", None)
        if rvec_list is not None and tvec_list is not None and len(landmarks) > 1:
            rvec = np.array(rvec_list, dtype=np.float32)
            tvec = np.array(tvec_list, dtype=np.float32)
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float32)
            dist_coeffs = np.zeros((4, 1))
            axis_points = np.array([
                (0.0, 0.0, 0.0),
                (120.0, 0.0, 0.0),
                (0.0, 120.0, 0.0),
                (0.0, 0.0, 120.0)
            ], dtype=np.float32)

            imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts = imgpts.astype(int)
            origin = tuple(imgpts[0].ravel())
            x_end = tuple(imgpts[1].ravel())
            y_end = tuple(imgpts[2].ravel())
            z_end = tuple(imgpts[3].ravel())

            cv2.line(det, origin, x_end, (0, 0, 255), 2)
            cv2.line(det, origin, y_end, (0, 255, 0), 2)
            cv2.line(det, origin, z_end, (255, 0, 0), 2)
            
        for box_info in obj_data.get("boxes", []):
            x1, y1, x2, y2 = box_info["box"]
            cls_name = box_info["class"]
            conf = box_info["conf"]
            
            color = (255, 0, 0) if cls_name == "person" else (0, 0, 255)
            cv2.rectangle(det, (x1, y1), (x2, y2), color, 2)
            cv2.putText(det, f"{cls_name.upper()} {conf:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Detections viewer HUD overlay
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
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 150), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        box_w, box_h = 500, 160
        bx, by = (w - box_w) // 2, (h - box_h) // 2
        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (10, 10, 10), -1)
        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (0, 0, 255), 3)
        
        cv2.putText(frame, "PROCTORING SESSION TERMINATED", (bx + 20, by + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Infraction Threshold Exceeded Continuously.", (bx + 20, by + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Saving video buffer and exiting...", (bx + 20, by + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
