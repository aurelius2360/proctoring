import cv2
import threading
import queue
import time
import socket
import numpy as np
import qrcode
from http.server import BaseHTTPRequestHandler, HTTPServer
from ultralytics import YOLO

HTML_CONTENT = """<!DOCTYPE html>
<html>
<head>
    <title>Proctoring Second Camera</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            background: #121212;
            color: #ffffff;
            margin: 0;
            padding: 20px;
        }
        h1 { font-size: 1.5rem; margin-top: 10px; color: #00e5ff; }
        p { font-size: 0.9rem; color: #b0bec5; }
        video {
            width: 100%;
            max-width: 480px;
            border: 2px solid #00e5ff;
            border-radius: 12px;
            background: black;
            box-shadow: 0 4px 15px rgba(0, 229, 255, 0.2);
        }
        #status-container {
            margin: 20px auto 10px auto;
            padding: 10px;
            max-width: 320px;
            border-radius: 8px;
            background: #1e1e1e;
        }
        #status {
            font-size: 1rem;
            font-weight: bold;
            color: #ffb300;
        }
        #toggle-btn {
            background: #00e5ff;
            color: #121212;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 0.95rem;
            font-weight: bold;
            cursor: pointer;
            margin: 10px auto 20px auto;
            display: inline-block;
            transition: background 0.2s ease;
        }
        #toggle-btn:hover {
            background: #00b0ff;
        }
        #error {
            color: #ff1744;
            margin-top: 15px;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <h1>Proctoring Second Camera</h1>
    <p>Place your phone to the side, pointing at your hands, keyboard, and screen.</p>
    <video id="webcam" autoplay playsinline></video>
    <div id="status-container">
        Status: <span id="status">INITIALIZING...</span>
    </div>
    <button id="toggle-btn">Toggle Camera Source (Front/Back)</button>
    <div id="error"></div>
    <canvas id="canvas" style="display:none;"></canvas>

    <script>
        const video = document.getElementById('webcam');
        const statusSpan = document.getElementById('status');
        const errorDiv = document.getElementById('error');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        let currentFacingMode = "environment";
        let localStream = null;

        function startCamera() {
            if (localStream) {
                localStream.getTracks().forEach(track => track.stop());
            }

            const constraints = {
                video: {
                    facingMode: currentFacingMode,
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            };

            navigator.mediaDevices.getUserMedia(constraints)
                .then(stream => {
                    localStream = stream;
                    video.srcObject = stream;
                    statusSpan.innerText = "CONNECTING...";
                    statusSpan.style.color = "#ffb300";
                })
                .catch(err => {
                    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
                        .then(stream => {
                            localStream = stream;
                            video.srcObject = stream;
                            statusSpan.innerText = "CONNECTING...";
                            statusSpan.style.color = "#ffb300";
                        })
                        .catch(err2 => {
                            errorDiv.innerText = "Camera Access Error: " + err2.message;
                            statusSpan.innerText = "FAILED";
                            statusSpan.style.color = "#ff1744";
                        });
                });
        }

        // Initialize camera
        startCamera();
        
        // Start sending frames at ~7 FPS (every 140ms)
        setInterval(sendFrame, 140);

        // Toggle camera facing mode
        document.getElementById('toggle-btn').addEventListener('click', () => {
            currentFacingMode = (currentFacingMode === "environment") ? "user" : "environment";
            startCamera();
        });

        function sendFrame() {
            if (video.readyState === video.HAVE_ENOUGH_DATA) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                canvas.toBlob(blob => {
                    if (!blob) return;
                    
                    fetch('/upload', {
                        method: 'POST',
                        body: blob,
                        headers: {
                            'Content-Type': 'image/jpeg'
                        }
                    })
                    .then(res => {
                        if (res.ok) {
                            statusSpan.innerText = "STREAMING ACTIVE";
                            statusSpan.style.color = "#00e676";
                        } else {
                            statusSpan.innerText = "SERVER ERROR";
                            statusSpan.style.color = "#ff1744";
                        }
                    })
                    .catch(err => {
                        statusSpan.innerText = "DISCONNECTED";
                        statusSpan.style.color = "#ff1744";
                    });
                }, 'image/jpeg', 0.6);
            }
        }
    </script>
</body>
</html>
"""

class PhoneCameraHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging HTTP requests to stdout
        pass

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/upload':
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                raw_data = self.rfile.read(content_length)
                nparr = np.frombuffer(raw_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    self.server.worker_ref.update_frame(frame)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

class SecondCameraWorker(threading.Thread):
    def __init__(self, decision_queue, auth_queue=None):
        super().__init__()
        self.decision_queue = decision_queue
        self.auth_queue = auth_queue
        self.running = False
        self.daemon = True

        # Frame buffers
        self.latest_raw_frame = None
        self.annotated_frame = None
        self.last_frame_time = 0.0
        self.frame_lock = threading.Lock()

        # Telemetry
        self.connected = False
        self.ever_connected = False
        self.phone_detected = False
        self.person_detected = False
        self.boxes = []
        self.active_proctoring = False
        self.last_ngrok_check = 0.0

        # Start HTTP server on available port starting at 8080
        self.port = 8080
        self.server = None
        while self.port < 8100:
            try:
                self.server = HTTPServer(('0.0.0.0', self.port), PhoneCameraHandler)
                self.server.worker_ref = self
                print(f"[SecondCamera] HTTP server listening on port {self.port}")
                break
            except Exception:
                self.port += 1

        self.all_detected_ips = []
        self.local_ip = self._get_local_ip()
        
        # Check if ngrok is running at startup or start programmatically
        prog_ngrok_url = self._start_programmatic_ngrok()
        if prog_ngrok_url:
            self.server_url = prog_ngrok_url
            print(f"[SecondCamera] Programmatic ngrok tunnel started! Public URL: {self.server_url}")
        else:
            ngrok_url = self._get_ngrok_url()
            if ngrok_url:
                self.server_url = ngrok_url
                print(f"[SecondCamera] Detected running manual ngrok tunnel! Public URL: {self.server_url}")
            else:
                self.server_url = f"http://{self.local_ip}:{self.port}/"
                print(f"[SecondCamera] No ngrok detected. Local URL: {self.server_url}")

        self.latest_data = {
            "connected": False,
            "ever_connected": False,
            "phone_detected": False,
            "person_detected": False,
            "boxes": [],
            "server_url": self.server_url,
            "timestamp": 0.0
        }
        self.data_lock = threading.Lock()
        
        # Start server loop in separate thread
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()

        # Generate QR code
        self.qr_image = self._generate_qr_code(self.server_url)

    def _get_local_ip(self):
        ips = []
        try:
            hostname = socket.gethostname()
            for info in socket.getaddrinfo(hostname, None):
                ip = info[4][0]
                if "." in ip and not ip.startswith("127."):
                    if ip not in ips:
                        ips.append(ip)
        except Exception:
            pass
            
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('10.255.255.255', 1))
            fallback_ip = s.getsockname()[0]
            s.close()
            if fallback_ip not in ips and not fallback_ip.startswith("127."):
                ips.append(fallback_ip)
        except Exception:
            pass
            
        self.all_detected_ips = ips if ips else ['127.0.0.1']
        print(f"[SecondCamera] Detected network adapter IPs: {self.all_detected_ips}")
        
        if not ips:
            return "127.0.0.1"
            
        # Prioritize standard home private IPs (192.168.x.x or 10.x.x.x)
        for ip in ips:
            if ip.startswith("192.168.") or ip.startswith("10."):
                return ip
                
        return ips[0]

    def _generate_qr_code(self, url, size=200):
        qr = qrcode.QRCode(version=1, box_size=8, border=1)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img_np = np.array(img.convert('RGB'))
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_cv2 = cv2.resize(img_cv2, (size, size))
        return img_cv2

    def _run_server(self):
        if self.server:
            self.server.serve_forever()

    def update_frame(self, frame):
        with self.frame_lock:
            self.latest_raw_frame = frame
            self.last_frame_time = time.time()
            self.connected = True
            self.ever_connected = True

    def get_latest_data(self):
        with self.data_lock:
            return self.latest_data.copy()

    def get_qr_image(self):
        return self.qr_image

    def get_annotated_frame(self):
        with self.frame_lock:
            if self.annotated_frame is not None:
                return self.annotated_frame.copy()
            elif self.latest_raw_frame is not None:
                return self.latest_raw_frame.copy()
            return None

    def start(self):
        self.running = True
        super().start()

    def stop(self):
        self.running = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("[SecondCamera] HTTP server stopped.")
        
        # Stop programmatic ngrok tunnels if any
        try:
            from pyngrok import ngrok
            ngrok.kill()
            print("[SecondCamera] Stopped pyngrok tunnel.")
        except Exception:
            pass
            
        try:
            import ngrok
            ngrok.disconnect()
            print("[SecondCamera] Stopped official ngrok tunnel.")
        except Exception:
            pass

    def run(self):
        # Initialize YOLO nano detector
        model = YOLO("yolov8n.pt")
        last_predict_time = 0.0
        last_auth_time = 0.0

        while self.running:
            current_time = time.time()

            # Connection health check (timeout after 4.0 seconds)
            if self.connected and (current_time - self.last_frame_time > 4.0):
                self.connected = False
                print("[SecondCamera] Connection lost (timeout).")

            # Inference loop on phone camera frame (1 FPS)
            if self.connected and (current_time - last_predict_time >= 1.0):
                last_predict_time = current_time
                
                with self.frame_lock:
                    frame = self.latest_raw_frame.copy() if self.latest_raw_frame is not None else None

                if frame is not None:
                    # Run DeepFace Auth for Second Cam (every 10s)
                    if self.auth_queue is not None and (current_time - last_auth_time >= 10.0):
                        last_auth_time = current_time
                        try:
                            if self.auth_queue.full():
                                self.auth_queue.get_nowait()
                            self.auth_queue.put_nowait({"source": "secondary", "frame": frame.copy()})
                        except queue.Full:
                            pass

                    # Filter: person, tv, laptop, mouse, remote, keyboard, cell phone
                    results = model(frame, classes=[0, 62, 63, 64, 65, 66, 67], verbose=False)
                    
                    phone_detected = False
                    person_detected = False
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
                                person_detected = False
                            elif cls_id == 67:
                                cls_name = "phone"
                                phone_detected = True
                            elif cls_id == 63:
                                cls_name = "tablet"
                                phone_detected = True
                            elif cls_id in [62, 64, 65, 66]:
                                cls_name = "device"
                            else:
                                continue

                            boxes_data.append({
                                "box": [int(x1), int(y1), int(x2), int(y2)],
                                "class": cls_name,
                                "conf": conf
                            })
                            
                            # Draw bounding boxes on the frame
                            color = (0, 0, 255) if cls_name in ["phone", "person"] else (0, 255, 255)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(frame, f"{cls_name.upper()} {conf:.2f}", (int(x1), int(y1) - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                    self.phone_detected = phone_detected
                    self.person_detected = person_detected
                    self.boxes = boxes_data

                    with self.frame_lock:
                        self.annotated_frame = frame

            # Push decision telemetry at 1-second interval
            if self.active_proctoring:
                event = {
                    "type": "second_cam",
                    "connected": self.connected,
                    "ever_connected": self.ever_connected,
                    "phone_detected": self.phone_detected if self.connected else False,
                    "person_detected": self.person_detected if self.connected else False,
                    "timestamp": current_time
                }
                try:
                    self.decision_queue.put_nowait(event)
                except queue.Full:
                    pass

            # Dynamic ngrok tunnel detection (check every 2 seconds if not already using ngrok)
            if "ngrok-free.app" not in self.server_url and (current_time - self.last_ngrok_check >= 2.0):
                self.last_ngrok_check = current_time
                ngrok_url = self._get_ngrok_url()
                if ngrok_url:
                    self.server_url = ngrok_url
                    self.qr_image = self._generate_qr_code(self.server_url)
                    print(f"\n[SecondCamera] Dynamic ngrok detected! URL updated to: {self.server_url}")

            # Cache latest telemetry data
            with self.data_lock:
                self.latest_data = {
                    "connected": self.connected,
                    "ever_connected": self.ever_connected,
                    "phone_detected": self.phone_detected if self.connected else False,
                    "person_detected": self.person_detected if self.connected else False,
                    "boxes": self.boxes,
                    "server_url": self.server_url,
                    "timestamp": current_time
                }

            time.sleep(0.1)

    def _get_ngrok_url(self):
        import urllib.request
        import json
        try:
            req = urllib.request.Request("http://127.0.0.1:4040/api/tunnels")
            with urllib.request.urlopen(req, timeout=0.5) as response:
                data = json.loads(response.read().decode())
                tunnels = data.get("tunnels", [])
                for tunnel in tunnels:
                    public_url = tunnel.get("public_url", "")
                    if public_url.startswith("https://"):
                        return public_url
        except Exception:
            pass
        return None

    def _start_programmatic_ngrok(self):
        import os
        token = None
        if os.path.exists("ngrok_token.txt"):
            try:
                with open("ngrok_token.txt", "r") as f:
                    token = f.read().strip()
            except Exception:
                pass

        # 1. Try official ngrok python package (ngrok)
        try:
            import ngrok
            if token:
                listener = ngrok.forward(self.port, authtoken=token)
            else:
                listener = ngrok.forward(self.port)
            print(f"[SecondCamera] Started official ngrok tunnel on port {self.port}")
            return listener.url()
        except Exception as e:
            print(f"[SecondCamera] Official ngrok tunnel failed: {e}")
            if "ERR_NGROK_4018" in str(e) or "not authenticated" in str(e).lower():
                print("\n" + "="*70)
                print("[!] NGROK AUTHENTICATION REQUIRED")
                print("To use the second camera over cellular data or a different network:")
                print("1. Get a free authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken")
                print("2. Create a file named 'ngrok_token.txt' in the project root directory.")
                print("3. Paste your authtoken inside the file and save it.")
                print("4. Restart the script, and the public QR code will generate automatically!")
                print("="*70 + "\n")

        # 2. Try pyngrok (the most common pip package for ngrok)
        try:
            from pyngrok import ngrok as pyngrok_client
            if token:
                pyngrok_client.set_auth_token(token)
            tunnels = pyngrok_client.get_tunnels()
            for t in tunnels:
                if t.public_url.startswith("https://"):
                    return t.public_url
            
            tunnel = pyngrok_client.connect(self.port)
            print(f"[SecondCamera] Started pyngrok tunnel on port {self.port}")
            return tunnel.public_url
        except Exception:
            pass

        return None
