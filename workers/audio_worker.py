import pyaudio
import webrtcvad
import numpy as np
import threading
import queue
import time
import os
from collections import deque

class AudioWorker(threading.Thread):
    def __init__(self, decision_queue):
        super().__init__()
        self.decision_queue = decision_queue
        self.running = False
        self.daemon = True
        
        self.latest_data = {
            "speech_detected": False,
            "db_level": 0.0,
            "timestamp": 0.0
        }
        self.data_lock = threading.Lock()
        
        # Audio recording buffers
        self.audio_ring_buffer = deque(maxlen=166)  # 166 chunks of 30ms = ~5s buffer
        self.whole_test_audio_path = None
        self.whole_test_audio_data = []
        
        # Audio configuration
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        # WebRTC VAD requires 10ms, 20ms, or 30ms chunk durations
        self.CHUNK_DURATION_MS = 30
        self.CHUNK_SIZE = int(self.RATE * self.CHUNK_DURATION_MS / 1000)
        
        # VAD aggressiveness mode: 0 (least aggressive), 1, 2, 3 (most aggressive)
        self.VAD_MODE = 2
        
        # Amplitude decibel ceiling RMS threshold
        # Max RMS of 16-bit integer is ~32767. We set threshold to 500 (~-36dB)
        self.RMS_THRESHOLD = 500

    def get_latest_data(self):
        with self.data_lock:
            return self.latest_data.copy()

    def start(self):
        self.running = True
        super().start()

    def stop(self):
        self.running = False

    def run(self):
        # Initialize VAD
        vad = webrtcvad.Vad(self.VAD_MODE)
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            print("[AudioWorker] PyAudio microphone stream opened successfully.")
        except Exception as e:
            print(f"[AudioWorker] Failed to open microphone stream: {e}")
            self.running = False
            p.terminate()
            return

        while self.running:
            try:
                # Read raw PCM chunk
                data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
            except Exception as e:
                time.sleep(0.01)
                continue

            if not data or len(data) < self.CHUNK_SIZE * 2:  # 16-bit PCM = 2 bytes per sample
                continue

            # Record audio frames
            self.audio_ring_buffer.append(data)
            if self.whole_test_audio_path is not None:
                self.whole_test_audio_data.append(data)

            # Calculate RMS (amplitude level)
            samples = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(samples.astype(np.float64)**2))
            
            # Map decibels logarithmically to [0.0, 100.0] for dashboard display
            if rms > 0:
                db_level = 20 * np.log10(rms / 32767.0)
                visual_db = np.clip((db_level + 60) * (100 / 60), 0.0, 100.0)
            else:
                visual_db = 0.0

            speech_detected = False
            
            # If amplitude breaches the decibel ceiling, validate via VAD layer
            if rms > self.RMS_THRESHOLD:
                try:
                    speech_detected = vad.is_speech(data, self.RATE)
                except Exception as e:
                    print(f"[AudioWorker] VAD validation exception: {e}")
                    speech_detected = False

            # Update cache
            with self.data_lock:
                self.latest_data = {
                    "speech_detected": speech_detected,
                    "db_level": float(visual_db),
                    "timestamp": time.time()
                }

            # Send voice flag indicator to Decision Engine
            event = {
                "type": "audio",
                "speech_detected": speech_detected,
                "db_level": float(visual_db),
                "timestamp": time.time()
            }
            
            try:
                self.decision_queue.put_nowait(event)
            except queue.Full:
                pass

            # Relinquish CPU slice
            time.sleep(0.005)

        # Cleanup stream resources
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        p.terminate()
        
        # Save remaining whole test audio if any
        self.stop_whole_test_recording()
        print("[AudioWorker] PyAudio microphone stream closed.")

    def start_whole_test_recording(self, timestamp_str):
        os.makedirs("session_recordings", exist_ok=True)
        self.whole_test_audio_path = f"session_recordings/whole_test_{timestamp_str}.wav"
        self.whole_test_audio_data = []
        print(f"[AudioWorker] Started whole session audio recording: {self.whole_test_audio_path}")

    def stop_whole_test_recording(self):
        if self.whole_test_audio_path is not None and len(self.whole_test_audio_data) > 0:
            import wave
            try:
                wf = wave.open(self.whole_test_audio_path, 'wb')
                wf.setnchannels(self.CHANNELS)
                # We initialize a temporary PyAudio instance just to query format size if needed, or use hardcoded 2 bytes for paInt16
                wf.setsampwidth(2) # 16-bit PCM = 2 bytes
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(self.whole_test_audio_data))
                wf.close()
                print(f"[AudioWorker] Whole session audio recording saved: {self.whole_test_audio_path}")
            except Exception as e:
                print(f"[AudioWorker] Error saving whole session audio: {e}")
        self.whole_test_audio_path = None
        self.whole_test_audio_data = []

    def save_infraction_audio(self, timestamp_str):
        os.makedirs("infractions", exist_ok=True)
        filepath = f"infractions/violation_{timestamp_str}.wav"
        chunks = list(self.audio_ring_buffer)
        if len(chunks) > 0:
            import wave
            try:
                wf = wave.open(filepath, 'wb')
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(2) # 16-bit PCM = 2 bytes
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(chunks))
                wf.close()
                print(f"[AudioWorker] Infraction audio saved: {filepath}")
            except Exception as e:
                print(f"[AudioWorker] Error saving infraction audio: {e}")
