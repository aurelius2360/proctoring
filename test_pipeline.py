import sys
import argparse

def check_dependencies():
    print("=" * 50)
    print("Checking Proctoring Pipeline (Dual Camera & LSTM) Dependencies...")
    print("=" * 50)
    
    dependencies = {
        "cv2": "opencv-python",
        "mediapipe": "mediapipe",
        "ultralytics": "ultralytics",
        "deepface": "deepface",
        "torch": "torch",
        "qrcode": "qrcode",
        "webrtcvad": "webrtcvad-wheels",
        "pyaudio": "pyaudio",
        "numpy": "numpy"
    }
    
    missing = []
    for module_name, pip_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"  [OK] {module_name:<12} (pip: {pip_name}) is installed.")
        except ImportError as e:
            print(f"  [FAIL] {module_name:<12} (pip: {pip_name}) is MISSING.")
            missing.append(pip_name)
            
    print("=" * 50)
    if missing:
        print("\n[!] WARNING: Some dependencies are missing. The pipeline cannot run without them.")
        print("Please install them using the following command:")
        print(f"    pip install -r requirements.txt")
        print("\nAlternatively, install specific missing packages:")
        print(f"    pip install " + " ".join(missing))
        return False
    else:
        print("\n[+] SUCCESS: All dependencies are satisfied! You can safely run the pipeline.")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proctoring Tracking Pipeline CLI (Dual Camera & LSTM) and Dependency Checker")
    parser.add_argument("--check-only", action="store_true", help="Only verify environment dependencies without running")
    parser.add_argument("--cam", type=int, default=0, help="Webcam capture index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file instead of live camera")
    parser.add_argument("--photo", type=str, default=None, help="Path to baseline ID photo for identity verification")
    args = parser.parse_args()

    deps_ok = check_dependencies()
    
    if args.check_only:
        sys.exit(0 if deps_ok else 1)

    if not deps_ok:
        print("\nAborting launch due to missing dependencies. Run with --check-only to just check.")
        sys.exit(1)

    # Prompt user for photo path if not provided via command line
    photo_path = args.photo
    if not photo_path:
        import os
        import tkinter as tk
        from tkinter import filedialog
        print("\nPlease select a baseline ID photo for identity verification...")
        try:
            root = tk.Tk()
            root.withdraw()
            photo_path = filedialog.askopenfilename(
                title="Select Baseline ID Photo",
                filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
            )
        except Exception:
            pass
        if not photo_path:
            photo_path = input("Enter path to baseline ID photo: ").strip()

    import os
    if not photo_path or not os.path.exists(photo_path):
        print(f"\nError: Baseline photo file '{photo_path}' not found or not provided.")
        sys.exit(1)

    # Import orchestrator and start
    from main_orchestrator_2cam import MainOrchestrator
    
    print(f"\nStarting Dual Camera & LSTM-based Proctoring System...")
    print("- Scan the QR code shown on the start screen with your phone camera.")
    print("- Point your phone camera from a side angle to monitor hands & keyboard.")
    print("- Press 'q' in any window to exit.")
    print("- Look at the camera to verify your identity against the uploaded photo before starting the test.")
    print("-" * 50)

    orchestrator = MainOrchestrator(camera_index=args.cam, video_path=args.video, baseline_photo_path=photo_path)
    try:
        orchestrator.start_pipeline()
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt received. Exiting...")
        orchestrator.stop_pipeline()
    except Exception as e:
        import traceback
        print(f"\n[Main] Critical system error: {e}")
        traceback.print_exc()
        orchestrator.stop_pipeline()
