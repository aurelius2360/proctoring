import sys
import argparse

def check_dependencies():
    print("=" * 50)
    print("Checking Proctoring Pipeline Dependencies...")
    print("=" * 50)
    
    dependencies = {
        "cv2": "opencv-python",
        "mediapipe": "mediapipe",
        "ultralytics": "ultralytics",
        "deepface": "deepface",
        "torch": "torch",
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
    parser = argparse.ArgumentParser(description="Proctoring Tracking Pipeline CLI and Dependency Checker")
    parser.add_argument("--check-only", action="store_true", help="Only verify environment dependencies without running")
    parser.add_argument("--cam", type=int, default=0, help="Webcam capture index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file instead of live camera")
    args = parser.parse_args()

    deps_ok = check_dependencies()
    
    if args.check_only:
        sys.exit(0 if deps_ok else 1)

    if not deps_ok:
        print("\nAborting launch due to missing dependencies. Run with --check-only to just check.")
        sys.exit(1)

    # Import orchestrator and start
    from main_orchestrator import MainOrchestrator
    
    print(f"\nStarting Proctoring System...")
    print("- Press 'q' in the dashboard window to exit.")
    print("- The session will auto-terminate if proctoring infractions are breached for 5 seconds.")
    print("-" * 50)

    orchestrator = MainOrchestrator(camera_index=args.cam, video_path=args.video)
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
