import cv2
from src.target_detector import CircleInQuadrilateralDetector, annotate

def run_test():
    detector = CircleInQuadrilateralDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Webcam not found, using demo scene...")
        from main import make_demo
        frame = make_demo()
        is_demo = True
    else:
        is_demo = False

    window_name = "Drone Vision"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"Starting native UI test. Press 'q' or 'Esc' to exit.")

    while True:
        if not is_demo:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
        
        # We can simulate some telemetry
        telemetry = [
            "STATE: SCANNING",
            "AGL: 12.50 m",
            "GPS: 50.000000, 14.000000"
        ]

        result = detector.detect(frame)
        output = annotate(frame, result, telemetry_lines=telemetry)
        
        cv2.imshow(window_name, output)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    if not is_demo:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_test()
