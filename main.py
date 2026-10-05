import cv2
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# Create output folder
os.makedirs("output", exist_ok=True)

# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Webcam could not be opened.")
    exit()

print("================================")
print("     MOTION DETECTION SYSTEM")
print("================================")
print("Camera Status : ON")
print("Press Q or ESC in the CAMERA WINDOW to stop.")

# Read first frame
ret, previous_frame = camera.read()

if not ret:
    print("ERROR: Could not read camera.")
    camera.release()
    exit()

previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
previous_gray = cv2.GaussianBlur(previous_gray, (21, 21), 0)

motion_count = 0
no_motion_count = 0
motion_records = []

try:

    while True:

        ret, frame = camera.read()

        if not ret:
            print("ERROR: Could not read frame.")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Compare frames
        difference = cv2.absdiff(previous_gray, gray)

        threshold = cv2.threshold(
            difference,
            25,
            255,
            cv2.THRESH_BINARY
        )[1]

        threshold = cv2.dilate(
            threshold,
            None,
            iterations=2
        )

        # Find moving objects
        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = False

        for contour in contours:

            if cv2.contourArea(contour) < 1000:
                continue

            motion_detected = True

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # Date and time
        now = datetime.now()

        date = now.strftime("%d-%m-%Y")
        time = now.strftime("%H:%M:%S")

        # Motion status
        if motion_detected:
            status = "MOTION DETECTED"
            motion_count += 1
            text_color = (0, 255, 0)
        else:
            status = "NO MOTION"
            no_motion_count += 1
            text_color = (255, 255, 255)

        # Store data
        motion_records.append({
            "date": date,
            "time": time,
            "motion_status": status
        })

        # Display date
        cv2.putText(
            frame,
            "Date: " + date,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Display time
        cv2.putText(
            frame,
            "Time: " + time,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Display motion status
        cv2.putText(
            frame,
            "Motion: " + status,
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2
        )

        # Display instructions
        cv2.putText(
            frame,
            "Press Q or ESC to exit",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Show camera
        cv2.imshow("Motion Detection System", frame)

        # Update previous frame
        previous_gray = gray

        # IMPORTANT
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == 27:
            print("Stopping camera...")
            break

except KeyboardInterrupt:
    print("\nCamera stopped using Ctrl+C.")

finally:

    # Release camera
    camera.release()
    cv2.destroyAllWindows()

    # Save JSON
    data = {
        "total_motion_frames": motion_count,
        "total_no_motion_frames": no_motion_count,
        "total_frames": motion_count + no_motion_count,
        "records": motion_records
    }

    with open("output/motion_data.json", "w") as file:
        json.dump(data, file, indent=4)

    print("\nJSON file saved:")
    print("output/motion_data.json")

    # Create Pandas DataFrame
    df = pd.DataFrame(motion_records)

    if not df.empty:

        print("\n================================")
        print("       MOTION STATISTICS")
        print("================================")

        print("Motion Detected :", motion_count)
        print("No Motion       :", no_motion_count)
        print("Total Frames    :", len(df))

        # Generate graph
        labels = [
            "Motion Detected",
            "No Motion"
        ]

        values = [
            motion_count,
            no_motion_count
        ]

        plt.figure()

        plt.bar(labels, values)

        plt.title("Motion Detection Statistics")
        plt.xlabel("Motion Status")
        plt.ylabel("Number of Frames")

        plt.tight_layout()

        plt.savefig(
            "output/motion_graph.png"
        )

        print("\nGraph saved:")
        print("output/motion_graph.png")

        plt.show()

    else:
        print("No motion data recorded.")

    print("\n================================")
    print("          PROGRAM ENDED")
    print("================================")