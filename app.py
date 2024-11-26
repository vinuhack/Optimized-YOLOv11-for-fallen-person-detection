<<<<<<< HEAD
import gradio as gr
import cv2
from ultralytics import YOLO
import requests
import time

# Telegram Bot Configuration
telegram_bot_token = "7923079722:AAEdnskxzCAp4J0o2YPChPNizAB-xIt6bQg"  # Replace with your bot token
telegram_chat_id = "2089950244"  # Replace with your chat ID

# Function to send Telegram alerts
def send_telegram_notification(message="⚠️ Fall detected! Immediate attention may be required."):
    """
    Sends a Telegram notification.

    Args:
        message (str): The message to send as a notification.
    """
    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message}
    response = requests.post(url, json=payload)
    if response.ok:
        print("Notification sent via Telegram.")
    else:
        print("Failed to send Telegram notification:", response.text)


# Video processing function
def process_video(input_path, output_path="output.mp4", model_path="best.pt", conf_threshold=0.3):
    """
    Perform inference on a video to detect and classify objects using YOLO.

    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the annotated output video.
        model_path (str): Path to the YOLO model weights.
        conf_threshold (float): Confidence threshold for filtering detections.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_path}")
        return None

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for output video

    # Initialize video writer for saving annotated video
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    fall_detected = False  # Prevent multiple alerts for the same fall

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference on the frame
        results = model.predict(source=frame, conf=conf_threshold, imgsz=640, stream=False)

        # Annotate frame with predictions
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = map(int, box)
                label = "Fall" if cls == 0 else "No Fall"  # Adjust class indices as per your dataset
                color = (0, 255, 0) if label == "No Fall" else (0, 0, 255)  # Green for No Fall, Red for Fall

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Send Telegram alert if fall is detected
                if label == "Fall" and not fall_detected:
                    send_telegram_notification()
                    fall_detected = True  # Prevent duplicate notifications for the same fall

        # Write annotated frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    return output_path


# Gradio interface
def gradio_process(input_video):
    """
    Process video uploaded via Gradio interface.

    Args:
        input_video (str): Path to the uploaded input video.
    Returns:
        str: Path to the annotated output video.
    """
    output_path = f"output_{int(time.time())}.mp4"  # Unique file names to avoid overwriting
    result = process_video(input_video, output_path)
    if result:
        return result
    else:
        return "Error processing video"


# Create and launch Gradio interface
gr.Interface(
    fn=gradio_process,
    inputs=gr.Video(label="Upload Video"),  # Corrected input
    outputs=gr.Video(label="Annotated Video"),
    title="AI-Powered Fall Detection",
    description="Upload a video to detect falls. Annotated video will highlight detections and send alerts if a fall is detected."
).launch()
=======
import gradio as gr
import cv2
from ultralytics import YOLO
import requests
import time

# Telegram Bot Configuration
telegram_bot_token = "7923079722:AAEdnskxzCAp4J0o2YPChPNizAB-xIt6bQg"  # Replace with your bot token
telegram_chat_id = "2089950244"  # Replace with your chat ID

# Function to send Telegram alerts
def send_telegram_notification(message="⚠️ Fall detected! Immediate attention may be required."):
    """
    Sends a Telegram notification.

    Args:
        message (str): The message to send as a notification.
    """
    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message}
    response = requests.post(url, json=payload)
    if response.ok:
        print("Notification sent via Telegram.")
    else:
        print("Failed to send Telegram notification:", response.text)


# Video processing function
def process_video(input_path, output_path="output.mp4", model_path="best.pt", conf_threshold=0.3):
    """
    Perform inference on a video to detect and classify objects using YOLO.

    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the annotated output video.
        model_path (str): Path to the YOLO model weights.
        conf_threshold (float): Confidence threshold for filtering detections.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_path}")
        return None

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for output video

    # Initialize video writer for saving annotated video
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    fall_detected = False  # Prevent multiple alerts for the same fall

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference on the frame
        results = model.predict(source=frame, conf=conf_threshold, imgsz=640, stream=False)

        # Annotate frame with predictions
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = map(int, box)
                label = "Fall" if cls == 0 else "No Fall"  # Adjust class indices as per your dataset
                color = (0, 255, 0) if label == "No Fall" else (0, 0, 255)  # Green for No Fall, Red for Fall

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Send Telegram alert if fall is detected
                if label == "Fall" and not fall_detected:
                    send_telegram_notification()
                    fall_detected = True  # Prevent duplicate notifications for the same fall

        # Write annotated frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    return output_path


# Gradio interface
def gradio_process(input_video):
    """
    Process video uploaded via Gradio interface.

    Args:
        input_video (str): Path to the uploaded input video.
    Returns:
        str: Path to the annotated output video.
    """
    output_path = f"output_{int(time.time())}.mp4"  # Unique file names to avoid overwriting
    result = process_video(input_video, output_path)
    if result:
        return result
    else:
        return "Error processing video"


# Create and launch Gradio interface
gr.Interface(
    fn=gradio_process,
    inputs=gr.Video(label="Upload Video"),  # Corrected input
    outputs=gr.Video(label="Annotated Video"),
    title="AI-Powered Fall Detection",
    description="Upload a video to detect falls. Annotated video will highlight detections and send alerts if a fall is detected."
).launch(share=True)
>>>>>>> 3a026e1 (Add application file)
