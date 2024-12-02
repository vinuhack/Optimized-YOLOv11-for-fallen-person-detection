# Optimized-YOLOv11-for-fallen-person-detection

A real-time, accurate, and non-intrusive fall detection system built using an optimized version of YOLOv8. This system is designed to analyze video streams for fall events and trigger alerts, providing an efficient solution for monitoring vulnerable individuals

Data preprocessing and Model building are done in this file: 
https://www.kaggle.com/code/vivekvittalbiragoni/falldetect0-1

Modified ultralytics code repo with inclusion CBAM:
https://github.com/vivekbiragoni/ultralytics   

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [How It Works](#how-it-works)

## Introduction

Falls are a leading cause of injuries among the elderly and other vulnerable groups. Existing detection methods are often intrusive and ineffective. This project provides a **real-time, non-intrusive fall detection system** that uses **enhanced YOLOv8** for high accuracy and reliability. It also integrates with Telegram to send instant alerts.

---

## Features

- **Real-Time Detection**: Detects falls in real-time from video streams.
- **Non-Intrusive**: Works passively without interfering with the user's daily activities.
- **High Accuracy**: Leveraging YOLOv8 with custom optimizations.
- **Alerts System**: Sends instant notifications via Telegram.
- **Customizable Confidence Threshold**: Control sensitivity by adjusting detection thresholds.
- **User-Friendly Interface**: Powered by Gradio for easy video uploads and annotated results.

---

## Technologies Used

- **YOLOv8**: For object detection and classification.
- **Gradio**: To create a user-friendly web interface.
- **OpenCV**: For video processing and visualization.
- **Python**: Core language for implementation.
- **Telegram Bot API**: For real-time notifications.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- A GPU with CUDA support (optional for faster inference)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
2. Install required packages:
    ```bash
   pip install -r requirements.txt
3. Configure Telegram Bot:
   Replace the placeholders in the script with your bot token and chat ID.

## Usage
### Running the Application
1. Start the Gradio interface:
   ```bash
   Copy code
   python app.py
2. Upload a video through the Gradio UI.
3. Annotated video results will be displayed in the interface, and Telegram alerts will be sent if falls are detected.

## How It Works
1. **Input Video**: The user uploads a video of an environment to be monitored.
2. **YOLOv8 Inference**: Each frame is analyzed for falls using a pre-trained YOLOv8 model.
3. **Threshold Filtering**: Detections below a set confidence threshold are discarded to reduce false positives.
4. **Alerts**: If a fall is detected, an alert is sent via Telegram.
5. **Output Video**: The system generates an annotated video showing detected falls.
