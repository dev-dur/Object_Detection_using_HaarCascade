Real-Time Object Detection with OpenCV DNN (SSD MobileNet v3)

This project demonstrates real-time object detection using OpenCV’s Deep Neural Network (DNN) module with a pre-trained SSD MobileNet v3 model on the COCO dataset.
It supports both video files and webcam streams, and saves the processed video with bounding boxes and class labels.

🚀 Features

Loads a pre-trained SSD MobileNet v3 COCO model

Detects 80+ object categories from the COCO dataset

Supports live webcam feed or video file input

Draws bounding boxes with class labels and confidence scores

Saves the detection results as a video (output_video.avi)

📂 Project Structure
- coco.txt                         # COCO class labels
- frozen_inference_graph.pb        # Pre-trained model weights
- ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt   # Model config
- test1.mp4                        # Example input video
- main.py                          # Main script
- README.md                        # Documentation

🛠️ Requirements
- Python 3.8+
- OpenCV (cv2)
- Matplotlib (for visualization)

Install dependencies:
pip install opencv-python matplotlib

🔧 Key Parameters
- Confidence threshold: thresh = 0.6
- Input size: (320, 320) (as per SSD MobileNet v3 requirements)
- Output format: .avi with MJPEG codec
