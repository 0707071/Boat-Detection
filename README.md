YOLO Image and Video Processing with Streamlit
This Streamlit application demonstrates real-time object detection using the YOLO (You Only Look Once) model for images and videos. It allows users to upload files and visualize the detection results with bounding boxes and confidence scores.

Features
Upload Files: Users can upload image (.jpg, .jpeg, .png) or video (.mp4) files.
Real-time Processing: The application processes uploaded files using a pre-trained YOLO model.
Visual Feedback: Detected objects are annotated with bounding boxes and confidence scores.
Download Results: Users can download processed images or videos with annotations.
Prerequisites
Ensure you have the following installed:

Python 3.7+
Required Python packages (streamlit, cv2, ultralytics)

code:
pip install streamlit opencv-python-headless ultralytics

Usage
Clone the repository:

code:
git clone https://github.com/0707071/Boat-Detection.git
cd Boat-Detection

cd your-repo
Install dependencies:

code:
pip install -r requirements.txt
Run the Streamlit application:

code:
streamlit run streamlit run yolo_application.py

Upload an image or video file using the file uploader.

Wait for the model to process the file and display the results.

Download the processed image or video with annotations if desired.


