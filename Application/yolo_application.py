import streamlit as st
import cv2
from ultralytics import YOLO 
import os

# Set the Title 
st.title("YOLO Image and Video Processing")

# Load the YOLO model from the uploaded file



# Give user the option to upload an image or video file
uploaded_file = st.file_uploader("Upload an image or video file", type=["jpg", "jpeg", "png", "mp4"])

# Load YOLO model
try:
  model = YOLO("C:\\Users\\HP\\Desktop\\Object_detection\\Application\\boat_detection_model.pt")
 
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")


# Function to predict and save bounding boxes on a test image using YOLO model
def predict_and_save_image(path_test_boat, output_image_path):
    """
    Predicts and saves the bounding boxes on the given test image using the trained YOLO model.
    
    Parameters:
    path_test_boat (str): Path to the test image file.
    output_image_path (str): Path to save the output image file.
    
    Returns:
    str: The path to the saved output image file.
    """
    results = model.predict(path_test_boat, device='cpu')  # Assuming 'model' is initialized somewhere
    image = cv2.imread(path_test_boat)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Iterate through each prediction result and draw bounding box
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, image)
    
    return output_image_path

# Function to predict and plot bounding boxes on a test video using YOLO model
def predict_and_plot_video(video_path, output_path):
    """
    Predicts and saves the bounding boxes on the given test video using the trained YOLO model.
    
    Parameters:
    video_path (str): Path to the test video file.
    output_path (str): Path to save the output video file.
    
    Returns:
    str: The path to the saved output video file.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb_frame, device='cpu')  # Assuming 'model' is initialized somewhere
        
        # Iterate through each prediction result and draw bounding box
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]  # Get the class name
                label = f"{class_name} {confidence*100:.2f}%"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    return output_path

# Function to process uploaded media file (image or video) and return saved output path
def process_media(input_path, output_path):
    """
    Processes the uploaded media file (image or video) and returns the path to the saved output file.
    
    Parameters:
    input_path (str): Path to the input media file.
    output_path (str): Path to save the output media file.
    
    Returns:
    str: The path to the saved output media file.
    """
    file_extension = os.path.splitext(input_path)[1].lower()
    
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        return predict_and_plot_video(input_path, output_path)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
        return predict_and_save_image(input_path, output_path)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None

# Main logic to handle file upload and process media
if uploaded_file is not None:
    input_path = f"temp/{uploaded_file.name}"
    output_path = f"output/{uploaded_file.name}"
    
    with open(input_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
        
    st.write("Processing Image or Video...")
    result_path = process_media(input_path, output_path)
