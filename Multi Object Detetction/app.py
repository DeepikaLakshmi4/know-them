
import streamlit as st
import subprocess
import cv2
from PIL import Image


def run_detection():
    
    # Execute YOLOv5 detection and capture the output
    process = subprocess.Popen(["python", "detect.py", "--weights", "yolov5x.pt", "--source", "captured_image.png"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Read the output line by line
    for line in process.stdout:
        if r"C:\Users\DELL\Desktop\know Them\yolov5\captured_image.png: " in line:
            st.write(line[76:-9])
            #st.write(line)
        #st.write(line)
    #     # Check if the line contains information about detected objects
    #     if "0: " in line:
    #         # Extract objects from the line
    #         objects = line.strip().split(" ")[2:-1]  # Extract objects between resolution and time
    #         detected_objects = []
    #         # Process each object in the list
    #         for obj in objects:
    #             # Split the object by space
    #             obj_parts = obj.split(" ")
    #             # Check if the object contains at least two parts
    #             if len(obj_parts) >= 2:
    #                 # Extract count and object name
    #                 count = obj_parts[0]
    #                 obj_name = " ".join(obj_parts[1:])  # Join remaining parts to handle objects with multiple words
    #                 detected_objects.append(f"{count} {obj_name}")
    #             else:
    #                 # Handle unexpected format
    #                 st.warning(f"Unexpected format: {obj}")
    #         # Combine all detected objects into a single string
    #         detected_objects_str = " ".join(detected_objects)
    #         # Display the formatted detected objects in the Streamlit app
    #         st.write(f"Detected {detected_objects_str}")

# Main function
def main():
    st.title("Object Detection")
    
    img_placeholder = st.image([])
    cam = cv2.VideoCapture(0) 
    
    # Read the input from the camera
    result, frame = cam.read() 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    captured  = st.button("Capture")
    stop_capture = st.button("Stop")
    img_placeholder.image(frame, caption='Camera Feed', use_column_width=True)
    cv2.imshow("Captured image", frame)
    while True:  
        if captured == 1:
            cv2.imwrite("captured_image.png", frame)
            img_placeholder.image("captured_image.png")
            st.write("captured")
            run_detection()
            captured = 0
            img_placeholder.image(frame, caption='Camera Feed', use_column_width=True)
            cv2.imshow("Captured image", frame)
        if stop_capture ==1 :
            cv2.waitKey(0) 
            cv2.destroyWindow("Captured Image")
            st.write("Stopped")
            break

if __name__ == "__main__":
    main()


