# Importing the Libraries Required

import cv2
import numpy as np
import os
import string

# Configuration
mode = 'testingData'
directory = f'dataSet/{mode}/'
minValue = 35 # Note: The original TestingDataCollection used 35, while TrainingDataCollection used 70.

capture = cv2.VideoCapture(0)
interrupt = -1

# Map lowercase key press to uppercase directory names
key_map = {ord(c): c for c in string.ascii_lowercase}
key_map[ord('0')] = '0' # For the blank sign

while True:
    try:
        _, frame = capture.read()

        # Check if frame capture was successful
        if frame is None:
            print("Error: Could not read frame from webcam. Check if the camera is in use or blocked.")
            break

        # Simulating mirror Image
        frame = cv2.flip(frame, 1)

        # Getting count of existing images
        count = {}
        for char in string.ascii_uppercase:
            count[char.lower()] = len(os.listdir(directory+char))
        count['zero'] = len(os.listdir(directory+'0'))

        # Printing the count of each set on the screen
        y_offset = 60
        cv2.putText(frame, "ZERO : " +str(count['zero']), (10, y_offset), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        for i, char in enumerate(string.ascii_uppercase):
            cv2.putText(frame, f"{char.lower()} : {count[char.lower()]}", (10, y_offset + 10 * (i + 1)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)

        # Coordinates of the ROI (Right half of the frame, square)
        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])

        # Drawing the ROI
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
        
        # Extracting the ROI
        roi = frame[y1:y2, x1:x2]

        cv2.imshow("Frame (Press ESC to exit)", frame)
        
        # --- Image Processing Pipeline (for visual feedback/pre-processing demonstration) ---
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 2)
            
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        # Output Image after the Image Processing
        test_image_resized = cv2.resize(test_image, (300,300))
        cv2.imshow("Processed ROI", test_image_resized)

        # Data Collection Logic
        interrupt = cv2.waitKey(10)
        
        if interrupt & 0xFF == 27: 
            # esc key
            break
        
        # Check if a mapped key was pressed for data collection
        if interrupt & 0xFF in key_map:
            char_key = key_map[interrupt & 0xFF]
            dir_name = char_key if char_key == '0' else char_key.upper()
            count_key = char_key if char_key == '0' else char_key.lower()
            
            # Save the RAW ROI image
            file_path = f"{directory}{dir_name}/{count[count_key]}.jpg"
            cv2.imwrite(file_path, roi)
            print(f"Saved: {file_path}")

    except cv2.error as e:
        print(f"OpenCV Error encountered: {e}. Stopping frame capture.")
        break
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        break
        
capture.release()
cv2.destroyAllWindows()
