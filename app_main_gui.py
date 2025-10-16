# Importing Libraries

import numpy as np
import cv2
import os, sys
import time
import operator

from string import ascii_uppercase

import tkinter as tk
from PIL import Image, ImageTk

# Note: The hunspell and enchant libraries require specific C/C++ build tools 
# on Windows. Make sure they are installed correctly using 'pip install cyhunspell pyenchant'
from hunspell import Hunspell
import enchant

from keras.models import model_from_json

# Setting environment variable (may not be strictly necessary for modern TensorFlow versions)
os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

# Application Class
class Application:

    def __init__(self):

        # Initializing Hunspell for autocorrect/suggestions
        # Requires the 'en_US' dictionary files to be installed and accessible to hunspell
        try:
            self.hs = Hunspell('en_US')
        except Exception as e:
            print(f"Error loading Hunspell: {e}. Autocorrect features may be disabled.")
            self.hs = None

        # Initialize video capture from webcam (0)
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        
        # --- FIX: Using raw strings (r"...") to fix SyntaxWarning: invalid escape sequence '\m' ---
        
        # Load Main Model (Layer 1)
        try:
            self.json_file = open(r"Models\model_new.json", "r")
            self.model_json = self.json_file.read()
            self.json_file.close()
            self.loaded_model = model_from_json(self.model_json)
            self.loaded_model.load_weights(r"Models\model_new.h5")
        except Exception as e:
            print(f"Error loading Main Model: {e}. Ensure 'Models\model_new.json' and 'model_new.h5' exist.")
            self.loaded_model = None

        # Load Sub-Classifier DRU (Layer 2)
        try:
            self.json_file_dru = open(r"Models\model-bw_dru.json" , "r")
            self.model_json_dru = self.json_file_dru.read()
            self.json_file_dru.close()
            self.loaded_model_dru = model_from_json(self.model_json_dru)
            self.loaded_model_dru.load_weights(r"Models\model-bw_dru.h5")
        except Exception as e:
            print(f"Error loading DRU Model: {e}. Sub-classification Layer 2 may be inaccurate.")
            self.loaded_model_dru = None

        # Load Sub-Classifier TKDI (Layer 2)
        try:
            self.json_file_tkdi = open(r"Models\model-bw_tkdi.json" , "r")
            self.model_json_tkdi = self.json_file_tkdi.read()
            self.json_file_tkdi.close()
            self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
            self.loaded_model_tkdi.load_weights(r"Models\model-bw_tkdi.h5")
        except Exception as e:
            print(f"Error loading TKDI Model: {e}. Sub-classification Layer 2 may be inaccurate.")
            self.loaded_model_tkdi = None

        # Load Sub-Classifier SMN (Layer 2)
        try:
            self.json_file_smn = open(r"Models\model-bw_smn.json" , "r")
            self.model_json_smn = self.json_file_smn.read()
            self.json_file_smn.close()
            self.loaded_model_smn = model_from_json(self.model_json_smn)
            self.loaded_model_smn.load_weights(r"Models\model-bw_smn.h5")
        except Exception as e:
            print(f"Error loading SMN Model: {e}. Sub-classification Layer 2 may be inaccurate.")
            self.loaded_model_smn = None

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0

        for i in ascii_uppercase:
          self.ct[i] = 0
        
        print("Loaded model attempts finished.")

        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x900")

        # GUI elements (Tkinter setup)
        self.panel = tk.Label(self.root)
        self.panel.place(x = 100, y = 10, width = 580, height = 580)
        
        self.panel2 = tk.Label(self.root) # Processed image panel
        self.panel2.place(x = 400, y = 65, width = 275, height = 275)

        self.T = tk.Label(self.root)
        self.T.place(x = 60, y = 5)
        self.T.config(text = "Sign Language To Text Conversion", font = ("Courier", 30, "bold"))

        self.panel3 = tk.Label(self.root) # Current Symbol
        self.panel3.place(x = 500, y = 540)

        self.T1 = tk.Label(self.root)
        self.T1.place(x = 10, y = 540)
        self.T1.config(text = "Character :", font = ("Courier", 30, "bold"))

        self.panel4 = tk.Label(self.root) # Word
        self.panel4.place(x = 220, y = 595)

        self.T2 = tk.Label(self.root)
        self.T2.place(x = 10,y = 595)
        self.T2.config(text = "Word :", font = ("Courier", 30, "bold"))

        self.panel5 = tk.Label(self.root) # Sentence
        self.panel5.place(x = 350, y = 645)

        self.T3 = tk.Label(self.root)
        self.T3.place(x = 10, y = 645)
        self.T3.config(text = "Sentence :",font = ("Courier", 30, "bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x = 250, y = 690)
        self.T4.config(text = "Suggestions :", fg = "red", font = ("Courier", 30, "bold"))

        # Suggestion buttons
        self.bt1 = tk.Button(self.root, command = self.action1, height = 0, width = 0)
        self.bt1.place(x = 26, y = 745)

        self.bt2 = tk.Button(self.root, command = self.action2, height = 0, width = 0)
        self.bt2.place(x = 325, y = 745)

        self.bt3 = tk.Button(self.root, command = self.action3, height = 0, width = 0)
        self.bt3.place(x = 625, y = 745)


        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"
        self.photo = "Empty"
        
        # Start the main loop
        self.video_loop()


    def video_loop(self):
        """Main loop for capturing video, preprocessing, predicting, and updating GUI."""
        ok, frame = self.vs.read()

        if ok:
            cv2image = cv2.flip(frame, 1)

            # Define Region of Interest (ROI) coordinates (right half of the frame)
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            # Draw ROI rectangle on the live feed
            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0) ,1)
            
            # Convert frame for Tkinter display (Main Panel)
            cv2image_for_panel = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image_for_panel)
            imgtk = ImageTk.PhotoImage(image = self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image = imgtk)

            # Extract ROI for processing
            roi_image = cv2image[y1 : y2, x1 : x2]

            # Image Processing Pipeline
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            # Adaptive thresholding for hand extraction
            th3 = cv2.adaptiveThreshold(blur, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            # Final binary thresholding (using OTSU to find optimal threshold)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Prediction
            if self.loaded_model:
                self.predict(res)

            # Convert processed image for Tkinter display (Secondary Panel)
            self.current_image2 = Image.fromarray(res)
            imgtk_processed = ImageTk.PhotoImage(image = self.current_image2)
            self.panel2.imgtk = imgtk_processed
            self.panel2.config(image = imgtk_processed)

            # Update text labels
            self.panel3.config(text = self.current_symbol, font = ("Courier", 30))
            self.panel4.config(text = self.word, font = ("Courier", 30))
            self.panel5.config(text = self.str,font = ("Courier", 30))

            # Update suggestions
            predicts = []
            if self.hs: # Only run if hunspell was loaded successfully
                predicts = self.hs.suggest(self.word.strip())
            
            # Configure buttons based on suggestions
            self.bt1.config(text = predicts[0], font = ("Courier", 20)) if len(predicts) > 0 else self.bt1.config(text = "")
            self.bt2.config(text = predicts[1], font = ("Courier", 20)) if len(predicts) > 1 else self.bt2.config(text = "")
            self.bt3.config(text = predicts[2], font = ("Courier", 20)) if len(predicts) > 2 else self.bt3.config(text = "")

        # Call the video_loop again after 5ms for continuous video feed
        self.root.after(5, self.video_loop)

    def predict(self, test_image):
        """Passes the processed image to the CNN models for prediction."""
        if not self.loaded_model:
            self.current_symbol = "MODEL_MISSING"
            return

        # Resize for 128x128 input (expected by the trained models)
        test_image = cv2.resize(test_image, (128, 128))
        # Reshape to (1, 128, 128, 1) for Keras model input
        input_tensor = test_image.reshape(1, 128, 128, 1)

        # Layer 1 Prediction (27 classes: A-Z + blank)
        result = self.loaded_model.predict(input_tensor, verbose=0) 

        # Layer 2 Sub-Classifiers
        result_dru = self.loaded_model_dru.predict(input_tensor, verbose=0) if self.loaded_model_dru else None
        result_tkdi = self.loaded_model_tkdi.predict(input_tensor, verbose=0) if self.loaded_model_tkdi else None
        result_smn = self.loaded_model_smn.predict(input_tensor, verbose=0) if self.loaded_model_smn else None

        prediction = {}
        prediction['blank'] = result[0][0]
        inde = 1

        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1

        # LAYER 1: Get the top overall prediction
        prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)
        self.current_symbol = prediction[0][0]

        # LAYER 2: Refine prediction for similar-looking signs
        
        # D, R, U refinement
        if result_dru is not None and (self.current_symbol == 'D' or self.current_symbol == 'R' or self.current_symbol == 'U'):
            sub_prediction = {}
            sub_prediction['D'] = result_dru[0][0]
            sub_prediction['R'] = result_dru[0][1]
            sub_prediction['U'] = result_dru[0][2]
            sub_prediction = sorted(sub_prediction.items(), key = operator.itemgetter(1), reverse = True)
            self.current_symbol = sub_prediction[0][0]

        # T, K, D, I refinement
        if result_tkdi is not None and (self.current_symbol == 'D' or self.current_symbol == 'I' or self.current_symbol == 'K' or self.current_symbol == 'T'):
            sub_prediction = {}
            sub_prediction['D'] = result_tkdi[0][0]
            sub_prediction['I'] = result_tkdi[0][1]
            sub_prediction['K'] = result_tkdi[0][2]
            sub_prediction['T'] = result_tkdi[0][3]
            sub_prediction = sorted(sub_prediction.items(), key = operator.itemgetter(1), reverse = True)
            self.current_symbol = sub_prediction[0][0]

        # S, M, N refinement (Note: The original code had specific logic here)
        if result_smn is not None and (self.current_symbol == 'M' or self.current_symbol == 'N' or self.current_symbol == 'S'):
            sub_prediction1 = {}
            sub_prediction1['M'] = result_smn[0][0]
            sub_prediction1['N'] = result_smn[0][1]
            sub_prediction1['S'] = result_smn[0][2]

            sub_prediction1 = sorted(sub_prediction1.items(), key = operator.itemgetter(1), reverse = True)

            # The original logic prioritizes the Layer 2 prediction only if it's 'S' 
            # or if the Layer 1 result was not strong enough. I'll stick close to the original for now.
            if(sub_prediction1[0][0] == 'S'):
                self.current_symbol = sub_prediction1[0][0]
            else:
                # If Layer 2 predicts M or N, the code falls back to the Layer 1 prediction.
                # Since Layer 1 prediction is the sorted 'prediction[0][0]' this seems redundant,
                # but I will keep it as intended by the original author.
                self.current_symbol = prediction[0][0]
        
        # Frame Count Logic (Stabilizing the prediction)
        if(self.current_symbol == 'blank'):
            for i in ascii_uppercase:
                self.ct[i] = 0

        self.ct[self.current_symbol] += 1

        if(self.ct[self.current_symbol] > 60): # 60 consecutive frames
            
            # Check for close competitors (original code logic)
            # This logic prevents a lock if a strong competitor is close in count.
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue

                tmp = self.ct[self.current_symbol] - self.ct[i]

                if tmp < 0:
                    tmp *= -1

                if tmp <= 20: # If difference is small (less than 20 frames)
                    self.ct['blank'] = 0
                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return # Exit prediction function, do not lock character

            # If no close competitor, proceed to word building/reset counters
            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0

            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0 and self.word.strip():
                        self.str += " "
                    
                    if self.word.strip():
                        self.str += self.word.strip()

                    self.word = ""
            else:
                if(len(self.str) > 50): # Longer sentence size limit
                    self.str = self.str[-50:] # Keep the last 50 chars (simple way to limit display)

                self.blank_flag = 0
                self.word += self.current_symbol

    # --- Button actions for selecting spelling suggestions ---

    def action1(self):
        """Apply first suggestion."""
        predicts = self.hs.suggest(self.word.strip()) if self.hs else []
        if(len(predicts) > 0):
            self.word = ""
            if len(self.str) > 0 and self.str[-1] != " ": self.str += " "
            self.str += predicts[0]

    def action2(self):
        """Apply second suggestion."""
        predicts = self.hs.suggest(self.word.strip()) if self.hs else []
        if(len(predicts) > 1):
            self.word = ""
            if len(self.str) > 0 and self.str[-1] != " ": self.str += " "
            self.str += predicts[1]

    def action3(self):
        """Apply third suggestion."""
        predicts = self.hs.suggest(self.word.strip()) if self.hs else []
        if(len(predicts) > 2):
            self.word = ""
            if len(self.str) > 0 and self.str[-1] != " ": self.str += " "
            self.str += predicts[2]
            
    # The original application included action4 and action5 in the code but not the GUI, 
    # so I've removed the functions from the class for cleanliness.
            
    def destructor(self):
        """Clean up resources upon closing the application."""
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()
    
print("Starting Application...")

(Application()).root.mainloop()
