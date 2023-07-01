import tkinter as tk
from tkinter import ttk
from tkinter.font import Font
import os
import cv2
import statistics
import numpy as np
import time

def train_and_write_new_model():
    # Destroy all existing widgets in the frame
    for widget in general_frame.winfo_children():
        widget.destroy()

    proceed_training_text = "Please wait.... Training model on user images..."
    text_label = tk.Label(general_frame, text=proceed_training_text, font=header_font)
    text_label.pack()
    window.update()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = []
    labels = []

    dataset_path = "images"
    for label, subfolder in enumerate(sorted(os.listdir(dataset_path)), start=1):
        start_training_text = "Starting training on images for user folder: '"+ subfolder.upper() +"' processing..."
        text_label = tk.Label(general_frame, text=start_training_text, font=header_font)
        text_label.pack()
        window.update()

        subfolder_path = os.path.join(dataset_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        counter = 0
        for image_name in os.listdir(subfolder_path):
            counter += 1
            start_training_counter = "Training "+ subfolder.upper() +" image No. "+ str(counter)
            text_label = tk.Label(general_frame, text=start_training_counter, font=header_font)
            text_label.pack()
            window.update()

            image_path = os.path.join(subfolder_path, image_name)
            if not os.path.isfile(image_path):
                continue

            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply face detection
            faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Assuming only one face per image, extract the face region
            if len(faces_detected) == 1:
                (x, y, w, h) = faces_detected[0]
                face_roi = gray[y:y+h, x:x+w]

                faces.append(face_roi)
                labels.append(label)

    recognizer.train(faces, np.array(labels))

    creating_yml_file = "Creating Training model '.yml file' to system storage."
    text_label = tk.Label(general_frame, text=creating_yml_file, font=header_font)
    text_label.pack()
    window.update()

    recognizer.save('trained_model.yml')

    created_yml_file = "Successfully Created trained model '.yml file' on system storage.\n"
    created_yml_file += "General Image Training completed successfully."
    text_label = tk.Label(general_frame, text=created_yml_file, font=header_font)
    text_label.pack()
    window.update()


# proceed with face authentication
def authenticate_user():
    # Destroy all existing widgets in the frame
    for widget in general_frame.winfo_children():
        widget.destroy()

    authentication_text = "\nAuthenticating user by face recognition using available camera..."
    text_label = tk.Label(general_frame, text=authentication_text, font=header_font)
    text_label.pack()
    window.update()

    final_recognized_user = [];
    min_num_tries = 0;
    max_num_tries = 20;
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trained_model.yml')

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Retrieve the label names from the trained model
    label_name_mapping = {}
    dataset_path = "images"
    for label, subfolder in enumerate(sorted(os.listdir(dataset_path)), start=1):
        label_name_mapping[label] = subfolder

    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces_detected:
            face_roi = gray[y:y+h, x:x+w]

            label, confidence = recognizer.predict(face_roi)

            if min_num_tries <= max_num_tries:
                if confidence < 70:
                    # recognized_person = label  # You can modify this to retrieve the person's name from a mapping dictionary
                    recognized_person = label_name_mapping.get(label, "Unknown")
                    cv2.putText(frame, f"Label: {recognized_person}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Replace this with the action you want to perform
                    if recognized_person != "Unknown":
                        final_recognized_user.append(recognized_person.upper())
                    else:
                        final_recognized_user.append('Unknown')

                else:
                    final_recognized_user.append('Unknown')
            
                min_num_tries += 1

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') :
            authentication_text = "\nQuit Application!..\n"
            text_label = tk.Label(general_frame, text=authentication_text, font=header_font)
            text_label.pack()
            window.update()
            break

        elif min_num_tries > max_num_tries:
            # Replace this with the action you want to perform
            mode_final_recognized_user = statistics.mode(final_recognized_user)
            
            if mode_final_recognized_user=="Unknown":
                authentication_text = "\nAccess DENIED! User is not recognized!\n"
                text_label = tk.Label(general_frame, text=authentication_text, font=header_font)
            else:
                authentication_text = "\nACCESS GRANTED for "+ statistics.mode(final_recognized_user) +"...\n"
                authentication_text += "\nLOGIN SUCCESSFUL!\n"
                text_label = tk.Label(general_frame, text=authentication_text, font=header_font)
            
            text_label.pack()
            window.update()
            break

    # Release the camera and close the windows
    camera.release()
    cv2.destroyAllWindows()

    # Create a frame to hold the buttons
    button_frame = tk.Frame(general_frame)
    button_frame.pack()

    # Create the 'Restart Authentication' button
    restart_button = tk.Button(button_frame, text="Restart Authentication", command=lambda: confirm_button_to_proceed_authentication("Yes"))
    restart_button.pack(side="left")

    # Create the 'Close' button
    no_button = tk.Button(button_frame, text="Close Application", command=lambda: confirm_button_to_proceed_authentication(no_button.cget("text")))
    no_button.pack(side="left")


# clsoinf TKinter window
def close_window():
    window.destroy()


# teminating the process
def terminate_interaction():
    # Destroy all existing widgets in the frame
    for widget in general_frame.winfo_children():
        widget.destroy()

    termination_text = "Terminating application for User Authentication by face recognition..."
    text_label = tk.Label(general_frame, text=termination_text, font=header_font)
    text_label.pack()
    window.update()

    termination_text = "\nApplication Terminated..."
    text_label = tk.Label(general_frame, text=termination_text, font=header_font)
    text_label.pack()
    window.update()

    termination_text = "\nClosing windows in 10 seconds..."
    text_label = tk.Label(general_frame, text=termination_text, font=header_font)
    text_label.pack()
    window.update()

    window.after(10000, close_window)


# Function to handle button clicks on confirming for authentication
def confirm_button_to_proceed_authentication(answer):
    if answer == "Yes":
        # Authenticating the user
        authenticate_user()
    else :
         terminate_interaction()


# Function to handle button clicks on confirming for training
def confirm_button_clicked_to_proceed_training(answer):
    if answer == "Yes":
        #Training the model
        train_and_write_new_model()

        confirm_to_proceed_authentication = "\nDo you want to proceed User authentication using face recognition right away?"
        text_label = tk.Label(general_frame, text=confirm_to_proceed_authentication, font=header_font)
        text_label.pack()
        window.update()

        # Create a frame to hold the buttons
        button_frame = tk.Frame(general_frame)
        button_frame.pack()

        # Create the 'Yes' button
        yes_button = tk.Button(button_frame, text="Yes", command=lambda: confirm_button_to_proceed_authentication(yes_button.cget("text")))
        yes_button.pack(side="left")

        # Create the 'No' button
        no_button = tk.Button(button_frame, text="No", command=lambda: confirm_button_to_proceed_authentication(no_button.cget("text")))
        no_button.pack(side="left")

    elif answer == "No":
        terminate_interaction()


# start user interaction
def start_interaction():
    dynamic_text_label = tk.Label(general_frame, text="Do you want to proceed?", font=header_font)
    dynamic_text_label.pack()

    # Create a frame to hold the buttons
    button_frame = tk.Frame(general_frame)
    button_frame.pack()

    # Create the 'Yes' button
    yes_button = tk.Button(button_frame, text="Yes", command=lambda: confirm_button_clicked_to_proceed_training(yes_button.cget("text")))
    yes_button.pack(side="left")

    # Create the 'No' button
    no_button = tk.Button(button_frame, text="No", command=lambda: confirm_button_clicked_to_proceed_training(no_button.cget("text")))
    no_button.pack(side="left")

    # Start the Tkinter event loop
    window.mainloop()


# initial configurations for TK windows 
window = tk.Tk()
window.title("Final Year Project On A Simple Face Recognition Application")
window.geometry("1400x900")

welcome_text = "\nWelcome, This is a User authentication application using face recognition!"
text_label = tk.Label(window, text=welcome_text, font=Font(size=15, weight="bold"))
text_label.pack()

welcome_text = "\nYou have to proceed by training the Face recognition AI on available images\n\n"
header_font = Font(size=12, weight="bold")
text_label = tk.Label(window, text=welcome_text, font=header_font)
text_label.pack()

# Create a horizontal rule (separator)
separator = ttk.Separator(window, orient="horizontal")
separator.pack(fill="x", pady=10)

# Create a general frame to hold the buttons
general_frame = tk.Frame(window)
general_frame.pack()
start_interaction()