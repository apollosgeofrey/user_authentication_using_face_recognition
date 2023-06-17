import cv2
import os
import time
import numpy as np


def readFromTrainedModel():
    print("\n\n>>>> Authenticating User by face recognition......")
    final_recognized_user = '';
    min_num_tries = 0;
    max_num_tries = 20;
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trained_model.yml')  # Load your trained model file

    # Retrieve the label names from the trained model
    label_name_mapping = {}
    dataset_path = "images"
    for label, subfolder in enumerate(sorted(os.listdir(dataset_path)), start=1):
        label_name_mapping[label] = subfolder

    # Initialize camera capture
    camera = cv2.VideoCapture(0)  # Use 0 for default camera, or specify the camera index if you have multiple cameras

    while True:
        # Read a frame from the camera
        ret, frame = camera.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(roi_gray)

            recognized_person = label_name_mapping.get(label, "Unknown")

            # Display the recognized label and confidence level on the frame
            cv2.putText(frame, f"Label: {recognized_person}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence}", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Perform action if face is recognized with correct label
            if recognized_person != "Unknown":
                # Capture image
                # cv2.imwrite('captured_image.jpg', frame)
                
                if min_num_tries <= max_num_tries:
                    min_num_tries += 1
                    final_recognized_user = recognized_person
                else:
                    final_recognized_user = recognized_person
                    
            else:
                # Replace this with the action you want to perform
                print("\n\nAccess Denied! \nUser is not recognized....,\nPlease reposition your face properly.")


        # Display the frame
        cv2.imshow('Face Recognition', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Release the camera and close the windows
            break
            
        elif min_num_tries > max_num_tries:
            # Release the camera and close the windows
            camera.release()
            cv2.destroyAllWindows()

            # Replace this with the action you want to perform
            print("\n\n>>>>\tAccess granted!!!\t<<<<\n>>>>\tLogin user successfully recognized as "+ final_recognized_user.upper() +"\t<<<<\n>>>>\tRedirecting to dashboard successful...\t<<<<")
            break





def trainAndWriteNewModel():
    print("\n\n>>> Please wait.... Starting to training model on users images...")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []

    # Set the path to the labeled dataset folder
    dataset_path = "images"

    print(">>> Processing images.... Please wait.......")

    # Iterate through each subfolder (identity) in the dataset folder
    for label, subfolder in enumerate(os.listdir(dataset_path), start=1):
        print("\n>>>> Starting training on images for user folder: '"+ subfolder +"' processing...")

        subfolder_path = os.path.join(dataset_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        counter = 0
        for image_name in os.listdir(subfolder_path):
            counter += 1
            print(">>>>>>> Hadling training for image No. "+ str(counter) +" on user folder: '"+ subfolder +"' processing...")
            image_path = os.path.join(subfolder_path, image_name)
            if not os.path.isfile(image_path):
                continue

            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces.append(gray)
            labels.append(label)
            print(">>>>>>> Completed training for image No. "+ str(counter) +" on user folder: '"+ subfolder +"' successfully.")
    
        print(">>>> Training images for user folder: '"+ subfolder +"' completed successfully.")

    # Train the face recognizer model
    recognizer.train(faces, np.array(labels))

    # Save the trained model as a YAML file
    print("\n>>> Creating Training model '.yml file' to system storage.")
    recognizer.save('trained_model.yml')
    print(">>> Successfully Created trained model '.yml file' on system storage.")

    print(">>> Training model on available user Images completed successfully.")





trainAndWriteNewModel()
readFromTrainedModel()