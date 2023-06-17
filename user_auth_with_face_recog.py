import os
import cv2
import statistics
import numpy as np

def train_and_write_new_model():
    print("\n\n>>> Please wait.... Training model on user images...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = []
    labels = []

    dataset_path = "images"
    for label, subfolder in enumerate(sorted(os.listdir(dataset_path)), start=1):
        print(">>>> Starting training on images for user folder: '"+ subfolder.upper() +"' processing...")
        subfolder_path = os.path.join(dataset_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        counter = 0
        for image_name in os.listdir(subfolder_path):
            counter += 1
            print(">>>>>>> Training "+ subfolder.upper() +" image No. "+ str(counter))
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
    print(">>> Creating Training model '.yml file' to system storage.")
    recognizer.save('trained_model.yml')
    print(">>> Successfully Created trained model '.yml file' on system storage.")
    print(">>> General Image Training completed successfully.")


def authenticate_user():
    print("\n>>> Authenticating user by face recognition using available camera...\n")
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

            if confidence < 100:
                # recognized_person = label  # You can modify this to retrieve the person's name from a mapping dictionary
                recognized_person = label_name_mapping.get(label, "Unknown")
                cv2.putText(frame, f"Label: {recognized_person}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Perform action if face is recognized with correct label
                if recognized_person != "Unknown":

                    # Replace this with the action you want to perform
                    if min_num_tries <= max_num_tries:
                        min_num_tries += 1
                        final_recognized_user.append(recognized_person.upper())
                else:
                    # Replace this with the action you want to perform
                    print(">>>> Access DENIED! User is not recognized...")

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elif min_num_tries > max_num_tries:
            # Replace this with the action you want to perform
            print(">>>>> \t ACCESS GRANTED for "+ statistics.mode(final_recognized_user) +"... \t <<<<<")
            print(">>>>> \t LOGIN SUCCESSFUL! \t <<<<< \n")
            break

    # Release the camera and close the windows
    camera.release()
    cv2.destroyAllWindows()


print('\n>>> Welcome to our USERs authentication application using face recognition....')
print('>>> You have to proceed by training the Face recognition AI on available images, Do you want to proceed?')

training_process_response = input('>>>>> Enter either Y/N: \t');
training_process_response_lower = training_process_response.lower()

if training_process_response_lower == 'y' or training_process_response_lower == 'yes':
    # Training the model
    train_and_write_new_model()

    print('\n>>> Do you want to proceed User authentication using face recognition right away?')
    authentication_process_response = input('>>>>> Enter either Y/N: \t');
    authentication_process_response_lower = authentication_process_response.lower()
    
    if authentication_process_response_lower == 'y' or authentication_process_response_lower == 'yes':
        # Authenticating the user
        authenticate_user()


print('\n>>> Terminating application for User Authentication by face recognition....')
print('>>> Application Terminated!!!')
