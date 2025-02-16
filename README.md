# AI-Face-Recognition-with-OpenCV-Deep-Learning
This project is an AI-powered face recognition system using deep learning and computer vision. It detects and recognizes faces in images or video streams, analyzing facial features for accurate identification. Built using OpenCV, dlib, and face-recognition libraries, this Jupyter Notebook provides an easy-to-use face recognition model.
# code
import cv2
import numpy as np

# Load the image
image_path = r"C: ( your path) after downloading the image write your path here.
image = cv2.imread(image_path)

# Check if the image loaded correctly
if image is None:
    print("Error: Image not found!")
    exit()

# Load a pre-trained face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert image to grayscale for better face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# List of names with picture numbers
names = {
    1: "John F. Kennedy",
    2: "Albert Einstein",
    3: "Pope John Paul II",
    4: "Liza Minnelli",
    5: "George W. Bush",
    6: "Elvis Presley",
    7: "Barbra Streisand",
    8: "Martin Luther King Jr.",
    9: "Bill Clinton",
    10: "Sammy Davis Jr.",
    11: "Princess Diana",
    12: "Bill Gates",
    13:"Ronald Reagan",
    14: "Muhammad Ali",
    15: "Lucille Ball",
    16: "Condoleezza Rice",
    17: "Winston Churchill",
    18: "Oprah Winfrey",
    19: "Queen Elizabeth II",
    20: "Humphrey Bogart"
}

# Sort faces by position (top-to-bottom, left-to-right)
faces = sorted(faces, key=lambda x: (x[1]//100, x[0]))

# Draw rectangles and labels
for i, (x, y, w, h) in enumerate(faces):
    index = i + 1  # 1-based index
    if index in names:
        name = f"{index}. {names[index]}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Show the result
cv2.imshow("Labeled Faces", image)
print("Press any key to close the window.")
cv2.waitKey(0)
cv2.destroyAllWindows()                                                                   
          
                                                                      
