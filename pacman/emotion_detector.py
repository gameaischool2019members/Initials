"""Emotion Detection functionality."""


from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np


# parameters for loading data and images
DETECTION_MODEL_PATH = '../FaceEmotion_ID/haarcascade_files/haarcascade_frontalface_default.xml'
EMOTION_MODEL_PATH = '../FaceEmotion_ID/models/_mini_XCEPTION.106-0.65.hdf5'
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

class EmotionDetector:
    """Class for detecting emotions from webcam."""


    def __init__(self, debug=False):
        """Initialize the detector.

        If debug is set to true then will show the camera image and predictions graphically.
        """
        # hyper-parameters for bounding boxes shape
        # loading models
        self.face_detection = cv2.CascadeClassifier(DETECTION_MODEL_PATH)
        self.emotion_classifier = load_model(EMOTION_MODEL_PATH, compile=False)

        self.debug = debug

        self.camera = cv2.VideoCapture(0)


    def predict(self):
        """Make prediction from current camera image and return vector of probabilities.

        Probabilities are aligned with EMOTIONS.
        """
        frame = self.camera.read()[1]

        # reading the frame
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)

        canvas = np.zeros((250, 300, 3), dtype="uint8")

        preds = np.zeros(len(EMOTIONS))

        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]

            (fX, fY, fW, fH) = faces
            # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
            # the ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = self.emotion_classifier.predict(roi)[0]
            label = EMOTIONS[preds.argmax()]

        if self.debug:
            frame_clone = frame.copy()

            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                              (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 2)
                cv2.putText(frame_clone, label, (fX, fY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frame_clone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)

            cv2.imshow('your_face', frame_clone)
            cv2.imshow("Probabilities", canvas)

        return preds


    def close(self):
        """Clean up."""
        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # simple test

    detector = EmotionDetector(False)


    while True:
        predictions = detector.predict()
        print(EMOTIONS[predictions.argmax()], list(zip(EMOTIONS, predictions)))
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    detector.close()