"""Emotion Detection functionality."""


from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
from multiprocessing.pool import ThreadPool

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

        # the following appears needed for the multithreading
        # stackoverflow magic from
        # https://stackoverflow.com/questions/51127344/tensor-is-not-an-element-of-this-graph-deploying-keras-model
        self.emotion_classifier._make_predict_function()

        self.debug = debug

        self.camera = cv2.VideoCapture(0)

        self.pool = ThreadPool(processes=1)
        self.async_result = None
        self.current_prediction = np.ones(len(EMOTIONS)) / len(EMOTIONS) # default to uniform


    def predict(self):
        """Make prediction from current camera image and return vector of probabilities.

        Probabilities are aligned with EMOTIONS.
        """

        def draw_stuff(predictions, frame, face):
            """Draw info to screen for debug purposes."""

            if face is not None:
                face_x, face_y, face_width, face_height = face

            canvas = np.zeros((250, 300, 3), dtype="uint8")
            frame_clone = frame.copy()


            label = EMOTIONS[predictions.argmax()]

            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, predictions)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                              (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 2)
    
            if face is not None:
                cv2.putText(frame_clone, label, (face_x, face_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frame_clone, (face_x, face_y), (face_x + face_width, face_y + face_height),
                              (0, 0, 255), 2)
    
            cv2.imshow('Webcam', frame_clone)
            cv2.imshow("Probabilities", canvas) 


        def predict_helper():
            """Actually predict.

            Runs in own thread.
            """

            frame = self.camera.read()[1]

            # reading the frame
            frame = imutils.resize(frame, width=400)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                         flags=cv2.CASCADE_SCALE_IMAGE)

            predictions = np.ones(len(EMOTIONS)) / len(EMOTIONS)
            face = None

            if len(faces) > 0:
                face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]

                (face_x, face_y, face_width, face_height) = face
                # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels,
                # and then prepare the ROI for classification via the CNN
                region_of_interest = gray[face_y:face_y + face_height, face_x:face_x + face_width]
                region_of_interest = cv2.resize(region_of_interest, (48, 48))
                region_of_interest = region_of_interest.astype("float") / 255.0
                region_of_interest = img_to_array(region_of_interest)
                region_of_interest = np.expand_dims(region_of_interest, axis=0)

                predictions = self.emotion_classifier.predict(region_of_interest)[0]

            return predictions, frame, face

        if self.async_result is None:
            self.async_result = self.pool.apply_async(predict_helper, ())

        elif self.async_result.ready():
            self.current_prediction, frame, face = self.async_result.get()
            draw_stuff(self.current_prediction, frame, face)
            self.async_result = self.pool.apply_async(predict_helper)

        return self.current_prediction



    def close(self):
        """Clean up."""
        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # simple test

    detector = EmotionDetector(True)


    while True:
        predictions = detector.predict()
        print(EMOTIONS[predictions.argmax()], list(zip(EMOTIONS, predictions)))
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        detector.async_result.wait()

    detector.close()