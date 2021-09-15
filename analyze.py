import cv2
from tensorflow.keras.models import load_model
import numpy as np

# define and load the models
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mask_model = load_model("face_mask_detection.h5")
#age_model = load_model('weights.28-3.73.hdf5')
not_found = cv2.imread('notfound.jpg')


class GenerateVideo(object):
    def __init__(self):
        self.video = cv2.VideoCapture(1)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, img = self.video.read()
        try:
            # face detection
            faces = face_detect.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces:
                # face images processing
                face_img = img[y:y + h, x:x + w]
                face_img1 = cv2.resize(face_img, (224, 224))
                face_img1 = face_img1[np.newaxis]
                #face_img2 = cv2.resize(face_img, (64, 64))
                #face_img2 = face_img2[np.newaxis]
                # predict mask / no mask
                mask_pred = mask_model.predict(face_img1)[0][0]

                if mask_pred > 0:  # no mask
                    #result = age_model.predict(face_img2)
                    #pred_gender = result[0]
                   # if pred_gender[0][0] > 0.5:
                   #     gender = 'Female'
                  #  else:
                  #      gender = 'Male'
                  #  ages = np.arange(0, 101).reshape(101, 1)
                   # age_pred = result[1].dot(ages).flatten()
                    mask_pred = 'No Mask'
                    color = (0, 0, 255)
                    text = mask_pred
                else:
                    text = 'MASK'
                    color = (0, 255, 0)
                cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        except IOError:
            img = not_found
        _, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
