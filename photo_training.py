import cv2
import numpy as np
import os

pathFace = './data/test/good3/face_1/'         #данные анфас
pathProfile = './data/test/good3/profile_1/'   #данные профиль

recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face = []
    ids = []
    for imagePath in image_paths:
        img = cv2.imread(imagePath)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        face.append(img)
        id = int(os.path.split(imagePath)[1].split(".")[0])
        ids.append(id)
    return face, ids

# данные анфас
face, ids = get_images_and_labels(pathFace) # чтение тренировочного набора из папки
recognizer.train(face, np.array(ids))
recognizer.write('./models/face_Alex_v1.1.yml')

# данные профиль
face, ids = get_images_and_labels(pathProfile)
recognizer.train(face, np.array(ids))
recognizer.write('./models/profile_Alex_v1.1.yml')
