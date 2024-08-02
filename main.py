
import time
import cv2

video1 = './data/training/VID-20240710-WA0034.mp4'  # me
video2 = './data/training/VID-20240710-WA0035.mp4'  # me
video3 = './data/training/VID-20240710-WA0036.mp4'  # me
video4 = './data/training/VID-20240710-WA00-1.mp4'  # no me
video5 = './data/training/VID-20240710-WA00-2.mp4'  # no me

recognizer_face = cv2.face.LBPHFaceRecognizer_create()
recognizer_face.read('./models/face_Alex_v1.1.yml')

recognizer_profile = cv2.face.LBPHFaceRecognizer_create()
recognizer_profile.read('./models/profile_Alex_v1.1.yml')

faceCascadeFront = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faceCascadeProfile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

font = cv2.FONT_HERSHEY_SIMPLEX
names = ['None', 'Alex']
zero_array = (0, 0, 0, 0)


def execute():
    # активация видеопотока с камеры
    # cam = cv2.VideoCapiture(0, cv2.CAP_DSHOW)

    # видеопоток из файла
    #cam = cv2.VideoCapture(video1)
    #cam = cv2.VideoCapture(video2)
    cam = cv2.VideoCapture(video3)
    #cam = cv2.VideoCapture(video4)
    #cam = cv2.VideoCapture(video5)

    cam.set(3, 848)  # размер видеокадра - ширина
    cam.set(4, 480)  # размер видеокадра - высота

    while True:
        start_time = time.time()

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        face = face_recognition_full_face_v2(gray)
        profile = face_recognition_profile_v2(gray)
        we_check_that_the_face_is_recognized_v2(face, profile, img, gray)

        cv2.imshow('video', img)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Elapsed time: ', elapsed_time)

        k = cv2.waitKey(10) & 0xff  # 'ESC' для выхода
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


def face_recognition_full_face_v2(gray):
    return faceCascadeFront.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(10, 10), )


def face_recognition_profile_v2(gray):
    return faceCascadeProfile.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(10, 10), )


def we_check_that_the_face_is_recognized_v2(face, profile, img, gray):

    face_size = len(face)
    profile_size = len(profile)

    if face_size == 0 and profile_size != 0:
        face = []
    if face_size != 0 and profile_size == 0:
        profile = []

    if face_size > profile_size:
        difference = face_size - profile_size
        for n in range(difference):
            profile.append(zero_array)
    elif face_size < profile_size:
        difference = profile_size - face_size
        for n in range(difference):
            face.append(zero_array)

    for ((x1, y1, w1, h1), (x2, y2, w2, h2)) in zip(face, profile):

        if x1 == 0 and y1 == 0 and w1 == 0 and h1 == 0:
            id, confidence = start_rendering_profile(x2, y2, w2, h2, img, gray)
            id_obj_final, confidence_final = we_check_that_the_face_is_recognized(confidence)
            adding_text_and_frame(img,  id_obj_final, confidence_final, x2, y2, w2, h2)
        elif x2 == 0 and y2 == 0 and w2 == 0 and h2 == 0:
            id, confidence = start_rendering_face(x1, y1, w1, h1, img, gray)
            id_obj_final, confidence_final = we_check_that_the_face_is_recognized(confidence)
            adding_text_and_frame(img,  id_obj_final, confidence_final, x1, y1, w1, h1)
        else:
            id1, confidence1 = start_rendering_face(x1, y1, w1, h1, img, gray)
            id2, confidence2 = start_rendering_profile(x2, y2, w2, h2, img, gray)

            if confidence1 > confidence2:
                id_obj_final, confidence_final = we_check_that_the_face_is_recognized(confidence1)
                adding_text_and_frame(img, id_obj_final, confidence_final, x1, y1, w1, h1)
            else:
                id_obj_final, confidence_final = we_check_that_the_face_is_recognized(confidence2)
                adding_text_and_frame(img, id_obj_final, confidence_final, x2, y2, w2, h2)


def start_rendering_face(x, y, w, h, img, gray):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return recognizer_face.predict(gray[y:y + h, x:x + w])


def start_rendering_profile(x, y, w, h, img, gray):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return recognizer_profile.predict(gray[y:y + h, x:x + w])


def we_check_that_the_face_is_recognized(confidence):
    # проверяем что лицо распознано
    if confidence < 100:
        id_obj = names[1]
        confidence = "  {0}%".format(round(100 - confidence))
    else:
        id_obj = names[0]
        confidence = "  {0}%".format(round(100 - confidence))
    return id_obj, confidence


def adding_text_and_frame(img, id_obj, confidence, x, y, w, h):
    cv2.putText(img, str(id_obj), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
    cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)


#запустить выполнение кода
execute()

