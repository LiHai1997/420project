import os
import time
from time import sleep
import cv2
import dlib
import numpy as np
from wide_resnet import WideResNet
from own_sift import *
stored_faces = []


def match_faces(sift, face):
    kp, descriptors = sift.detectAndCompute(face, None)
    # bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    for stored_face in stored_faces:
        kp, descriptors = sift.detectAndCompute(face, None)
        # matches = bf.match(descriptors, stored_face)
        # matches = sorted(matches, key = lambda x:x.distance)
        distance = matching_and_plot(descriptors, stored_face, "L3")
        print(len(distance))

        if len(distance) < 3:
            return False
        print(distance[0], distance[1], distance[2])
        # if (matches[0].distance < 1200 and matches[1].distance < 1400 and matches[2].distance < 1600):
        #     return True

        if (distance[0] < 250 and distance[1] < 300 and distance[2] < 300):
            return True
    stored_faces.append(descriptors)
    return False


def main():
    sift_1 = cv2.xfeatures2d.SIFT_create()
    image_list = os.listdir("visitors/")
    for img in image_list:
        gray_faces = cv2.imread("visitors/"+img, 0)
        kp, descriptors = sift_1.detectAndCompute(gray_faces, None)
        stored_faces.append(descriptors)
    detector = dlib.get_frontal_face_detector()

    margin = 1.0
    # load weights
    img_size = 64
    model = WideResNet(img_size, depth=16, k=8)()
    model.load_weights('model/weights.hdf5')

    screen_capture = False
    screen_text_countdown = 0
    screen_message = ''
    screen_fontsize = 1.5
    screen_text_point = (120, 200)

    # Take video from camera
    capture = cv2.VideoCapture(0)

    while 1:
        ret, frame = capture.read()
        if not capture.isOpened():
            sleep(5)
        blur = cv2.GaussianBlur(frame, (7,7), 0)
        input_img = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(frame)
        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right() + 1
                y2 = d.bottom() + 1
                w = d.width()
                h = d.height()
                xm1 = max(int(x1 - margin * w), 0)
                ym1 = max(int(y1 - margin * h), 0)
                xm2 = min(int(x2 + margin * w), img_w - 1)
                ym2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (92, 91, 241), 2)
                faces[i, :, :, :] = cv2.resize(frame[ym1 : ym2 + 1, xm1 : xm2 + 1, :], (img_size, img_size))

            # predict ages and genders
            results = model.predict(faces)
            genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            # Match faces if requested
            sift = cv2.xfeatures2d.SIFT_create()
            if (screen_capture == True):
                new_num = 0
                old_num = 0
                if not stored_faces:
                    new_num = len(detected)
                    old_num = 0
                    for face in faces:
                        face_temp = np.zeros((64, 64, 3), dtype="uint8")
                        face_temp[:, :, :] = face[:, :, :]
                        gray_face = cv2.cvtColor(face_temp, cv2.COLOR_BGR2GRAY)
                        kp, descriptors = sift.detectAndCompute(gray_face, None)
                        stored_faces.append(descriptors)
                        timestr = time.strftime("%Y%m%d-%H%M%S")
                        cv2.imwrite("visitors/"+timestr+".jpg", gray_face)
                else:
                    for face in faces:
                        face_temp = np.zeros((64, 64, 3), dtype="uint8")
                        face_temp[:, :, :] = face[:, :, :]
                        gray_face = cv2.cvtColor(face_temp, cv2.COLOR_BGR2GRAY)
                        if (match_faces(sift, gray_face)):
                            old_num += 1
                        else:
                            new_num += 1
                            timestr = time.strftime("%Y%m%d-%H%M%S")
                            cv2.imwrite("visitors/"+timestr + ".jpg", gray_face)
                screen_message = "{} Visitor(s), {} New Visitor(s), {} Visited".format(faces.shape[0], new_num, old_num)
                screen_text_point = (20, 20)
                screen_text_countdown = 8
                screen_fontsize = 0.8


            # draw results
            for i, d in enumerate(detected):
                age = int(predicted_ages[i])
                if age < 20:
                    age_period = "under 20"
                elif age < 30:
                    age_period = "20-30"
                elif age < 40:
                    age_period = "30-40"
                elif age < 50:
                    age_period = "40-50"
                else:
                    age_period = "above 50"
                label = "{}, {}".format(age_period,
                                        "Male" if genders[i][0] < 0.5 else "Female")
                cv2.putText(frame, label, (d.left(), d.top()), cv2.FONT_HERSHEY_DUPLEX, 1.0, (92, 91, 241), 1, cv2.LINE_AA)
        else:
            if (screen_capture == True):
                screen_message = 'No Face Detected'
                screen_fontsize = 1.5
                screen_text_countdown = 5
                screen_text_point = (120, 200)

        if (screen_text_countdown > 0):  
            # print(screen_message)             
            cv2.putText(frame, screen_message, screen_text_point, cv2.FONT_HERSHEY_DUPLEX, screen_fontsize, (92, 91, 241), 1, cv2.LINE_AA)
            screen_text_countdown -= 1

        screen_capture = False

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(50)

        
        # esc
        if key == 27:
            break
        # keyboard c (CAREFUL! NO CAPS LOCK)
        if key == 99:
            screen_capture = True

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
