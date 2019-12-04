import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


def extract_keyPoints_feature(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    key_points, features = sift.detectAndCompute(gray, None)
    img_kp = cv2.drawKeypoints(img_rgb, key_points, None)
    plt.imshow(img_kp, cmap="gray")
    plt.show()
    return key_points, features


def matching_and_plot(features1, features2, Ltype):
    match_pair = []
    features1 = np.array(features1)
    features2 = np.array(features2)
    distance = []
    for i in range(np.shape(features1)[0]):
        L1_lst = []
        for j in range(np.shape(features2)[0]):
            if Ltype == "L1":
                dist = compute_L1(features1[i], features2[j])
            elif Ltype == "L2":
                dist = compute_L2(features1[i], features2[j])
            else:
                dist = compute_L3(features1[i], features2[j])
            L1_lst.append([dist, j])
        L1_lst.sort(key=getfirst)
        # nearest
        a = L1_lst[0][1]
        # second nearest
        b = L1_lst[1][1]
        ratio = np.linalg.norm(features1[i] - features2[a])/np.linalg.norm(features1[i] - features2[b])
        if ratio < 0.8:
            match_pair.append([ratio, i, a])
            distance.append(dist)

    distance.sort()
    match_pair.sort(key=getfirst)
    return distance


def getfirst(ele):
    return ele[0]


def compute_L1(matrix1, matrix2):
    matrix = abs(matrix1 - matrix2)
    return sum(matrix)


def compute_L2(matrix1, matrix2):
    sum_sq = np.sum(np.square(matrix1 - matrix2))
    return math.sqrt(sum_sq)


def compute_L3(matrix1, matrix2):
    sum_tr = np.sum(np.power(abs(matrix1 - matrix2), 3))
    return np.cbrt(sum_tr)


def addRandomNoise(image):
    normalizedImg = np.zeros(np.shape(image))
    normalizedImg = cv2.normalize(image, normalizedImg, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    row, col, channel = np.shape(image)
    mean = 0
    sigma = 0.08
    noise = np.random.normal(mean, sigma, (row, col, channel))

    noise = noise.reshape(row, col, channel)
    image_noised = normalizedImg + noise

    image_noised = cv2.normalize(image_noised, image_noised, 0, 255, cv2.NORM_MINMAX)
    image_noised = image_noised.astype('uint8')

    return image_noised


def sift_rgb(image1, image2):
    keyPoints1, feature1 = extract_keyPoints_feature(image1)
    keyPoints2, feature2 = extract_keyPoints_feature(image2)

    kp_1, r_f1, g_f1, b_f1 = extract_all(image1, keyPoints1)
    kp_2, r_f2, g_f2, b_f2 = extract_all(image2, keyPoints2)

    match_pair = []
    f_size1 = np.shape(kp_1)[0]
    f_size2 = np.shape(kp_2)[0]
    for i in range(f_size1):
        L1_lst = []
        for j in range(f_size2):
            dist = compute_L2(r_f1[i], r_f2[j]) + compute_L2(g_f1[i], g_f2[j]) + compute_L2(b_f1[i], b_f2[j])
            L1_lst.append([dist, j])
        L1_lst.sort(key=getfirst)
        a = L1_lst[0][1]
        b = L1_lst[1][1]
        distanceA = compute_L2(r_f1[i], r_f2[a]) + compute_L2(g_f1[i], g_f2[a]) + compute_L2(b_f1[i], b_f2[a])
        distanceB = compute_L2(r_f1[i], r_f2[b]) + compute_L2(g_f1[i], g_f2[b]) + compute_L2(b_f1[i], b_f2[b])
        ratio = distanceA / distanceB
        if ratio < 0.8:
            match_pair.append([ratio, i, a])
    match_pair.sort(key=getfirst)

    # concatenate two images and draw lines for top 10 matches
    col_1 = np.shape(image1)[1]
    row_1 = np.shape(image1)[0]
    row_2 = np.shape(image2)[0]
    helper_matrix = np.zeros((row_2 - row_1, col_1, 3), dtype="uint8")

    img = np.vstack([image1, helper_matrix])
    img = np.hstack([img, image2])

    for i in range(10):
        first = match_pair[i][1]
        second = match_pair[i][2]
        x1 = keyPoints1[first].pt[0]
        y1 = keyPoints1[first].pt[1]
        x2 = keyPoints2[second].pt[0]
        y2 = keyPoints2[second].pt[1]
        plt.plot([int(x1), int(x2 + col_1)], [int(y1), int(y2)], linewidth=1, color='green')
    plt.imshow(img)
    plt.show()


def extract_all(image, kp):
    r, g, b = cv2.split(image)
    r_kp, r_f = extract(r, kp)
    g_kp, g_f = extract(g, kp)
    b_kp, b_f = extract(b, kp)

    return kp, np.array(r_f), np.array(g_f), np.array(b_f)


def extract(channel, kp):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, features = sift.compute(channel, kp)
    return kp, features

