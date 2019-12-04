import numpy as np
import cv2
import scipy.io
from scipy.io import loadmat
from datetime import datetime
from tqdm import tqdm


def calculate_age(photo_date, date_of_birth):
    birth = datetime.fromordinal(max(int(date_of_birth) - 366, 1))

    if birth.month < 7:
        return photo_date - birth.year
    else:
        return photo_date - birth.year - 1


def get_meta(target_mat_path):
    meta = loadmat(target_mat_path)
    full_path = meta["wiki"][0, 0]["full_path"][0]
    date_of_birth = meta["wiki"][0, 0]["dob"][0]
    gender = meta["wiki"][0, 0]["gender"][0]
    photo_date = meta["wiki"][0, 0]["photo_taken"][0]
    face_score = meta["wiki"][0, 0]["face_score"][0]
    second_face_score = meta["wiki"][0, 0]["second_face_score"][0]
    age = [calculate_age(photo_date[i], date_of_birth[i]) for i in range(len(date_of_birth))]

    return full_path, date_of_birth, gender, photo_date, face_score, second_face_score, age


def main():
    result_mat_path = "data/wiki.mat"
    wiki_data_path = "data/wiki/"
    target_mat_path = "data/wiki/wiki.mat"
    img_size = 32
    min_score = 1.0

    full_path, _, gender, photo_taken, face_score, second_face_score, age = get_meta(target_mat_path)

    result_genders = []
    result_ages = []
    sample_num = len(face_score)
    result_img = np.empty((sample_num, img_size, img_size, 3), dtype=np.uint8)
    valid_sample_num = 0

    for i in tqdm(range(sample_num)):
        # ignore the photos with minimum score
        if face_score[i] < min_score:
            continue

        # ignore the photos with 2 faces
        if (not np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        # ignore the photos with invalid or to large age
        if not(0 <= age[i] <= 100):
            continue

        # ignore the photos with no gender labels
        if np.isnan(gender[i]):
            continue

        # ignore the photos with None
        img = cv2.imread(wiki_data_path + str(full_path[i][0]))
        if img is not None:
            result_genders.append(int(gender[i]))
            result_ages.append(age[i])
            result_img[valid_sample_num] = cv2.resize(img, (img_size, img_size))
            valid_sample_num += 1

    result_path = {"image": result_img[:valid_sample_num], "gender": np.array(result_genders), "age": np.array(result_ages),
              "db": "wiki", "img_size": img_size, "min_score": min_score}
    scipy.io.savemat(result_mat_path, result_path)
    return


if __name__ == '__main__':
    main()
