import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures['serving_default']


def out_data(frame, keypoints_with_scores):
    pos_vals = []
    for person in keypoints_with_scores:
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(person, [y, x, 1]))
        width = ((shaped[5][0] - shaped[6][0])**2 + (shaped[5][1] - shaped[6][1])**2)**0.5
        mid_point = (shaped[5][0] + shaped[6][0]) / 2, (shaped[5][1] + shaped[6][1]) / 2
        if shaped[5][2] > 0.1 and shaped[6][2] > 0.1:
            if width < 20:
                width = 20

            person_pos_val = mid_point, width
            print(person_pos_val)
            pos_vals.append(person_pos_val)
    return pos_vals


def cull_duplicates(people_vals, cull_distance):
    codes = []  # value given to each "person" 0 = accepted person, 1 = duplicate person
    pos = 0
    for person in people_vals:
        if pos < len(people_vals) - 1:
            person_acceptance = 0
            for other_person in people_vals[pos + 1:]:
                distance = ((person[0][0] - other_person[0][0]) ** 2 + (person[0][1] - other_person[0][1]) ** 2) ** 0.5
                if distance < cull_distance:
                    person_acceptance = 1
            codes.append(person_acceptance)
        else:
            codes.append(0)
        pos = pos + 1
    return codes


def social_distancing_determinor(frame, people_vals, min_distance):
    social_codes = []  # 0 for socially distanced 1 for notSocially distanced
    too_close = []
    pos = 0
    for person in people_vals:
        person_code = 0
        for other_person in [x for i, x in enumerate(people_vals) if i != pos]:
            if person[1] > other_person[1]:
                ratio = ((person[1] + other_person[1])/other_person[1])**2
                mult = math.exp((person[1]/other_person[1])**1.5)
            else:
                ratio = ((other_person[1] + person[1])/person[1])**2
                mult = math.exp((other_person[1] / person[1])**1.5)
            ratio = ratio * mult

            pythag_distance = ((person[0][0] - other_person[0][0]) ** 2 + (person[0][1] - other_person[0][1]) ** 2) ** 0.5
            distance = (pythag_distance**2 + ratio**2)**0.5
            if distance <= min_distance:
                person_code = 1
                couple = person, other_person
                too_close_member = couple, distance
                too_close.append(too_close_member)
        social_codes.append(person_code)
        pos = pos + 1
    return social_codes, too_close


def new_vals(lst, codes):
    out_vals = []
    pos = 0
    for code in codes:
        if code == 0:
            out_vals.append(lst[pos])
        pos = pos + 1
    return out_vals


def plot_socially_distanced(frame, people_vals, social_codes):
    pos = 0
    for person in people_vals:
        mid_point = person[0]
        if social_codes[pos] == 0:
            cv2.circle(frame, (int(mid_point[1]), int(mid_point[0])), 10, (0, 255, 0), -1)
        else:
            cv2.circle(frame, (int(mid_point[1]), int(mid_point[0])), 10, (225, 0, 0), -1)
        pos = pos + 1


def check_equal(person, other_person):
    return person[0][0] == other_person[0][0] and person[0][1] == other_person[0][1]


def too_close_cull_duplicates(too_close):
    pos = 0
    codes = []
    for member in too_close:
        person, other_person = member[0]
        code = 0
        if(pos < len(too_close) - 1):
            for other_member in too_close[pos + 1:]:
                if (other_member):
                    other_member_person, other_member_other_person = other_member[0]
                    if (check_equal(person, other_member_other_person)) and (
                    check_equal(other_person, other_member_person)):
                        code = 1
        codes.append(code)
        pos = pos + 1
    return codes


def plot_too_close_lines(frame, too_close):
    for member in too_close:
        couple, distance = member
        person, other_person = couple
        pt1 = (int(person[0][1]), int(person[0][0]))
        pt2 = (int(other_person[0][1]), int(other_person[0][0]))
        mid_pt = (pt1[0], pt1[1]) if pt1[0] < pt2[0] else (pt2[0], pt2[1])
        mid_pt = mid_pt[0], mid_pt[1] - 50
        print(int(distance))
        txt = str(int(distance)) + "cm"
        cv2.putText(frame, txt, mid_pt, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,0,0), 2)
        cv2.line(frame, pt1, pt2, (255, 0, 0), 8)


def run_ai(image_in):
    img = image_in
    img_scaled = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), img.shape[0], img.shape[1])
    input_img = tf.cast(img_scaled, dtype=tf.int32)
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
    people_vals = out_data(img, keypoints_with_scores)
    codes = cull_duplicates(people_vals, 5)
    people_vals = new_vals(people_vals, codes)
    social_codes, too_close = social_distancing_determinor(img, people_vals, 200)
    too_close_codes = too_close_cull_duplicates(too_close)
    too_close = new_vals(too_close, too_close_codes)
    plot_socially_distanced(img, people_vals, social_codes)
    plot_too_close_lines(img, too_close)
    plt.imshow(img)
    plt.show()



print("getting image")
img_address = input("Address of the image (type 'exit' to quit): ")
while(img_address != "exit"):
    in_img = cv2.imread(img_address)
    run_ai(in_img)
    img_address = input("Address of the image (type 'exit' to quit): ")

