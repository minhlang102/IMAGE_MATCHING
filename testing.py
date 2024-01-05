import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10

selected = 'ORB'

features_detetion_time = 0
feature_matching_outliers_rejected_time = 0
total_matching_time = 0
start_time_1 = 0
start_time_2 = 0

print(selected)

img1 = cv.imread('sample1.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('sample2.png', cv.IMREAD_GRAYSCALE)

kp1 = []
kp2 = []
des1 = []
des2 = []

if selected == 'SIFT':
    sift = cv.xfeatures2d.SIFT_create(contrastThreshold=0.04, sigma=1.6)
    start_time_1 = cv.getTickCount()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    end_time = cv.getTickCount()
    features_detetion_time = (end_time - start_time_1)/cv.getTickFrequency()
elif selected == 'SURF64':
    surf = cv.xfeatures2d.SURF_create()
    surf.setHessianThreshold(100)
    surf.setExtended(True)
    start_time_1 = cv.getTickCount()
    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(img2,None)
    end_time = cv.getTickCount()
    features_detetion_time = (end_time - start_time_1)/cv.getTickFrequency()
elif selected == 'SURF32':
    surf = cv.xfeatures2d.SURF_create(100)
    surf.setExtended(False)
    start_time_1 = cv.getTickCount()
    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(img2,None)
    end_time = cv.getTickCount()
    features_detetion_time = (end_time - start_time_1)/cv.getTickFrequency()
elif selected == 'ORB':
    orb = cv.ORB_create(100000)
    start_time_1 = cv.getTickCount()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    end_time = cv.getTickCount()
    features_detetion_time = (end_time - start_time_1)/cv.getTickFrequency()

# BFMatcher with default params
start_time_2 = cv.getTickCount()

bf = cv.BFMatcher()
if selected == 'ORB':
    bf = cv.BFMatcher(cv.NORM_HAMMING)
else:
    bf = cv.BFMatcher(cv.NORM_L1)
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

    # Get the number of features detected, matched, and outliers rejected
    num_features_detected = len(kp1)
    num_features_matched = len(good)
    num_outliers_rejected = np.sum(1 - np.array(matchesMask))

    print("Number of Features Detected: ", num_features_detected)
    print("Number of Features Matched: ", num_features_matched)
    print("Number of Outliers Rejected: ", num_outliers_rejected)

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

end_time = cv.getTickCount()
total_time = (end_time - start_time_1)/cv.getTickFrequency()
feature_matching_outliers_rejected_time = (end_time - start_time_2)/cv.getTickFrequency()

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
print("Feature detection time: ", features_detetion_time)
print("Feature matching & outliers rejected time: ", feature_matching_outliers_rejected_time)
print("Total time: ", total_time)
plt.title(selected)
plt.axis('off')
plt.imshow(img3, 'gray'), plt.show()
