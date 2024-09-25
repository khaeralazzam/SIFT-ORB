import numpy as np
import cv2 as cv

# Initialize webcam and DroidCam video capture
webcam = cv.VideoCapture(0)
droidcam = cv.VideoCapture(2)

# Initiate SIFT detector
sift = cv.SIFT_create()

# Create a window to display the results
cv.namedWindow('SIFT Feature Matching', cv.WINDOW_NORMAL)
cv.resizeWindow('SIFT Feature Matching', 1200, 600)

while True:
    # Capture frame from webcam
    ret_webcam, frame_webcam = webcam.read()
    if not ret_webcam:
        print("Failed to capture frame from webcam.")
        break

    # Capture frame from DroidCam video feed
    ret_droidcam, frame_droidcam = droidcam.read()
    if not ret_droidcam:
        print("Failed to capture frame from DroidCam.")
        break

    # Convert frames to grayscale
    img_webcam = cv.cvtColor(frame_webcam, cv.COLOR_BGR2GRAY)
    img_droidcam = cv.cvtColor(frame_droidcam, cv.COLOR_BGR2GRAY)

    # Find keypoints and descriptors with SIFT
    kp1_sift, des1_sift = sift.detectAndCompute(img_webcam, None)
    kp2_sift, des2_sift = sift.detectAndCompute(img_droidcam, None)

    if des1_sift is None or des2_sift is None:
        print("Failed to compute SIFT descriptors.")
        continue

    # Convert descriptors to float32
    des1_sift = des1_sift.astype(np.float32)
    des2_sift = des2_sift.astype(np.float32)

    # Brute-force matcher for SIFT
    bf_sift = cv.BFMatcher()
    matches_sift = bf_sift.knnMatch(des1_sift, des2_sift, k=2)

    # Ratio test as per Lowe's paper
    matchesMask_sift = [[0, 0] for _ in range(len(matches_sift))]
    for i, (m, n) in enumerate(matches_sift):
        if m.distance < 0.7 * n.distance:
            matchesMask_sift[i] = [1, 0]

    draw_params_sift = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask_sift, flags=cv.DrawMatchesFlags_DEFAULT)
    img_matches_sift = cv.drawMatchesKnn(img_webcam, kp1_sift, img_droidcam, kp2_sift, matches_sift, None, **draw_params_sift)

    # Display the result
    cv.imshow('SIFT Feature Matching', img_matches_sift)

    if cv.waitKey(1) == ord('q'):
        break

# Release and destroy
webcam.release()
droidcam.release()
cv.destroyAllWindows()
