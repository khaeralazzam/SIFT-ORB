import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Initialize webcam and DroidCam video capture
webcam = cv.VideoCapture(0)  # Index 0 represents the default webcam
droidcam = cv.VideoCapture(2)  # Use the appropriate index for DroidCam capture (e.g., index 2)

# Initiate SIFT detector
sift = cv.SIFT_create()

# Initiate ORB detector
orb = cv.ORB_create()

# Create a window to display the results
cv.namedWindow('Feature Matching', cv.WINDOW_NORMAL)
cv.resizeWindow('Feature Matching', 1200, 600)

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

    # Find the keypoints and descriptors with SIFT for the webcam frame
    kp1_sift, des1_sift = sift.detectAndCompute(img_webcam, None)

    # Find the keypoints and descriptors with SIFT for the DroidCam frame
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

    # Need to draw only good matches for SIFT, so create a mask
    matchesMask_sift = [[0, 0] for _ in range(len(matches_sift))]

    # Ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches_sift):
        if m.distance < 0.7 * n.distance:
            matchesMask_sift[i] = [1, 0]

    draw_params_sift = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=matchesMask_sift,
        flags=cv.DrawMatchesFlags_DEFAULT
    )

    img_matches_sift = cv.drawMatchesKnn(img_webcam, kp1_sift,
                                         img_droidcam, kp2_sift, matches_sift, None, **draw_params_sift)

    # Find the keypoints and descriptors with ORB for the webcam frame
    kp1_orb, des1_orb = orb.detectAndCompute(img_webcam, None)

    # Find the keypoints and descriptors with ORB for the DroidCam frame
    kp2_orb, des2_orb = orb.detectAndCompute(img_droidcam, None)

    if des1_orb is None or des2_orb is None:
        print("Failed to compute ORB descriptors.")
        continue

    # Create BFMatcher object for ORB
    bf_orb = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches_orb = bf_orb.match(des1_orb, des2_orb)

    # Sort the matches by distance
    matches_orb = sorted(matches_orb, key=lambda x: x.distance)

    # Draw only the first 10 matches for ORB
    img_matches_orb = cv.drawMatches(img_webcam, kp1_orb, img_droidcam,
                                     kp2_orb, matches_orb[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Create a composite image with the SIFT and ORB matches
    composite_img = np.zeros((600, 1200, 3), dtype=np.uint8)
    composite_img[:600, :600] = cv.resize(img_matches_sift, (600, 600))
    composite_img[:600, 600:1200] = cv.resize(img_matches_orb, (600, 600))

    # Display the composite image
    cv.imshow('Feature Matching', composite_img)

    if cv.waitKey(1) == ord('q'):
        break

# Release webcam and DroidCam video capture, and destroy any open windows
webcam.release()
droidcam.release()
cv.destroyAllWindows()
