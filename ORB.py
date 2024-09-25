import numpy as np
import cv2 as cv

# Initialize webcam and DroidCam video capture
webcam = cv.VideoCapture(0)
droidcam = cv.VideoCapture(2)

# Initiate ORB detector
orb = cv.ORB_create()

# Create a window to display the results
cv.namedWindow('ORB Feature Matching', cv.WINDOW_NORMAL)
cv.resizeWindow('ORB Feature Matching', 1200, 600)

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

    # Find keypoints and descriptors with ORB
    kp1_orb, des1_orb = orb.detectAndCompute(img_webcam, None)
    kp2_orb, des2_orb = orb.detectAndCompute(img_droidcam, None)

    if des1_orb is None or des2_orb is None:
        print("Failed to compute ORB descriptors.")
        continue

    # Brute-force matcher for ORB with NORM_HAMMING
    bf_orb = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches_orb = bf_orb.match(des1_orb, des2_orb)

    # Sort the matches by distance
    matches_orb = sorted(matches_orb, key=lambda x: x.distance)

    # Draw only the first 10 matches
    img_matches_orb = cv.drawMatches(img_webcam, kp1_orb, img_droidcam, kp2_orb, matches_orb[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the result
    cv.imshow('ORB Feature Matching', img_matches_orb)

    if cv.waitKey(1) == ord('q'):
        break

# Release and destroy
webcam.release()
droidcam.release()
cv.destroyAllWindows()

