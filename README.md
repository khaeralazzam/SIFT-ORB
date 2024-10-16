# SIFT-ORB
This project demonstrates real-time feature matching using **SIFT** (Scale-Invariant Feature Transform) and **ORB** (Oriented FAST and Rotated BRIEF) in OpenCV. The program captures video streams from a webcam and a DroidCam feed, detects keypoints and descriptors in real-time using both SIFT and ORB, and performs feature matching between the two frames.

## Features
- **SIFT (Scale-Invariant Feature Transform)**: A popular feature detection and description algorithm, particularly effective for matching across scale and rotation differences.
- **ORB (Oriented FAST and Rotated BRIEF)**: A computationally efficient alternative to SIFT, optimized for real-time applications with limited resources.

The application shows both SIFT and ORB matching results side by side in a resizable window.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy

You can install the necessary dependencies using pip:

```bash
pip install opencv-python opencv-contrib-python numpy
```
## How To Run

**1. Clone the repository** (if applicable):

```bash
git clone <repository-url>
cd <repository-folder>
```
**2. Connect your webcam** and set up DroidCam on your smartphone (or any secondary video source).

**3. Run the script** for either SIFT or ORB feature matching:

- To run the **SIFT** version:

```bash
python SIFT.py
```

- To run the **ORB** version:

```bash
ptyhon ORB.py
```

4. Press the q key to exit the application at any time.

## Code overview

### SIFT Feature Matching (SIFT.py)
This script captures video frames from both the webcam and DroidCam, converts the frames to grayscale, detects keypoints and computes descriptors using the SIFT algorithm, and then performs a brute-force matching between the descriptors. The good matches are displayed in a window.

### ORB Feature Matching (ORB.py)
This script functions similarly to the SIFT version but uses the ORB algorithm for feature detection and descriptor computation. It uses a brute-force matcher with the Hamming norm to match the descriptors and displays the top 10 matches in a window.

## How it Works
- The webcam and DroidCam feeds are captured and converted to grayscale.
- For both SIFT and ORB:
  - Keypoints are detected, and descriptors are computed for each frame.
  - A brute-force matcher compares the descriptors between the two frames.
  - Good matches are visualized using the OpenCV drawMatches function.
- The results are displayed in real-time in a window.

## Dependencies
- OpenCV: Provides tools for capturing video, detecting features, and performing image processing.
- NumPy: Used for numerical operations such as converting descriptors.

## Troubleshooting
- **Video Capture Issues:** If the script cannot capture video from either source, make sure that the index numbers (0 for the webcam and 2 for DroidCam) are correctly set according to your system.
- **DroidCam Setup:** Make sure that DroidCam is correctly set up on your phone and that the IP or connection settings match in the app and your PC.
- **SIFT or ORB Failure:** If feature detection fails, ensure that the frames are correctly captured, and you have sufficient lighting for clear image recognition.
