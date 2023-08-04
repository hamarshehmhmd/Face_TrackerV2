# Face and Eye Detection Project

This is a simple Python script that uses the OpenCV and face_recognition libraries to perform face and eye detection on live video feed from a webcam. The script also checks if a detected face matches a reference image provided by the user.

## Requirements

- Python 3
- OpenCV (`cv2`)
- Numpy (`numpy`)
- face_recognition (`face_recognition`)

## Installation

You can install the required libraries using pip:

```pip install opencv-python numpy face-recognition```


## Usage

1. Clone or download the project files to your local machine.

2. Ensure you have a reference image that you want to use for face matching. Place this image on your desktop and update the `reference_image_path` variable in the script with the correct file name.

3. Run the script:

```python face_eye_detection.py```


4. The script will open a new window showing the video feed from your webcam with rectangles drawn around detected faces and eyes. If a face matches the reference image, it will be labeled as "Identity Match" in green; otherwise, it will be labeled as "Human" in red.

5. To quit the script, press the 'q' key in the video feed window.

## Troubleshooting

1. If you encounter any issues related to the face_recognition library, make sure you have installed it correctly. You can find installation instructions on the [face_recognition GitHub page](https://github.com/ageitgey/face_recognition).

2. If the script is not detecting faces or eyes properly, you can try adjusting the `scaleFactor` and `minNeighbors` parameters in the `face_cascade.detectMultiScale()` and `eye_cascade.detectMultiScale()` functions, respectively.

## Limitations

- The accuracy of face matching heavily depends on the quality and similarity of the reference image with the faces in the video feed.
- This script may not be suitable for real-world face recognition applications that require high accuracy and security.

## Credits

- This project uses the OpenCV library for face and eye detection.
- The face recognition is powered by the face_recognition library.

## License

This project is licensed under the [MIT License](LICENSE).
