# Smart Surface Scratch Detection System

## About the Project
This project detects scratches on steel surfaces using computer vision and the YOLO11n object detection model. It allows users to upload an image and identify possible scratches with bounding boxes.

## How It Works
1. Collect steel surface images from the NEU-DET dataset.
2. Prepare images and convert XML annotations into YOLO format.
3. Train the YOLO11n model to detect scratches.
4. Upload an image through the Streamlit application.
5. Detect scratches and display bounding boxes with the detection count.

## Technologies Used
- Python
- YOLO11n
- OpenCV
- Streamlit
- PyTorch

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

Upload a steel surface image and click **Detect Scratches** to view the results.

## Dataset
NEU Surface Defect Database (Scratch Class).

## Future Improvements
- Real-time webcam detection.
- Improve detection accuracy.
- Add detection history and reports.

## Author
Kalyani Shinde
