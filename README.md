# TextiScan Pro
### AI-Powered Textile Defect Detection Using YOLOv8n

TextiScan Pro is a computer vision-based textile inspection prototype developed by Computer Science students at Salim Habib University, Karachi.

The project uses a YOLOv8n object detection model trained on the BDFD9 fabric defect dataset to identify and highlight defects in fabric images.

## Key Features

- Fabric defect detection using YOLOv8n
- Support for nine fabric defect classes
- Image upload and camera-based inspection
- Red bounding boxes with defect labels and confidence scores
- Audio alerts for detected defects
- Downloadable PDF inspection reports
- Inspection history and analytics
- Manual review and feedback

## Supported Defect Classes

1. DropStitch
2. Hole
3. Horizontal Mark
4. Knot
5. Oil Spot
6. Setup
7. Slub
8. Vertical LineMark
9. Yurn Dust

## Dataset and Model

The model was trained using the BDFD9 fabric defect dataset.

- **Model:** YOLOv8n
- **Training epochs:** 30
- **Image size:** 640 × 640
- **Training hardware:** CPU
- **Framework:** Ultralytics YOLO

## Evaluation Results

The latest evaluation was performed on a modified test folder containing 111 images and 134 labeled defect instances.

| Metric | Result |
|---|---|
| Precision | 73.54% |
| Recall | 84.62% |
| mAP@0.50 | 87.62% |
| mAP@0.50:0.95 | 48.88% |

The test folder included additional clean-fabric images used for exploratory testing. These results should not be treated as a direct benchmark on the unchanged original test split.

## Technologies Used

Python, YOLOv8n, PyTorch, OpenCV, Streamlit, SQLite, Pandas, NumPy, Pillow, and ReportLab.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run textiscan_pro_v3.py
```

The trained model file `best.pt` must be available in the repository's main directory, and the application must be configured to load it from that location.

## Project Limitations

TextiScan Pro is a research prototype. Its predictions may include false positives or missed defects. Performance on different fabric materials and real industrial environments requires further testing.

## Code Availability

The application source code and trained YOLOv8n model weights are available in this repository for academic review and reproducibility.
