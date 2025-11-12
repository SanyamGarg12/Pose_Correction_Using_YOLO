# Real-Time AI Posture Correction for Powerlifting Exercises Using YOLOv5 and MediaPipe

![Project Banner](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/7a8d6fc1-2d21-45a9-b92f-fdd8cadc43b9)

A real-time AI-powered posture correction system for the "Big Three" powerlifting exercises (Bench Press, Squat, and Deadlift) using YOLOv5 for person detection and MediaPipe for pose estimation. The system provides real-time feedback on exercise form and counts repetitions automatically.

- **Demo Video**: https://youtu.be/u4f_sdjk1Ig
- **GitHub Repository**: https://github.com/SanyamGarg12/Pose_Correction_Using_YOLO.git

## 📋 Table of Contents

- [Citation](#citation)
- [Description](#description)
- [Features](#features)
- [Development Environment](#development-environment)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data](#data)
- [Training & Evaluation](#training--evaluation)
- [Feedback System](#feedback-system)
- [How to Use](#how-to-use)
- [Project Timeline](#project-timeline)
- [Award](#award)

## Citation

If you use this repository in your research, please cite our paper:

```bibtex
@article{ko2024real,
  author = {Ko, Yeong-Min and Nasridinov, Aziz and Park, So-Hyun},
  title = {Real-time AI posture correction for powerlifting exercises using YOLOv5 and MediaPipe},
  journal = {IEEE Access},
  year = {2024},
  volume = {4}
}
```

**Paper**: [IEEE Xplore - Paper Download](https://ieeexplore.ieee.org/abstract/document/10798440)

**Please CITE** our paper whenever this repository is used to help produce published results or incorporated into other software.

## Description

This project presents a comprehensive study on AI-powered posture correction for the three primary powerlifting exercises (Bench Press, Squat, and Deadlift) using YOLOv5 and MediaPipe. The system combines object detection, pose estimation, and machine learning classification to provide real-time feedback on exercise form.

**Research Duration**: September 1, 2023 - November 20, 2023

**Primary Contributor**: Yeong-Min Ko

## Features

- **Real-time Person Detection**: Uses YOLOv5 to detect and track the person closest to the camera
- **Pose Estimation**: MediaPipe Holistic for accurate body landmark detection
- **Exercise Classification**: Random Forest models for classifying correct/incorrect postures for each exercise
- **Repetition Counting**: Automatic counting of exercise repetitions
- **Real-time Feedback**: Audio and visual feedback for posture corrections
- **Multi-Exercise Support**: Bench Press, Squat, and Deadlift
- **Angle Calculation**: Real-time calculation of joint angles for biomechanical analysis

## Development Environment

- **OS**: macOS M1 & Windows 11 (NVIDIA GeForce RTX 4080 Ti)
- **Frameworks & Libraries**: 
  - YOLOv5 (Ultralytics)
  - MediaPipe
  - OpenCV
  - Streamlit
  - PyTorch
  - scikit-learn
- **Device**: iPhone 12 Pro (WebCam using iVCam)

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/SanyamGarg12/Pose_Correction_Using_YOLO.git
cd Pose_Correction_Using_YOLO
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download YOLOv5 model weights**:
   - The trained YOLOv5 model weights should be placed in `./models/best_big_bounding.pt`
   - If not available, the system will download the base YOLOv5 model from Ultralytics

4. **Ensure model files are present**:
   - `./models/benchpress/benchpress.pkl`
   - `./models/squat/squat.pkl`
   - `./models/deadlift/deadlift.pkl`

## Project Structure

```
Pose_Correction_Using_YOLO/
├── Streamlit.py                 # Main Streamlit application
├── Streamlit_NoneYolo.py       # Alternative version without YOLO
├── requirements.txt             # Python dependencies
├── Main.ipynb                   # Main notebook for development
├── yolov5_onlyPerson/           # YOLOv5 person detection training
│   ├── person.yaml              # YOLOv5 dataset configuration
│   ├── person/                  # Person detection dataset
│   └── trained_model/           # Trained YOLOv5 models
├── models/                      # Trained classification models
│   ├── benchpress/
│   ├── squat/
│   ├── deadlift/
│   └── best_big_bounding.pt    # YOLOv5 person detection model
├── labeling/                    # Data labeling notebooks
│   ├── benchpress/
│   ├── squat/
│   └── deadlift/
├── Afterprocessing/            # Post-processing notebooks
├── deeplearning_models/        # Deep learning model variants
├── resources/                   # Images, videos, and audio files
│   ├── images/
│   ├── videos/
│   └── sounds/                 # TTS audio feedback files
├── make_sounds/                # Text-to-speech generation
├── references/                 # Research papers and references
└── PAPER/                      # Research paper files
```

## Data

### YOLOv5 Person Detection Dataset

The YOLOv5 model was trained to detect only the person performing exercises. The dataset was compiled from multiple sources:

- **Bench Press**:
  - Google Search images (keyword: "Bench Press")
  - [Faller Computer Vision Project](https://universe.roboflow.com/jiangsu-ocean-universit/faller) (lying down postures)

- **Squat**:
  - [Squat-Depth Image Dataset](https://universe.roboflow.com/nejc-graj-1na9e/squat-depth/dataset/14/download)
  - [HumonBody1 Computer Vision Project](https://universe.roboflow.com/models/object-detection) (standing postures)

- **Deadlift**:
  - [SDT Image Dataset](https://universe.roboflow.com/isbg/sdt/dataset/5)

- **Additional Postures** (bending, lying, sitting, standing):
  - [Silhouettes of Human Posture](https://www.kaggle.com/datasets/deepshah16/silhouettes-of-human-posture)

### Exercise Posture Classification Dataset

Custom datasets were created for each exercise with labeled postures:
- Correct form
- Excessive lower-back arch
- Non-neutral spine
- Grip too wide/narrow
- Knees caving in
- Stance too wide
- And other common form errors

**Shooting Stand Position**:
- Bench Press: Side view recommended
- Squat and Deadlift: Front/side view recommended

## Training & Evaluation

### YOLOv5 Person Detection

- **Objective**: Detect only the person exercising
- **Hyperparameters**:
  - Epochs: 200 (early stopping at 167)
  - Batch size: 16
  - Initial weights: yolov5s.pt
  - Other parameters: default YOLOv5 settings

- **Performance Metrics**:
  | Precision | Recall | mAP_0.5 | mAP_0.5:0.95 |
  |-----------|--------|---------|--------------|
  | 0.987     | 0.990  | 0.99    | 0.686        |

### Exercise Classification Models

All classification models use **Random Forest** algorithm:

**Bench Press**:
| Accuracy | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| 0.961    | 0.963     | 0.961  | 0.961    |

**Squat**:
| Accuracy | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| 0.989    | 0.989     | 0.989  | 0.989    |

**Deadlift**:
| Accuracy | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| 0.947    | 0.949     | 0.947  | 0.948    |

## Feedback System

The system provides real-time feedback for common form errors:

### Bench Press
- **Excessive lower-back arch**: "Avoid arching your lower back too much; try to keep your chest open."
- **Grip too wide**: "Your grip is too wide. Hold the bar a bit narrower."

### Squat
- **Non-neutral spine**: "Try to avoid excessive curvature in your spine."
- **Knees caving in**: "Be cautious not to let your knees cave in."
- **Stance too wide**: "Narrow your stance to about shoulder width."

### Deadlift
- **Non-neutral spine**: "Try to avoid excessive curvature in your spine."
- **Grip too wide**: "Your grip is too wide. Hold the bar a bit narrower."
- **Grip too narrow**: "Hold the bar slightly wider than shoulder width."

## How to Use

1. **Start the Streamlit application**:
```bash
streamlit run Streamlit.py
```

2. **Select an exercise** from the sidebar dropdown:
   - Bench Press
   - Squat
   - Deadlift

3. **Position yourself** in front of the camera:
   - Ensure good lighting
   - Position yourself so you're clearly visible
   - Follow the recommended camera angles for each exercise

4. **Begin exercising**:
   - The system will automatically detect you
   - Real-time pose estimation will be displayed
   - Repetitions will be counted automatically
   - Feedback will be provided when form errors are detected

5. **Monitor your performance**:
   - View real-time joint angles in the sidebar
   - Check repetition count
   - Listen for audio feedback on form corrections

## Project Timeline

### Major Milestones

- **2023/09/10**: Successfully completed YOLOv5 person detection implementation
- **2023/09/11**: Integrated MediaPipe; improved labeling with additional spatial dimensions
- **2023/09/16**: Refined bounding boxes; achieved high accuracy pose estimation with YOLOv5 + MediaPipe; implemented Streamlit interface
- **2023/09/30 - 2023/10/02**: Collected datasets for exercise posture classification
- **2023/10/03 - 2023/10/08**: Completed dataset labeling, model training, and evaluation
- **2023/10/18**: Connected bench press model to server; implemented repetition counting algorithm
- **2023/10/24**: Successfully integrated all models (bench press, squat, deadlift) with server; completed paper
- **2023/11/05**: Implemented feedback mechanisms for each posture
- **2023/11/20**: Submitted finalized paper with experimental results

### Weekly Progress

- **Week 1**: Requirement Analysis
- **Week 2**: Prototype Development & Mini Test
- **Week 3**: Retrained person detection model; implemented holistic pose estimation
- **Week 4**: Paper writing
- **Week 5**: Paper writing and machine learning pipeline development
- **Week 6**: Mid-progress presentation
- **Week 7**: Linked bench press model to Streamlit server; implemented repetition counting
- **Week 8**: Paper writing; linked all models (bench press, squat, deadlift) to server
- **Week 9**: Implemented feedback for each posture
- **Week 10-11**: Paper feedback and revisions
- **Week 12**: Project completion

## Award

🏆 **Outstanding Paper Award**

This research received recognition for its contribution to the field of AI-powered fitness and posture correction.

## License

This project is open source. Please refer to the repository for license details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- YOLOv5 by Ultralytics
- MediaPipe by Google
- All dataset contributors and sources mentioned in the Data section

## Contact

For questions or inquiries, please open an issue on the GitHub repository.

---

**Note**: This project was developed as part of academic research. Please cite the paper if you use this work in your research or applications.
