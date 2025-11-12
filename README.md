# Real-Time AI Posture Correction for Powerlifting Exercises Using YOLOv5 and MediaPipe
![rea](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/7a8d6fc1-2d21-45a9-b92f-fdd8cadc43b9)

- Demo Video: https://youtu.be/u4f_sdjk1Ig

## Citation
Details of algorithms and experimental results can be found in our following paper:
```bibtex
@article{ko2024real,
  author = {Ko, Yeong-Min and Nasridinov, Aziz and Park, So-Hyun},
  title = {Real-time AI posture correction for powerlifting exercises using YOLOv5 and MediaPipe},
  journal = {IEEE Access},
  year = {2024},
  volume = {4}
}
```

Paper: <a href="https://ieeexplore.ieee.org/abstract/document/10798440">Paper Download</a><br>
<b>Please CITE</b> our paper whenever this repository is used to help produce published results or incorporated into other software.

### Description
A Study on the big three exercises AI posture correction service Using YOLOv5 and MediaPipe<br>
Research Duration: 2023.09.01 ~ 2023.11.20 <br>
|<b>Yeong-Min Ko</b>|
|:--:|
|<img height="180" src="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/6b04d3f5-e87a-4a2b-a2a9-e406e575b6fd">|

### Development Environment
- OS: MAC m1 & Windows 11(NVIDIA GeForce RTX 4080 Ti)<br>
- Frameworks & Libraries: YOLOv5, MediaPipe, OpenCV, Streamlit
- Device: iPhone 12 Pro(WebCam using iVCam)

## Data
- YOLOv5: <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/tree/main/yolov5_onlyPerson">Detect only one Person</a>
  - Scraping Images from Google & Roboflow
    - Bench Press
      - Google Search Keyword: Bench Press
      - <a href="https://universe.roboflow.com/jiangsu-ocean-universit/faller">Faller Computer Vision Project</a> with lying down
    - Squat
      - <a href="https://universe.roboflow.com/nejc-graj-1na9e/squat-depth/dataset/14/download">Squat-Depth Image Dataset</a>
      - <a href="https://universe.roboflow.com/models/object-detection">HumonBody1 Computer Vision Project</a> with Standing
    - Deadlift
      - <a href="https://universe.roboflow.com/isbg/sdt/dataset/5">SDT Image Dataset</a>
    - More(bending, lying, sitting, standing)
      - <a href="https://www.kaggle.com/datasets/deepshah16/silhouettes-of-human-posture">Silhouettes of human posture</a>
  - Inference
    |Test 1|Test 2|
    |:---:|:---:|
    |![Screenshot 2023-09-17 15-48-41](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/87a0b701-2df5-400b-9001-d4f526bf8211)|![Screenshot 2023-09-17 15-48-18](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/750a4a99-cfd6-44cc-ae95-a4db42e7b67a)|
    |<b>Applied 1</b>|<b>Applied 2</b>|
    |<img width="561" alt="Deadlift Example 2" src="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/e3d2e1f9-8fda-422d-9096-64d77b337b00">|<img width="564" alt="Squat Example 2" src="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/5e17f5a1-dc61-4152-82ea-24d86c563ee9">|
- Exercise Posture Correction
  - Shooting Stand Position
    |Bench Press|Squat and Deadlift|
    |:--:|:--:|
    |![KakaoTalk_Photo_2023-10-04-13-37-53 001](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/59701404-4648-497d-9893-7e617e1dd928)|![KakaoTalk_Photo_2023-10-04-13-37-54 003](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/fc403019-f221-4113-a1cf-44a5de9a042f)|
  - Example of Shooting stand position for Bench Press
    |Picture(Left, Center, Right)|
    |:--:|
    |![KakaoTalk_20231004_132437850](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/d2aec3e9-59ed-4eba-b4e5-8649cbe18260)|
  - Bench Press: <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/tree/main/labeling/benchpress">read more</a>
    - ![Screenshot 2023-10-08 19-20-17](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/319898a4-56ba-4c6c-9b89-63b4cf885148)
  - Squat: <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/tree/main/labeling/squat">read more</a>
    - ![Screenshot 2023-10-08 20-30-46](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/15e25bf1-5a02-4b62-ba07-65f7a1ac8bfe)
  - Deadlift: <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/tree/main/labeling/deadlift">read more</a>
    - ![Screenshot 2023-10-08 20-34-30](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/ef3795db-e999-477c-824b-58ab4845ab55)

## Training & Evaluation
### YOLOv5
  - Detect only a Person exercising something
    - Hyperparameters to train
      - epochs 200(but early stopping: 167)
      - batch 16
      - weights yolov5s.pt
      - etc are set by 'default'
  - Performance Evaluation
    |Precision|Recall|mAP_0.5|mAP_0.5:0.95|
    |:--:|:--:|:--:|:--:|
    |0.987|0.990|0.99|0.686|
    
### Exercise Classfication
  - Bench Press (Algorithm: Random Forest)
    |Accuracy|Precision|Recall|F1-Score|
    |:--:|:--:|:--:|:--:|
    |0.961|0.963|0.961|0.961|
  - Squat (Algorithm: Random Forest)
    |Accuracy|Precision|Recall|F1-Score|
    |:--:|:--:|:--:|:--:|
    |0.989|0.989|0.989|0.989|
  - Deadlift (Algorithm: Random Forest)
    |Accuracy|Precision|Recall|F1-Score|
    |:--:|:--:|:--:|:--:|
    |0.947|0.949|0.947|0.948|

## Feedback
  |Bench Press|Squat|Deadlift|
  |:--:|:--:|:--:|
  |<b>Excessive lower-back arch</b><br>Avoid arching your lower back too much; try to keep your chest open.|<b>Non-neutral spine</b><br>Try to avoid excessive curvature in your spine.|<b>Non-neutral spine</b><br>Try to avoid excessive curvature in your spine.|
  |<b>Excessive lower-back arch</b><br>Raise your pelvis slightly and brace your core to keep your back flat.|<b>Non-neutral spine</b><br>Lift your chest and pull your shoulders back.|<b>Non-neutral spine</b><br>Lift your chest and pull your shoulders back.|
  |<b>Grip too wide</b><br>Your grip is too wide. Hold the bar a bit narrower.|<b>Knees caving in</b><br>Be cautious not to let your knees cave in.|<b>Grip too wide</b><br>Your grip is too wide. Hold the bar a bit narrower.|
  |<b>Grip too wide</b><br>When gripping the bar, hold it slightly wider than shoulder width.|<b>Knees caving in</b><br>Push your hips back to keep your knees and toes aligned.|<b>Grip too wide</b><br>When gripping the bar, hold it slightly wider than shoulder width.|
  ||<b>Stance too wide</b><br>Narrow your stance to about shoulder width.|<b>Grip too narrow</b><br>Hold the bar slightly wider than shoulder width.|

## How to Use
- Open your terminal in mac, linux or your command prompt in Windows. Then, type "<b>Streamlit run Streamlit.py</b>".<br>
  <img width="387" alt="Screenshot Streamlit Run Command" src="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/23a85105-0836-4632-86a9-a0f87017852d">
  |This Service|
  |:---:|
  |<img width="632" alt="Screenshot 2023-12-03 21-01-41" src="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/e891d8fd-3af1-4d9a-a51d-4425aa9df452">|
  
## Major project records
- 2023/09/10: 2023/09/10: Successfully concluded a project utilizing YOLOv5 to detect a singular individual.
- 2023/09/11: Integration with Mediapipe yielded lower accuracy than anticipated. Consequently, we decided to enhance labeling by introducing additional spatial dimensions around individuals.
- 2023/09/16: Significantly refined bounding boxes for model training, resulting in a triumphant pose estimation with remarkable accuracy when employing YOLOv5 and Mediapipe in tandem. Implemented a Streamlit file for holistic pose estimation after detecting the nearest person using YOLOv5. And <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/Streamlit.py">the streamlit file</a> was impleted to estimate holistic pose after detecting only person closest to the camera using yolov5.
- 2023/09/30 ~ 2023/10/02: Gathered datasets for training an exercise posture classification model.
- 2023/10/03 ~ 2023/10/08: Commenced with class labeling of the dataset, followed by model training and conclusive evaluations.
- 2023/10/18: Established a connection between the bench press model and the server, implementing an algorithm to count bench press repetitions. Additionally, in the process of linking two additional models: deadlift and squat.
- 2023/10/24: Successfully integrated all models and the server, culminating in the completion of the paper.
- 2023/11/05: Implemented feedback mechanisms for each specific posture.
- 2023/11/20: Submitted the finalized paper along with experimental results.

## Project Progress
- Week 1: Requirement Analysis
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_1.pdf">Read More</a>
- Week 2: Prototype Development & Mini Test
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_2.pptx">Read More</a>
- Week 3: Retrain the model detecting only person and Estimate holistic pose after detecting only person closest to the camera using yolov5
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_3.pdf">Read More</a>
- Week 4: Write the paper
- Week 5: Write the paper and Develop machine learning pipelines
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_5.pdf">Read More</a>
- Week 6: Presentation of project mid-progress
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/mid-term%20presentation.pdf">Read More</a>
- Week 7: Link the bench press model and the streamlit server / Implement an algorithm to count the number of bench press
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_7.pdf">Read More</a>
- Week 8: Write the paper and Link all models(bench press, squat, deadlift) and the streamlit server
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_8.pdf">Read More</a>
- Week 9: Implement feedback for each posture
  - <a href="https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/blob/main/docs/presentation_9.pptx">Read More</a>
- Week 10: Paper Feedback
- Week 11: Paper Feedback
- Week 12: Finish the project

## Award
![Outstanding Paper Award](https://github.com/PSLeon24/AI_Exercise_Pose_Feedback/assets/59058869/73ec0496-63c6-4a10-80cc-86c20fffb3da)
