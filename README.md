# Detecting Facial Expressions Through Webcam Using YOLO-Based Tracking
I trained YOLO models on a substantial dataset (~70k images) of facial expressions to recognize emotional patterns that human faces make. The project's purpose was to optimize a YOLO model for the task of detecting mood changes of a person in real-time, specifically on my desktop webcam.

<img width="1920" height="1280" alt="val_batch2_pred" src="https://github.com/user-attachments/assets/4e0af6a0-9add-4514-a414-110b80aecd9f" />

# Motivation
I chose this project because YOLO is something that I enjoyed working with in my undergraduate classes (Senior Design I & II). However, at that time I was only able to work with a very small dataset that was difficult to optimize because of the limited amount of data I could use to train the YOLO model. I had the idea for this project when I was looking for YOLO datasets in Kaggle and came across a dataset with tens of thousands of facial expressions [2]. My motivation for this dataset came about while I was playing a game because whenever I lost I would be sad for a couple of seconds. I wondered if YOLO could tell when this happens without any additional information about what I was doing. Some of what I've been doing in Digital Image Processing is trying to recognize patterns in images and I was wondering if I could get my YOLO algorithm to do the same for facial expressions. I wanted my YOLO model to do this as well as a human can because people are able to tell the expressions on another person’s face pretty well. I can tell when the people around me are happy, sad, or angry.

# Code
All of the code necessary to redo this project on your own is provided in this repo. Parts of this project were done in Google Colab so purchasing Colab Pro may be necessary because I ran into some runtime disconnections during my sessions more frequently using their free tier. It also gives you access to higher-tier GPUs that should complete the runtime faster. Some important libraries that are necessary for this project are Ultralytics, NumPy, and OpenCV. 

The [YOLO_Expressions](YOLO_Expressions.ipynb) Colab notebook contains all of the Python code necessary to train and test the YOLO model on my training and test datasets. You can follow the descriptions in the notebook to understand what each code cell does and if it needs something specifically from the user.

The [real_time_tracking](real_time_tracking.py) file is used to test the final model's real-time performance. My version is a modification of the one found in another github repo [1]. It records fps and latency calculations that will be discussed later in my results section. I ran this Python code locally on my machine to capture real-time data using my webcam so you may need to make some modifications to file paths for where the weights "best.pt" exist. Or you could just have the "best.pt" in the same directory as the [real_time_tracking](real_time_tracking.py) file.

The final model weights are available in the path:

[yolo_runs/yolo11n_run_final/weights](yolo_runs/yolo11n_run_final/weights)

# Dataset & Requirements
<img align="right" width="221" height="450" alt="canvas" src="https://github.com/user-attachments/assets/38bd98e0-0367-43a5-bce1-eba4e15cfe53">

The dataset I use for this project is titled “9 Facial Expressions for YOLO,” [2]. I discovered the dataset searching through Kaggle for YOLO datasets. Using this pre-made YOLO dataset eliminates the time-consuming step of collecting data, creating a labeling standard, and labeling the images myself. The dataset pre-organizes the folder structure as well so it's ready to be used after downloading it.

The dataset itself has images with facial expressions in a wide range of classes. The nine classes are as follows: angry, contempt, disgust, fear, happy, natural, sad, sleepy, and surprised. Images of each class are provided on the right. This dataset is perfect for this project because it contains a wide spread of emotions that can be seen on people’s faces. As such they are very well-suited for training machine learning algorithms.

The images in the dataset are formatted as JPGs and the size/resolution of the images vary. However, this doesn’t matter because YOLO handles the resizing itself automatically during the training process. It resizes images of varying resolutions to a standard size. Thus, anyone who uses this dataset does not have to worry about doing it themselves. 

There are 68,284 images in total split amongst the 3 different datasets (train, valid, test).

Dataset Split | Images (#) | Images (%)
--- | --- | --- 
Train | 64864 | 94.99%
Valid | 1720 | 2.52%
Test | 1700 | 2.49%
Total | 68284 | 100%

There is an overwhelming number of images for training but that’s how it should be for such a large dataset. Most of the learning that the YOLO model does is during the training phase so obviously that’s where most of the images will go. A small amount is used for validation just so the model can test its prediction ability on separate unseen images and evaluate its performance so it can tune hyperparameters and prevent overfitting. The training and validation images are unique meaning images do not appear in each other’s datasets. The test set has 1700 images (2.49% of the total). This amount is good enough for the testing because it is large enough
to show and measure the performance of my model. There’s no point in having a larger slice of the dataset because it’s not even learning at this point. The images in the test set are also unique in that none of the images that appear in the test dataset show up in the training or validation datasets. Thus, it is a fair test, and the performance results are not skewed due to repeat images.

The entire dataset was collected from multiple open-source facial expression datasets hosted on Roboflow. According to the author of the dataset, duplicates and corrupted files were removed during preprocessing. Images included in the datasets are sourced from real-world images, synthetic data, or annotated frames from videos. The creator of this dataset also warns that although the dataset is diverse there are some class imbalances. Expressions like contempt or sleepy appear less frequently in the dataset than common expressions such as happy or sad. This imbalance is inevitable because we see some emotions more often than others in real life. There will be some performance hits between the different classes due to this fact. I expect the common expressions to perform better.

<img width="1780" height="446" alt="image" src="https://github.com/user-attachments/assets/8d9e9b82-413f-432f-b242-934f436ccce2" />

# Work Flow
Talk about work flow. What did i start with, models I evaluated and analyzed, metric evaluation for the best model (similar metrics but yolo11n better than yolov8n for real time)

# Competetive Results
There was a recent IEEE paper "Facial Expression Recognition with YOLOv11 and YOLOv12: A Comparative Study" [3] from 2025 that reported good performance metrics I use as a comparison for my results. They provide results for YOLOv11n and YOLOv12n on two different datasets (FER2013 & KDEF) but I will only discuss their results using YOLOv11n because my final model ended up using YOLOv11n as well. It just makes it easier to compare results using the same model.

One of the easiest metrics they measured and reported were their confusion matrices provided below. A confusion matrix just gives you a visually easy method to identify True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) by comparing the prediction labels with the true labels. A perfect confusion matrix would have all values on the main diagonal squares. They have relatively good results in the left confusion matrix. The dark colors in the diagonals show a good amount of correct predictions for their classes. However, there is a reason why the left matrix performs so well that could be a detriment for actual implementation (Hint: Limited KDEF dataset). On the other hand, the right matrix does not perform as well. It has a good number of incorrect predictions for both false positives and false negatives.

<p align="center">
  <img width="808" height="385" alt="image" src="https://github.com/user-attachments/assets/edb79f17-db35-4ebf-998b-19363c990e1e" />
</p>

The table down below provides their reported performance metrics such as Precision, Recall, and Mean Average Precision for YOLOv11n. Again, the models that train on the KDEF dataset perform well but there's an issue with taking those results at face value. The performance for the dataset FER2013 does worse than KDEF in all metrics.

Model | Dataset | Precision% | Recall% | mAP@0.5%
--- | --- | --- | --- |---
YOLOv11n | FER2013 | 65.2 | 60.5 | 60.8
YOLOv11n | KDEF | 87.7 | 91.1 | 94.5

Their paper shows very strong results for the dataset KDEF but there are limitations with this dataset. It is not as well-suited for real-life use because it has a small dataset (4,900 samples) and has a very limited variation in poses and lighting. Thus, it performs well during testing on their test dataset but it will encounter a lot more problems trying to detect facial expressions from more diverse settings and environments. The dataset FER2013 performs poorly compared to KDEF because it just has a worse collection of data even if it has a much larger dataset (35,887 samples). A very important component missing from the paper is that it does not include any results for real-time implemention of their tracking model so we don't have any accuracy or latency performance to grade their models.

# My Results


# Future Work

# References
1. Ali Hassan, "YOLOv11 Face Emotion Detection," 2025, GitHub repository. [Online]. Available: https://github.com/alihassanml/Yolo11-Face-Emotion-Detection
2. Aklima Akter Rimi, "9 Facial Expressions for YOLO," 2025, Kaggle dataset. [Online]. Available: https://www.kaggle.com/datasets/aklimarimi/8-facial-expressions-for-yolo
3. U. Aymon, N. S. Kamarudin and A. F. A. Nasir, "[Facial Expression Recognition with YOLOv11 and YOLOv12: A Comparative Study](https://ieeexplore.ieee.org/abstract/document/11279248)," 2025 IEEE 9th International Conference on Software Engineering & Computer Systems (ICSECS), Pekan, Pahang, Malaysia, 2025, pp. 18-23.
