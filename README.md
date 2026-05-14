# Detecting Facial Expressions Through Webcam Using YOLO-Based Tracking
I trained YOLO models on a substantial dataset (~70k images) of facial expressions to recognize emotional patterns that human faces make. The project's purpose was to optimize a YOLO model for the task of detecting mood changes of a person in real-time, specifically on my desktop webcam.

<img width="1920" height="1280" alt="val_batch2_pred" src="https://github.com/user-attachments/assets/4e0af6a0-9add-4514-a414-110b80aecd9f" />

# Motivation
I chose this project because YOLO is something that I enjoyed working with in my undergraduate classes (Senior Design I & II). However, at that time I was only able to work with a very small dataset that was difficult to optimize because of the limited amount of data I could use to train the YOLO model. I had the idea for this project when I was looking for YOLO datasets in Kaggle and came across a dataset with tens of thousands of facial expressions. My motivation for this dataset came about while I was playing a game because whenever I lost I would be mad for a couple of seconds. I wondered if YOLO could tell when this happens without any additional information about what I was doing. Some of what I've been doing in Digital Image Processing is trying to recognize patterns in images and I was wondering if I could get my YOLO algorithm to do the same for facial expressions. I wanted my YOLO model to do this as well as a human can because people are able to tell the expressions on another person’s face pretty well. I can tell when the people around me are happy, sad, or angry.

# Code
All of the code necessary to run my project on your own is provided in this repo. Parts of this project were done in Google Colab so purchasing Colab Pro may be necessary because I ran into some runtime disconnections during my sessions more frequently using their free tier. It also gives you access to higher-tier GPUs that should complete the runtime more quickly. Some important libraries that are necessary for this project are Ultralytics to use YOLO and OpenCV to capture live video. 

The [YOLO_Expressions](YOLO_Expressions.ipynb) Colab notebook contains all of the Python code necessary to train and test the YOLO model on my training and test datasets. You can follow the descriptions in the notebook to understand what each code cell does and if it needs something specifically from the user.

The [real_time_tracking](real_time_tracking.py) file is used to test the final model's real-time performance. I modified the one found in this github [repo](https://github.com/alihassanml/Yolo11-Face-Emotion-Detection) for my use. FPS and latency are recorded in this file and will be discussed later in [My Results](#my-results). I ran this Python code locally on my machine to capture real-time data using my webcam so you may need to make some modifications to file path for where the weights "best.pt" exist. Or you could just put the weights in the same directory as the [real_time_tracking](real_time_tracking.py) file.

My model's final weights are available in the following path:

[yolo_runs/yolo11n_run_final/weights](yolo_runs/yolo11n_run_final/weights)

# Dataset
<img align="right" width="221" height="450" alt="canvas" src="https://github.com/user-attachments/assets/38bd98e0-0367-43a5-bce1-eba4e15cfe53">

The dataset I use for this project is [“9 Facial Expressions for YOLO”](https://www.kaggle.com/datasets/aklimarimi/8-facial-expressions-for-yolo). I discovered this YOLO dataset while searching through Kaggle. Using this pre-made YOLO dataset eliminates the time-consuming step of collecting data, creating a labeling standard, and labeling the images myself. The dataset pre-organizes the folder structure as well so it's ready to be used after downloading it.

The dataset itself has images with facial expressions in a wide range of classes. The nine classes are as follows: angry, contempt, disgust, fear, happy, natural, sad, sleepy, and surprised. Examples of each class are provided on the right. This dataset is perfect for this project because it contains a wide spread of emotions that can be seen on people’s faces. As such they are very well-suited for training machine learning algorithms.

The images in the dataset are formatted as JPGs and the size/resolution of the images vary. However, this doesn’t matter because YOLO handles the resizing itself automatically during the training process. It resizes images of varying resolutions to a standard size by padding. Thus, anyone who uses this dataset does not have to worry about doing it themselves. 

There are 68,284 images in total split amongst the 3 different datasets (train, valid, test).

Dataset Split | Images (#) | Images (%)
--- | --- | --- 
Train | 64864 | 94.99%
Valid | 1720 | 2.52%
Test | 1700 | 2.49%
Total | 68284 | 100%

There is an overwhelming number of images for training but that’s how it should be for a large dataset. Most of the learning that the YOLO model does is during the training phase so obviously that’s where most of the images will go. A small amount is used for validation just so the model can test its prediction ability on separate unseen images and evaluate its performance so it can tune hyperparameters and prevent overfitting. The test set has 1700 images (2.49% of the total). This amount is good enough for the testing because it is large enough to show and measure the performance of my model. There’s no point in the test set having a larger slice because it’s not even learning at this point. The training, validation, and test set images are unique meaning images do not appear in each other’s datasets. Thus, it is a fair test and the performance results are not skewed due to repeat images.

The entire dataset was collected from multiple open-source facial expression datasets hosted on Roboflow. According to the author of the dataset, duplicates and corrupted files were removed during preprocessing. Images included in the datasets are sourced from real-world images, synthetic data, or annotated frames from videos. The creator also warns that although the dataset is diverse there are some class imbalances. Expressions like contempt or sleepy appear less frequently in the dataset than common expressions such as happy or sad. This imbalance is inevitable because we see some emotions more often than others in real life. There will be some performance hits between the different classes due to this fact. I expect the common expressions to perform better.

<img width="1780" height="446" alt="image" src="https://github.com/user-attachments/assets/8d9e9b82-413f-432f-b242-934f436ccce2" />

# Work Flow
I started this project training two differents YOLO nano models (v8 & v11) to see which one performed better. I chose the nano models for this project because they perform the best for real-time and my dataset was not large enough to really justify larger models. YOLO provides some pre-trained models on the MS COCO dataset that I started with because it transfers some of the knowledge gained from the over 330,000 images trained there. My model has some foundation to begin training on my much smaller dataset. For example, the pre-training allows my model to already understand what a person is because that's one of the classes in the MS COCO dataset.

The training was done in the [YOLO_Expressions](YOLO_Expressions.ipynb) notebook. It contains the last run I did using my final parameters and model but some of the parameters I changed constantly during training were the number of epochs, image size, and batch size. I left mostly all the other parameters that could optimize the training like the optimizer or learning rate default because I'll admit that I don't have the understanding necessary to do better than what YOLO chooses automatically.

When evaluating results for all the models I trained I was looking at the performance of the confusion matrices, recall curve, precision curve, and f1 score curves. They all performed relatively similarly but I ended up using YOLO v11 because it is preferable over YOLO v8 in real-time tracking. 

During real-time tracking I decided not to run it in Colab because accessing my webcam is easier if I run [real_time_tracking](real_time_tracking.py) locally on my machine. I used OpenCV to access my webcam and Ultralytics to run my final YOLO model using the weights produced during training. I used a confidence of 0.4 for my model's detection because I got the best results for most of my classes in that range. I tried different ranges of confidence but higher confidence would only really show 'happy' and lower confidence would confuse classes a lot more often.

# Competitive Results
There was a recent IEEE paper ["Facial Expression Recognition with YOLOv11 and YOLOv12: A Comparative Study"](https://ieeexplore.ieee.org/abstract/document/11279248) from 2025 that reported good performance metrics I use as a comparison for my results. They provide results for YOLOv11n and YOLOv12n on two different datasets (FER2013 & KDEF) but I will only discuss their results using YOLOv11n because my final model ended up using YOLOv11n as well. It just makes it easier to compare results using the same model. One thing to keep in mind is that my results will report on two more classes (contempt and sleepy) than the ones here.

One of the easiest metrics they measured and reported were their confusion matrices provided below. A confusion matrix just gives you a visually easy method to identify True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) by comparing the prediction labels with the true labels. A perfect confusion matrix would have all values on the main diagonal squares. They have relatively good results in the left confusion matrix. The dark colors in the diagonals show a good amount of correct predictions for their classes. However, there is a reason why the left matrix performs so well that could be a detriment for actual implementation (Hint: Limited KDEF dataset). On the other hand, the right matrix does not perform as well. It has a good number of incorrect predictions for both false positives and false negatives.

<p align="center">
  <img width="808" height="385" alt="image" src="https://github.com/user-attachments/assets/edb79f17-db35-4ebf-998b-19363c990e1e" />
</p>

The table down below provides their reported performance metrics such as Precision, Recall, and Mean Average Precision for YOLOv11n. Again, the models that train on the KDEF dataset perform well but there's an issue with taking those results at face value. The performance for the dataset FER2013 does worse than KDEF in all metrics.

Model | Dataset | Precision% | Recall% | mAP@0.5%
--- | --- | --- | --- |---
YOLOv11n | FER2013 | 65.2 | 60.5 | 60.8
YOLOv11n | KDEF | 87.7 | 91.1 | 94.5

Their paper shows very strong results for the KDEF dataset but they are very misleading. The problem with KDEF is that it's a very small dataset (4,900 samples) and the facial expressions were collected while having as perfect conditions as possible. What I mean by that is the data was collected in a studio environment from very limited variation in poses and perfect lighting. That's why it is not as well-suited for real-life use. It performs very well on their test dataset but it will encounter a lot more problems trying to detect facial expressions from more diverse environments. The dataset FER2013 performs poorly compared to KDEF because it just has a worse collection of data even if it has a much larger dataset (35,887 samples). A very important component missing from the paper is that it does not include any results for real-time implemention of their tracking model so we don't have any accuracy or latency performance to grade their models.

# My Results
I include multiple training and testing results in [yolo_runs](yolo_runs) but the results from my best model are provided in [yolo_runs/yolo11n_test_final](yolo_runs/yolo11n_test_final). I only discuss the performance results from my best model's run on the test dataset.

The confusion matrix on the left reports the raw numbers while the confusion matrix on the right reports the normalized values. I will only compare to the IEEE paper using my normalized values since that's what their confusion matrices report on. My final model shows very good results for all classes. As you can see, most of the dark colors land on the main diagonal squares. As expected, the best performing class ended up being "happy" because it shows up the most out of any other class in my dataset. My confusion matrix shows better results than the confusion matrix for the FER2013 dataset. For example, disgust performs way better in my model using my dataset than on FER2013. Even on the KDEF dataset, my confusion matrix has very comparable results on all similar classes which was very unexpected.
<p align="center">
  <img width="5322" height="2148" alt="output" src="https://github.com/user-attachments/assets/c42d7892-c3ea-4312-bc4f-314126b9c1e3" />
</p>

The table down below provides my reported performance metrics such as Precision, Recall, and Mean Average Precision for YOLOv11n. The data was collected from the last code cell's output summary in the [YOLO_Expressions](YOLO_Expressions.ipynb) notebook. My model performs relatively well. Most of my results reported 80-88% for all three metrics. Most of FER2013's results reported 60-65% while KDEF's results reported 87-94% for all three metrics. My model performs much better than FER2013 and a little worse for KDEF which was expected. I was aiming for better scores than their FER2013 dataset results which I achieved.

Model | Dataset | Precision% | Recall% | mAP@0.5%
--- | --- | --- | --- |---
YOLOv11n | Kaggle Dataset | 81.7 | 80.9 | 88.9

The real-time performance metrics I measured were FPS and latency. This is something that the IEEE paper did not include in any of their findings because they did not implement their model for real-time object detection. The table down below provides my FPS and latency results for video capture without the model and with the model active.
Real-time Video | FPS | Latency
--- | --- | ---
No Model | 27-31 | 0 ns (reference)
YOLOv11n | 27-31 | 6-9 ns

The accuracy of the model's ability to capture facial expressions suffers compared to my test dataset results. I have to express myself very clearly to get it to recognize some facial expressions. For example, I have to open my mouth and point my lips downwards while closing my eyes to get it to detect sad. The model also has a really tough time differentiating between angry and disgust. It really likes disgust even though it has a smaller number of samples than angry in the training set. Contempt and fear rarely show at all and if it does it happens for maybe a couple of frames. Sometimes it will mistake every single facial expression I make with sleepy. The best performing expressions that don't get confused all that often were happy, natural, and surprised. 

# Future Work
Make accuracy improvements for the real-time detection. This may mean more samples, balanced dataset, better parameters during training, fewer classes or a combination of these changes. 

I did not have time to try a different non-YOLO model to compare performance to so that may be something I can try out. I would need to convert my YOLO dataset to a different format for whatever model I choose.

# References
1. Ali Hassan, "YOLOv11 Face Emotion Detection," 2025, GitHub repository. [Online]. Available: https://github.com/alihassanml/Yolo11-Face-Emotion-Detection
2. Aklima Akter Rimi, "9 Facial Expressions for YOLO," 2025, Kaggle dataset. [Online]. Available: https://www.kaggle.com/datasets/aklimarimi/8-facial-expressions-for-yolo
3. U. Aymon, N. S. Kamarudin and A. F. A. Nasir, "[Facial Expression Recognition with YOLOv11 and YOLOv12: A Comparative Study](https://ieeexplore.ieee.org/abstract/document/11279248)," 2025 IEEE 9th International Conference on Software Engineering & Computer Systems (ICSECS), Pekan, Pahang, Malaysia, 2025, pp. 18-23.
