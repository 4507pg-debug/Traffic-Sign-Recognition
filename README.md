# Traffic-Sign-Recognition
Traffic Sign Recognition using EfficientNetV2 

This project implements a high-accuracy Traffic Sign Classifier using Deep Learning. It utilizes Transfer Learning with the EfficientNetV2B0 architecture to classify traffic signs from images. The model is robust to variations in lighting, orientation, and size, achieving high performance even on imbalanced datasets.

Features

> This Uses EfficientNetV2B0 pretrained on ImageNet for superior feature extraction.

> Label Smoothing (0.1): Prevents overfitting by stopping the model from becoming too confident in its predictions.

> Class Weights: Automatically balances training to handle rare traffic signs.

> Fine-Tuning: Unfreezes top layers to adapt specific feature detectors for traffic signs.

> Data Augmentation: Randomly rotates, zooms, translates, and adjusts brightness of training images to improve generalization.

> Auto-Fix Dataset Loading: Automatically detects and handles nested folder structures (e.g., datasets/traffic_Data/classes).

> Detailed Reporting: Generates a comprehensive classification report (Precision, Recall, F1-Score) and Overall Accuracy metrics.

Properties
> Python 3.13

> Deep Learning Framework: TensorFlow / Keras

> Computer Vision: OpenCV

> Data Handling: Numpy, Scikit-Learn

> Visualization: Matplotlib

Dataset Structure

The code is designed to work with a dataset organized by folders, where each folder name is the class label:

Dataset/
├── Stop/
│   ├── img1.jpg
│   ├── img2.jpg
├── Yield/
│   ├── img1.jpg
│   └── ...
└── SpeedLimit60/
    └── ...

How to Run

1. Install Dependencies

pip install tensorflow opencv-python matplotlib scikit-learn


2. Configuration

Open traffic_sign_classifier.py and update the DATASET_PATH variable to point to your dataset folder (e.g., Google Drive path or local path).

3. Train the Model

Run the script to start training. The process includes an initial training phase followed by a fine-tuning phase.

python traffic_sign_classifier.py


4. Inference

The script includes a helper function upload_and_predict() (for Google Colab) to test the model on new images immediately after training.

Results

The model utilizes a two-phase training strategy:

Head Training: Trains only the top layers for 20 epochs.

Fine-Tuning: Unfreezes the top 50% of the base model and trains for another 20 epochs with a low learning rate (1e-5).

Metrics:

Loss Function: Categorical Crossentropy with Label Smoothing.

Optimizer: Adam (Phase 1) -> RMSprop (Phase 2).

License

This project is open-source and available under the MIT License.
