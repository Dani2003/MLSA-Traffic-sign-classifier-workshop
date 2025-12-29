# MLSA Workshop: Traffic Sign Classifier

Build an AI-powered traffic sign recognition system using Convolutional Neural Networks. This workshop guides you through creating a model that identifies 43 different types of traffic signs from the German Traffic Sign Recognition Benchmark dataset.

## Workshop Information

**Microsoft Learn Student Ambassadors Event**
Date: December 29, 2024
Time: 7:00 PM

This hands-on workshop is designed for students interested in artificial intelligence, computer vision, and machine learning. No prior deep learning experience required.

---

## Prerequisites: Learn Computer Vision Fundamentals

Before starting this workshop, complete this Microsoft Learn module to build your foundation:

**Required Module:**
[Introduction to Computer Vision - Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/introduction-computer-vision/?wt.mc_id=studentamb_264805)
[Introduction to Machine Learning - Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/fundamentals-machine-learning/?wt.mc_id=studentamb_264805)

Duration: 15-20 minutes
Level: Beginner-friendly

This module covers essential concepts you'll apply in this workshop:
- What is computer vision
- What is Machine Learning
- Image classification fundamentals
- How neural networks process images
- Real-world computer vision applications

---

## Additional Resources

**CS50 AI Project Reference:**
For detailed specifications and theoretical background:
[CS50 AI - Traffic Project](https://cs50.harvard.edu/ai/projects/5/traffic/)

---

## Getting Started

### Step 0: System Requirements

- Python 3.12 or lower (required for TensorFlow compatibility)
- Git installed on your machine
- Code editor (VS Code, PyCharm, etc.)
- At least 2GB free disk space for the dataset

### Step 1: Clone or Fork This Repository

**Option A - Clone directly:**
```bash
git clone https://github.com/YOUR_USERNAME/mlsa-traffic-sign-classifier-workshop.git
cd mlsa-traffic-sign-classifier-workshop
```

**Option B - Fork first (recommended for saving your progress):**
1. Click the "Fork" button at the top right
2. Clone your forked repository:
```bash
git clone https://github.com/YOUR_USERNAME/mlsa-traffic-sign-classifier-workshop.git
cd mlsa-traffic-sign-classifier-workshop
```

### Step 2: Download Project Starter Files

Download and extract the starter code:

**Using curl (command line):**
```bash
curl -o traffic.zip https://cdn.cs50.net/ai/2023/x/projects/5/traffic.zip
unzip traffic.zip
```

**Manual download:**
Download from: https://cdn.cs50.net/ai/2023/x/projects/5/traffic.zip
Extract the contents into your project directory.

### Step 3: Download the GTSRB Dataset

Download and extract the dataset (approximately 300MB):

**Using curl (command line):**
```bash
curl -o gtsrb.zip https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip
unzip gtsrb.zip
```

**Manual download:**
Download from: https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip
Extract into your project directory.

**Verify your directory structure:**
```
mlsa-traffic-sign-classifier-workshop/
├── gtsrb/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── ...
│   └── 42/
├── traffic.py
├── requirements.txt
└── README.md
```

### Step 4: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 5: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- TensorFlow (neural network framework)
- OpenCV (image processing)
- scikit-learn (data splitting utilities)
- NumPy (numerical operations)

### Step 6: Understand the Code Structure

Open `traffic.py` in your editor and examine the three main functions:

**load_data(data_dir)**
- Loads images from the dataset
- Resizes images to 30x30 pixels
- Returns images and their corresponding labels

**get_model()**
- Defines the Convolutional Neural Network architecture
- Includes layers: Conv2D, MaxPooling, Dense, Dropout
- Returns a compiled model ready for training

**main()**
- Orchestrates the entire workflow
- Splits data into training and testing sets
- Trains the model
- Evaluates performance
- Optionally saves the trained model

Take time to understand:
- How images are preprocessed
- The CNN layer structure
- The training process
- How accuracy is measured

### Step 7: Run Your First Training

**Basic training:**
```bash
python traffic.py gtsrb
```

**Training with model save:**
```bash
python traffic.py gtsrb model.h5
```

The program will:
1. Load all images from the dataset
2. Split into training (60%) and testing (40%) sets
3. Train for 10 epochs
4. Display accuracy and loss metrics
5. Save the model (if specified)

---

## Understanding the Dataset

**German Traffic Sign Recognition Benchmark (GTSRB)**

- Total categories: 43 different traffic sign types
- Total images: Over 50,000
- Image characteristics: Real-world photos with varying lighting, angles, and weather conditions
- Processing: All images resized to 30x30 pixels for uniform input

**Category examples:**
- 0: Speed limit (20km/h)
- 1: Speed limit (30km/h)
- 13: Yield
- 14: Stop
- 17: No entry
- And 38 more categories...

---

## Workshop Learning Objectives

By completing this workshop, you will:

1. Understand image preprocessing techniques
2. Build a Convolutional Neural Network from scratch
3. Train a deep learning model on real-world data
4. Evaluate model performance using accuracy metrics
5. Apply computer vision concepts to practical problems
6. Gain hands-on experience with TensorFlow and Keras

---

## Troubleshooting Common Issues

**Issue: "ValueError: zero-size array to reduction operation maximum"**

Solution:
- Verify the `gtsrb` folder is in the same directory as `traffic.py`
- Check that subdirectories (0 through 42) exist inside `gtsrb/`
- Ensure images are present in each subdirectory

**Issue: TensorFlow compatibility errors**

Solution:
- Confirm Python version is 3.12 or lower: `python --version`
- Reinstall TensorFlow: `pip install --upgrade tensorflow`
- If using Python 3.13+, downgrade to 3.11 or 3.12

**Issue: Out of memory during training**

Solution:
- Close other applications
- Reduce `EPOCHS` value in `traffic.py` (line 11)
- Consider using a smaller portion of the dataset initially

**Issue: Slow training performance**

Solution:
- This is normal without GPU acceleration
- Expected time: 10-30 minutes depending on your CPU
- Consider reducing epochs for initial testing

---

## Extending Your Learning

After completing the basic workshop:

1. Experiment with different CNN architectures
2. Adjust hyperparameters (learning rate, batch size, epochs)
3. Add more convolutional layers
4. Try different activation functions
5. Implement data augmentation
6. Compare performance metrics

---

## Additional Microsoft Learn Resources

Continue your AI and computer vision journey:

- [Get Started with AI on Azure](https://learn.microsoft.com/en-us/training/paths/get-started-with-artificial-intelligence-on-azure/?wt.mc_id=studentamb_264805)
- [Build AI Solutions with Azure Machine Learning](https://learn.microsoft.com/en-us/training/paths/build-ai-solutions-with-azure-ml-service/?wt.mc_id=studentamb_264805)
- [Create Computer Vision Solutions with Azure](https://learn.microsoft.com/en-us/training/paths/create-computer-vision-solutions-azure-cognitive-services/?wt.mc_id=studentamb_264805)

---

## Technical Stack

- Python 3.12 or lower
- TensorFlow 2.x
- Keras (included with TensorFlow)
- OpenCV for image processing
- NumPy for numerical operations
- scikit-learn for data splitting

---

## Further Reading

- [CS50 AI Course](https://cs50.harvard.edu/ai/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [Understanding CNNs](https://cs231n.github.io/)

---

## Workshop Feedback

We value your input! After completing the workshop:
- Share your model's accuracy
- Report any issues or bugs
- Suggest improvements to the materials
- Connect with other participants

---

## License

This project is based on CS50's educational materials. See CS50's License for details.

---

## Contact

**Microsoft Learn Student Ambassadors**
Workshop Lead: Abdul Bari
Event Date: December 29, 2024

For questions or support during the workshop, please reach out to the facilitator.

---

Built for MLSA Workshop Series | Computer Vision and Deep Learning
