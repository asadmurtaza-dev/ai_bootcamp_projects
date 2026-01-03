# Deep Learning Projects Collection

This repository contains three deep learning projects demonstrating real-world applications in image classification, transfer learning, and text classification (NLP).

---

## Project 1: Cat vs Dog Image Classifier (CNN)

**Level:** Beginner – Intermediate  
**Dataset:** Kaggle Cats vs Dogs  
**Task Type:** Binary Image Classification  

**Overview:**  
Classifies images as cat or dog using a Convolutional Neural Network.  

**Steps / Approach:**  
- Load and preprocess images (rescale and resize).  
- Build a CNN model with convolution, pooling, flatten, and dense layers.  
- Train and validate the model using training and testing datasets.  

**Expected Results:**  
- Accuracy around 85–90%.  
- Can predict unseen cat and dog images.  

**Applications:**  
- Pet recognition apps.  
- Animal image classification systems.

---

## Project 2: Flower Species Classification (Transfer Learning)

**Level:** Intermediate  
**Dataset:** TensorFlow Flowers Dataset  
**Task Type:** Multi-class Image Classification  

**Overview:**  
Classifies flowers into different species using pre-trained MobileNetV2 for transfer learning.  

**Steps / Approach:**  
- Load dataset and resize images to 224x224.  
- Use MobileNetV2 as the base model and add pooling and dense layers.  
- Train and validate the model.  

**Expected Results:**  
- Accuracy around 85–88%.  
- Can classify new flower images accurately.  

**Applications:**  
- Mobile apps for plant identification.  
- Botanical research tools.

---

## Project 3: Spam Email / SMS Detection (NLP)

**Level:** Beginner – Intermediate  
**Dataset:** SMS Spam Collection  
**Task Type:** Text Classification (Binary)  

**Overview:**  
Detects whether a message is spam or ham using TF-IDF vectorization and Naive Bayes classifier.  

**Steps / Approach:**  
- Load and clean dataset.  
- Map labels (ham = 0, spam = 1).  
- Split dataset into training and testing sets.  
- Transform text using TF-IDF and train Naive Bayes classifier.  
- Evaluate performance using accuracy.  

**Expected Results:**  
- Accuracy around 97–98%.  
- Reliable spam detection for unseen messages.  

**Applications:**  
- Email spam filters.  
- SMS spam detection systems.  
- Social media monitoring tools.

---


## How to Use

1. Clone this repository.  
2. Navigate to each project folder.  
3. Run the Python scripts or notebooks.  
4. Make sure datasets are downloaded and paths updated.

---

## License

Educational / Training Purposes
