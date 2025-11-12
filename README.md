# **Deep Learning Specialization (Coursera | Stanford)**

Welcome to my repository for the Deep Learning Specialization\! As a follow-up to the Machine Learning Specialization, this repo contains all my coursework, labs, and projects from the five courses in this program taught by Andrew Ng.

This portfolio demonstrates my practical understanding of building and optimizing a wide range of neural network architectures, from foundational NNs to modern Transformer models.

## **Specialization Overview**

This specialization is a five-course program that covers the foundations of Deep Learning, how to build and optimize neural networks, and how to apply them to various domains like computer vision and natural language processing.
- Build and train deep neural networks, identify key architecture parameters, implement vectorized neural networks and deep learning to applications
- Train test sets, analyze variance for DL applications, use standard techniques and optimization algorithms, and build neural networks in TensorFlow
- Build a CNN and apply it to detection and recognition tasks, use neural style transfer to generate art, and apply algorithms to image and video data
- Build and train RNNs, work with NLP and Word Embeddings, and use HuggingFace tokenizers and transformer models to perform NER and Question Answering

1. **Course 1: Neural Networks and Deep Learning**  
   * Foundations of deep learning, building a neural network from scratch.  
   In the first course of the Deep Learning Specialization, you will study the foundational concept of neural networks and deep learning. 
   * By the end, you will be familiar with the significant technological trends driving the rise of deep learning; build, train, and apply fully connected deep neural networks; implement efficient (vectorized) neural networks; identify key parameters in a neural network’s architecture; and apply deep learning to your own applications. The Deep Learning Specialization is our foundational program that will help you understand the capabilities, challenges, and consequences of deep learning and prepare you to participate in the development of leading-edge AI technology. It provides a pathway for you to gain the knowledge and skills to apply machine learning to your work, level up your technical career, and take the definitive step in the world of AI.
2. **Course 2: Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization**  
   * Regularization (L2, Dropout), optimization (Adam, RMSProp), and hyperparameter tuning.  
   * In the second course of the Deep Learning Specialization, you will open the deep learning black box to understand the processes that drive performance and generate good results systematically. By the end, you will learn the best practices to train and develop test sets and analyze bias/variance for building deep learning applications; be able to use standard neural network techniques such as initialization, L2 and dropout regularization, hyperparameter tuning, batch normalization, and gradient checking; implement and apply a variety of optimization algorithms, such as mini-batch gradient descent, Momentum, RMSprop and Adam, and check for their convergence; and implement a neural network in TensorFlow. The Deep Learning Specialization is our foundational program that will help you understand the capabilities, challenges, and consequences of deep learning and prepare you to participate in the development of leading-edge AI technology. It provides a pathway for you to gain the knowledge and skills to apply machine learning to your work, level up your technical career, and take the definitive step in the world of AI.
3. **Course 3: Structuring Machine Learning Projects**  
   * The strategy of ML development, error analysis, and end-to-end project management.
   * In the third course of the Deep Learning Specialization, you will learn how to build a successful machine learning project and get to practice decision-making as a machine learning project leader. By the end, you will be able to diagnose errors in a machine learning system; prioritize strategies for reducing errors; understand complex ML settings, such as mismatched training/test sets, and comparing to and/or surpassing human-level performance; and apply end-to-end learning, transfer learning, and multi-task learning. This is also a standalone course for learners who have basic machine learning knowledge. This course draws on Andrew Ng’s experience building and shipping many deep learning products. If you aspire to become a technical leader who can set the direction for an AI team, this course provides the "industry experience" that you might otherwise get only after years of ML work experience. The Deep Learning Specialization is our foundational program that will help you understand the capabilities, challenges, and consequences of deep learning and prepare you to participate in the development of leading-edge AI technology. It provides a pathway for you to gain the knowledge and skills to apply machine learning to your work, level up your technical career, and take the definitive step in the world of AI.  
4. **Course 4: Convolutional Neural Networks (CNNs)**  
   * Building CNNs, object detection (YOLO), facial recognition (Siamese Networks), and neural style transfer.  
   * In the fourth course of the Deep Learning Specialization, you will understand how computer vision has evolved and become familiar with its exciting applications such as autonomous driving, face recognition, reading radiology images, and more. By the end, you will be able to build a convolutional neural network, including recent variations such as residual networks; apply convolutional networks to visual detection and recognition tasks; and use neural style transfer to generate art and apply these algorithms to a variety of image, video, and other 2D or 3D data. The Deep Learning Specialization is our foundational program that will help you understand the capabilities, challenges, and consequences of deep learning and prepare you to participate in the development of leading-edge AI technology. It provides a pathway for you to gain the knowledge and skills to apply machine learning to your work, level up your technical career, and take the definitive step in the world of AI.
5. **Course 5: Sequence Models (RNNs, LSTMs)**  
   * Recurrent Neural Networks (RNNs), LSTMs, GRUs, Attention Models, and Transformers.
   * In the fifth course of the Deep Learning Specialization, you will become familiar with sequence models and their exciting applications such as speech recognition, music synthesis, chatbots, machine translation, natural language processing (NLP), and more. By the end, you will be able to build and train Recurrent Neural Networks (RNNs) and commonly-used variants such as GRUs and LSTMs; apply RNNs to Character-level Language Modeling; gain experience with natural language processing and Word Embeddings; and use HuggingFace tokenizers and transformer models to solve different NLP tasks such as NER and Question Answering. The Deep Learning Specialization is a foundational program that will help you understand the capabilities, challenges, and consequences of deep learning and prepare you to participate in the development of leading-edge AI technology. It provides a pathway for you to take the definitive step in the world of AI by helping you gain the knowledge and skills to level up your career.

## **Repository Structure**

This repository is organized into folders corresponding to each of the five courses, with labs and projects organized by week.

* /C1 \- Neural Networks and Deep Learning/: Contains all labs and assignments for Course 1, including building a NN from scratch.  
* /C2 \- Improving Deep Neural Networks Hyperparameter Tuning, Regularization and Optimization/: Contains labs for Course 2, focusing on optimization, batch norm, and TensorFlow.  
* /C3 \- Structuring Machine Learning Projects/: Contains labs and case studies for Course 3 on ML strategy and error analysis.  
* /C4 \- Convolutional Neural Networks/: Contains all labs for Course 4, from basic convnets to ResNets and U-Nets.  
* /C5 \- Sequence Models/: Contains labs for Course 5, covering RNNs, LSTMs, and Attention.  
* deeplearning.mplstyle: Root-level helper file for Matplotlib styling used in the notebooks.  
* utils.py: Root-level helper file with utility functions used across various labs.

## **Key Concepts & Highlights**

Here are some of the key topics from the specialization, along with (example) links to the relevant work in this repo.

### **1\. Building a Neural Network from Scratch**

* **File:** [C1.../Week 4/C1W4\_Assignment\_NN\_from\_Scratch.ipynb\](./C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%204/C1W4\_Assignment\_NN\_from\_Scratch.ipynb)  
* **Description:** Implemented a multi-layer neural network from scratch using only NumPy, including forward and backward propagation. This demonstrates a deep understanding of the core mechanics.

### **2\. Optimization & Regularization**

* **File:** [C2.../Week 1/C2W1\_Lab\_Regularization\_and\_Dropout.ipynb\](./C2%20-%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuning,%20Regularization%20and%20Optimization/Week%201/C2W1\_Lab\_Regularization\_and\_Dropout.ipynb)  
* **Description:** Implemented L2 Regularization and Dropout to prevent overfitting. Also explored optimization algorithms like Adam and RMSProp to train models faster and more effectively.

### **3\. ML Project Strategy & Error Analysis**

* **File:** [C3.../Week 1/C3W1\_Lab\_Error\_Analysis.ipynb\](./C3%20-%20Structuring%20Machine%20Learning%20Projects/Week%201/C3W1\_Lab\_Error\_Analysis.ipynb)  
* **Description:** A crucial part of ML. This involves analyzing misclassified examples from a dev set to prioritize the most effective next steps (e.g., collecting more data, improving the algorithm, or cleaning labels).

### **4\. Convolutional Neural Networks (ResNets)**

* **File:** [C4.../Week 2/C4W2\_Assignment\_ResNets.ipynb\](./C4%20-%20Convolutional%20Neural%20Networks/Week%202/C4W2\_Assignment\_ResNets.ipynb)  
* **Description:** Built and trained a Residual Network (ResNet), a state-of-the-art architecture. This project shows proficiency in building deep, complex models for computer vision tasks.

### **5\. Sequence Models with Attention**

* **File:** [C5.../Week 3/C5W3\_Assignment\_Machine\_Translation.ipynb\](./C5%20-%20Sequence%20Models/Week%203/C5W3\_Assignment\_Machine\_Translation.ipynb)  
* **Description:** Implemented an Attention-based model for machine translation. This demonstrates an understanding of modern NLP architectures (the foundation for Transformers) and how to handle sequence-to-sequence tasks.

## **Dependencies**

* The lab notebooks (.ipynb) run in a Jupyter environment.  
* Core Libraries: TensorFlow (Keras), NumPy, Pandas, Matplotlib, Scikit-Learn.  
* The deeplearning.mplstyle and utils.py files are helper files provided by the course and are expected to be in the root directory.

## **Connect with Me**

I'm passionate about building effective and scalable AI solutions. Let's connect\!

* **LinkedIn:** [Youssef Maaod](https://www.linkedin.com/in/youssef-maaod-989a17257?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3Bkd9xJ7P5QkCSKhtU3L57VA%3D%3D)
* **GitHub:** [yuseiff](https://github.com/yuseiff)
