# Alexnet on CIFAR-10 (PyTorch Implementation)

**The summary of the paper is posted on my [Blog](https://sidthoviti.com/summary-of-alexnet/)**

AlexNet was trained and tested on CIFAR-10 as a part of Recent Trends in Machine Learning (RTML) course.
The repository contains jupyter notebook as well as python files for the experiment.
- CIFAR-10 is split into 80% training and 20% validation.
- AlexNet has 57,044,810 trainable parameters in this implementation.
- Training Details:
  - Batch size=8
  - Optimization: SGD with learning rate=0.001, momentum=0.9.
  - Loss function: Cross Entropy Loss
  - Epochs: 10
  - The experiments compares the performance with and without Local Response Normalization (LRN). Training hyperparameters remain the same for the both cases.

## Results
- Without LRN, validation accuracy = 81.0%
- With LRN, validation accuracy = 81.01%

## Credits
Dr. Matthew N. Dailey (Course Instructor)

