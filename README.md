# Different-techniques-to-deal-with-data-imbalance
This is part of the main "human behavior" analysis project.

The Datasets used for this project is confidential, so its have not displayed anywhere, to give an overview dataset contains around 2100 training, 400 validation and 300 test images with labels, and resized to 128*128*3. Class distribution is heavily imbalanced such as 0th class has more than 90% possessions. To address this problem, several tricks are tried as explained below. I have used f1 measure to calculate accuracy

1. Providing different weights to different class i.e {0: 1, 1: 50}, its consider every 1s sample to 50 samples., to compute class weighs I used sklearn compute_class_weight('balanced') and then passed new weights to model.fit()

2. I randomly copied and oversampled  minority class sample to make class distribution equal

3. I used SMOTE(Synthetic Minority Over-sampling Technique) to oversample minority class, its better than just copying minority class samples because it smartly creates synthesis samples (https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html)

4. I also tried some data augmentation and transfer learning since I have small datasets. 

In the end, I got some good results with smote and transfer learning. but you can try all methods, this issue is very subjective to data.
