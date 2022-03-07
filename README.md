# Detecting Blindness with Deep Learning

![sample](https://i.postimg.cc/dVjwCDr2/blindness.png)

## Project summary

Diabetic retinopathy (DR) is one of the leading causes of vision loss. Early detection and treatment are crucial steps towards preventing DR. This project considers DR detection as an ordinal classification task and aims at developing a deep learning model for predicting the severity of DR disease based on the patient's retina photograph.

The project has been completed within the scope of [Udacity ML Engineer Nanodegree](https://confirm.udacity.com/LMMJDA7C) program. We use data employed in the [APTOS 2019 Blindness Detection competition](https://www.kaggle.com/c/aptos2019-blindness-detection/data) on Kaggle.

A detailed walkthrough covering important project stages is available in [this blog post](https://kozodoi.me/python/deep%20learning/computer%20vision/competitions/2020/07/11/blindness-detection.html). File `report.pdf` also contains a detailed PDF description of the project pipeline.


## Project structure

The project has the following structure:
- `codes/`: codes with modules and functions implementing preprocessing, datasets, model and utilities.
- `notebooks/`: notebooks covering different project stages: data preparation, modeling and ensembling.
- `efficientnet-pytorch/`: module with EfficientNet weights pre-trained on ImageNet. The weights are not included due to the size constraints and can be downloaded from [here](https://www.kaggle.com/hmendonca/efficientnet-pytorch).
- `figures/`: figures exported from the notebooks during the data preprocessing and training.
- `input/`: input data including the main data set and the supplementary data set. The images are not included due to size constraints. The main data set can be downloaded [here](https://www.kaggle.com/c/aptos2019-blindness-detection/data). The supplementary data is available [here](https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resized).
- `models/`: model weights saved during training.
- `submissions/`: predictions produced by the trained models.

There are four notebooks:
- `code_1_data_exploration.ipynb`: data exploration and visualization
- `code_2_pre_training.ipynb`: pre-training the CNN model on the supplementary 2015 data set
- `code_3_training.ipynb`: fine-tuning the CNN model on the main 2019 data set
- `code_4_inference.ipynb`: classifying test images with the trained model

More details are provided within the notebooks as well as in the `report.pdf` file.


## Requirements

To run the project codes, you can create a new virtual environment in `conda`:

```
conda create -n aptos python=3.7
conda activate aptos
```

and then install the requirements:

- pytorch (torch, torchvision)
- efficientnet-pytorch (pre-trained model weights)
- cv2 (image preprocessing library)
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- tqdm
