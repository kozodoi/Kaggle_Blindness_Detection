# Detecting Blindness with Deep Learning

## Project summary

Diabetic retinopathy (DR) is one of the leading causes of vision loss. Early detection and treatment are crucial steps towards preventing DR. This project onsiders DR detection as an ordinal classification task and aims at developing a deep learning model for predicting the severity of DR disease based on the patient's retina photograph. 

The project uses a data set that has been employed in the APTOS 2019 Blindness Detection competition on Kaggle and is available for the download at the [competition website](https://www.kaggle.com/c/aptos2019-blindness-detection/data).

File `report.pdf` contains a detailed description of the project pipeline.


## Project structure

The project has the follwoing structure:
- `codes/`: jupyter notebooks with codes for different project stages: data preparation, modeling and ensembling.
- `efficientnet-pytorch/`: module with EfficientNet weights pre-trained on ImageNet. The weights are not included due to the size constraints and can be downloaded from [here](https://www.kaggle.com/hmendonca/efficientnet-pytorch).
- `figures/`: figures exported from the jupyter notebooks during the data preprocessing and training.
- `input/`: input data including the main data set and the supplementary data set. The images are not included due to size constraints. The main data set can be downloaded [here](https://www.kaggle.com/c/aptos2019-blindness-detection/data). The supplementary data is available [here](https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resized).
- `models/`: model weights saved during training.
- `submissions/`: predictions produced by the trained models.

There are four notebooks:
- `code_1_data_exploration.ipynb`: data exploration and visuzlization.
- `code_2_pre_training.ipynb`: pre-training the CNN model on the supplementary 2015 data set.
- `code_3_training.ipynb`: fine-tuning the CNN model on the main 2019 data set.
- `code_4_inference.ipynb`: classifying test images with the trained model.

More details are provided within the notebooks.


## Requirments

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
