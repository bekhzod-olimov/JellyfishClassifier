# JellyfishClassifier

This repository contains a deep learning (DL)-based artificial intelligence (AI) image classification model training to classify jellyfish types. The AI model used for the classification task is RexNet ([paper](https://arxiv.org/pdf/2007.00992.pdf) and [code](https://github.com/clovaai/rexnet)) and the dataset for training is [Jellyfish Image Dataset](https://www.kaggle.com/datasets/anshtanwar/jellyfish-types). The project in [Kaggle](https://www.kaggle.com/) can be found [here](https://www.kaggle.com/code/killa92/pytorch-classification-100-test-accuracy).

# Manual on how to use the repo:

1. Clone the repo to your local machine using terminal via the following script:

```python
git clone https://github.com/bekhzod-olimov/BrainTumorClassification.git
```

2. Create conda environment from yml file using the following script:
```python
conda env create -f environment.yml
```
Then activate the environment using the following command:
```python
conda activate speed
```

3. Train the AI model using the following script:
```python
python main.py --root PATH_TO_THE_DATA --batch_size = 64 device = "cuda:0"
```
The training parameters can be changed using the following information:

![image](https://github.com/bekhzod-olimov/CT-Brain-Tumor-Classification/assets/50166164/4d5f3ddb-ec9e-49f7-a821-cdb5c8e013ec)

The training process progress:

![image](https://github.com/bekhzod-olimov/CT-Brain-Tumor-Classification/assets/50166164/e03b857e-e80e-47bc-87a7-efecec6b3733)