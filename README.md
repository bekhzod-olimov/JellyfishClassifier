# JellyfishClassifier

This repository contains a deep learning (DL)-based artificial intelligence (AI) image classification model training to classify jellyfish types. The AI model used for the classification task is RexNet ([paper](https://arxiv.org/pdf/2007.00992.pdf) and [code](https://github.com/clovaai/rexnet)) and the dataset for training is [Jellyfish Image Dataset](https://www.kaggle.com/datasets/anshtanwar/jellyfish-types). The project in [Kaggle](https://www.kaggle.com/) can be found [here](https://www.kaggle.com/code/killa92/pytorch-classification-100-test-accuracy).

# Manual on how to use the repo:

1. Clone the repo to your local machine using terminal via the following script:

```python
git clone https://github.com/bekhzod-olimov/JellyfishClassifier.git
```

2. Create conda environment from yml file using the following script:
```python
conda env create -f environment.yml
```
Then activate the environment using the following command:
```python
conda activate speed
```

3. Data Visualization

![image](https://github.com/bekhzod-olimov/JellyfishClassifier/assets/50166164/9b795a3a-b9b6-454a-95af-0581f9ae2cd3)

4. Train the AI model using the following script:
```python
python main.py --root PATH_TO_THE_DATA --batch_size = 64 device = "cuda:0"
```
The training parameters can be changed using the following information:

![image](https://github.com/bekhzod-olimov/JellyfishClassifier/assets/50166164/cc82512f-acc1-4762-9858-f7b870fd8637)

The training process progress:

![image](https://github.com/bekhzod-olimov/JellyfishClassifier/assets/50166164/ca68a5af-bef1-4b41-9412-648c2f4942c3)

5. Learning curves:

* Train and validation loss curves:
![loss_learning_curves](https://github.com/bekhzod-olimov/JellyfishClassifier/assets/50166164/ab064c7a-39c7-412d-a353-8f6c723a6ea0)

* Train and validation accuracy curves:
![acc_learning_curves](https://github.com/bekhzod-olimov/JellyfishClassifier/assets/50166164/7493c4f3-fc18-443e-8002-5bcd62a12b55)

7. Conduct inference using the trained model:
```python
python inference.py --root PATH_TO_THE_DATA --batch_size = 64 device = "cuda:0"
```

The inference progress:

![image](https://github.com/bekhzod-olimov/JellyfishClassifier/assets/50166164/8fee0d75-c43c-4b85-9fcd-9a285a4cdf4a)



