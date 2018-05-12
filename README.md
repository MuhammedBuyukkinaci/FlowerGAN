# FlowerGAN
FlowerGAN : A DCGAN implementation in TensorFlow in Python 3 on Caltech-UCSD Birds dataset.

# Dependencies

```pip install -r requirements.txt```

or

```pip3 install -r requirements.txt```

# Training

```git clone https://github.com/MuhammedBuyukkinaci/FlowerGAN.git```

```cd FlowerGAN```

```python FlowerGAN.py ```

# Jupyter Notebook

```jupyter lab ``` or ```jupyter notebook ```

# Data
No MNIST or CIFAR-10. 

This is a repository containing datasets of 5200 training images of 4 classes and 1267 testing images.No problematic image.

Download .tgz extension version from [here](
http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) or .npy extension version from [here](
https://www.dropbox.com/s/20jkiactn0k5sss/multiclass_datasets_zip.zip?dl=0). It is 67 MB.

Extract files from multiclass_datasets.rar. Then put it in TensorFlow-Multiclass-Image-Classification-using-CNN-s folder.
train_data_bi.npy is containing training photos with labels.

test_data_bi.npy is containing 1267 testing photos with labels.

Classes are chair & kitchen & knife & saucepan.

Classes are equal(1300 glass - 1300 kitchen - 1300 knife- 1300 saucepan) on training data. 

# GPU
I trained on GTX 1050. 1 epoch lasted 7-8 minutes approximately.

If you don't want to wait for a month, use a GPU.

# Architecture
Images are resized to (64,64,3) .

![alt text](https://cdn-images-1.medium.com/max/1800/1*JnUzBXe5Zq-HT--iNKrCuQ.png) 

# Generated Photos
Predictions for first 64 testing images are below. Titles are  the predictions of our Model.

![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Multiclass-Image-Classification-using-CNN-s/blob/master/mc_preds.png)
