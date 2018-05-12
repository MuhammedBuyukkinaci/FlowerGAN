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
https://www.dropbox.com/s/gmu2cxgjktcnw40/flower_photos.npy?dl=0). It is about 95 MB.

If you downloaded the dataset, extract files from 102flowers.tgz . Then put it in FlowerGAN folder.

If you download .npy file from Dropbox, put flower_photos.npy in FlowerGAN folder.

# GPU
I trained on GTX 1050. 1 epoch lasted 7-8 minutes. I left my laptop overnight and obtained outputs.

If you don't want to wait for one month, use a GPU.

# Architecture
Images are resized to (64,64,3) .

![alt text](https://cdn-images-1.medium.com/max/1800/1*JnUzBXe5Zq-HT--iNKrCuQ.png) 

# Generated Photos
Predictions for first 64 testing images are below. Each picture contains 16 generated photos.

![alt text](https://github.com/MuhammedBuyukkinaci/FlowerGAN/blob/master/generated_photos.gif)
