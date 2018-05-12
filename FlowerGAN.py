import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle
from tqdm import tqdm      # Cool for loops.
import tensorflow as tf # For graphical operations
import datetime

# our photos are in the size of IMG_SIZE,IMG_SIZE,3
IMG_SIZE = 64

MAIN_DIR = os.getcwd()

import os
import platform
if platform.system() == 'Windows':
    separator = '\\'
    TRAIN_DIR = os.getcwd() + separator + 'jpg'
else:
    separator = '/'
    TRAIN_DIR = os.getcwd() + separator + 'jpg'

#Changing directories to where images are.
os.chdir(TRAIN_DIR)

print(os.getcwd())

#Function for importing data from train directory.
def reading_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = TRAIN_DIR + separator + img
        img = cv2.imread(path,1)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append(np.array(img))
    shuffle(training_data)
    os.chdir(MAIN_DIR)
    np.save('flower_photos.npy', training_data)
    return training_data

os.chdir(MAIN_DIR)
#flower_data = reading_data()
flower_data = np.load('flower_photos.npy')

#For plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

#A flower figure
plt.imshow(np.array(flower_data[1451]))

#Reshaping
X = np.array([i for i in flower_data]).reshape(-1,IMG_SIZE,IMG_SIZE,3)

print(X.shape)

#Hyperparameters
bs = 16
lr = 0.0003
num_epoch = 100

#Resetting the graph
tf.reset_default_graph()

#Function for batch normalization
def batch_normalization(input, name='bn'):
    
    with tf.variable_scope(name):
        output_dim = input.get_shape()[-1]
        beta = tf.get_variable('BnBeta', [output_dim], initializer=tf.zeros_initializer())
        gamma = tf.get_variable('BnGamma', [output_dim], initializer=tf.ones_initializer())
    
        if len(input.get_shape()) == 2:
            mean, var = tf.nn.moments(input, [0])
        else:
            mean, var = tf.nn.moments(input, [0, 1, 2])
        return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-5)

#Discriminator		
def discriminator(images,reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        #CONVOLUTION LAYER 1
        d_w1 = tf.get_variable('d_w1', [5, 5, 3, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [128], initializer=tf.constant_initializer(1))
        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 2, 2, 1], padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.nn.leaky_relu(tf.layers.batch_normalization(d1, training=True))
        #CONVOLUTION LAYER 2
        d_w2 = tf.get_variable('d_w2', [5, 5, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [256], initializer=tf.constant_initializer(1))
        d2 = tf.nn.conv2d(input=d1, filter=d_w2 , strides=[1, 2, 2, 1], padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.nn.leaky_relu(tf.layers.batch_normalization(d2, training=True))
        #CONVOLUTION LAYER 3
        d_w3 = tf.get_variable('d_w3', [5, 5, 256, 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [512], initializer=tf.constant_initializer(1))
        d3 = tf.nn.conv2d(input=d2, filter=d_w3 , strides=[1, 2, 2, 1], padding='SAME')
        d3 = d3 + d_b3
        d3 = tf.nn.leaky_relu(tf.layers.batch_normalization(d3, training=True))
        #CONVOLUTION LAYER 4
        d_w4 = tf.get_variable('d_w4', [5, 5, 512, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1024], initializer=tf.constant_initializer(1))
        d4 = tf.nn.conv2d(input=d3, filter=d_w4 , strides=[1, 2, 2, 1], padding='SAME')
        d4 = d4 + d_b4
        d4 = tf.nn.leaky_relu(tf.layers.batch_normalization(d4, training=True))
        #CONVOLUTION LAYER 5
        d_w5 = tf.get_variable('d_w5', [4, 4, 1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b5 = tf.get_variable('d_b5', [1], initializer=tf.constant_initializer(1))
        d5 = tf.nn.conv2d(input=d4, filter=d_w5 , strides=[1, 1, 1, 1], padding='SAME')
        d5 = d5 + d_b5
        d5 = tf.nn.leaky_relu(d5)
        o = tf.nn.sigmoid(d5)
        return o
#Generator
def generator(input_x, batch_size):
    #DECONVOLUTION LAYER 1
    W1 = tf.get_variable('g_w1', [4, 4, 1024, input_x.get_shape()[-1]],
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
    b1 = tf.get_variable('g_b1', [1024], initializer=tf.constant_initializer(1))
    input_shape1 = input_x.get_shape().as_list()
    output_shape1 = [batch_size,int(input_shape1[1] * 1), int(input_shape1[2] * 1), 1024]
    d1 = tf.nn.conv2d_transpose(input_x, W1, output_shape=output_shape1, strides=[1, 1, 1, 1])
    d1 = d1 + b1
    d1 = tf.nn.leaky_relu(batch_normalization(d1,name='bn_1'))
    #print(d1)
    #DECONVOLUTION LAYER 2
    W2 = tf.get_variable('g_w2', [5, 5, 512, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b2 = tf.get_variable('g_b2', [512], initializer=tf.constant_initializer(1))
    input_shape2 = d1.get_shape().as_list()
    output_shape2 = [batch_size,int(input_shape2[1] * 2), int(input_shape2[2] * 2), 512]
    d2 = tf.nn.conv2d_transpose(d1, W2, output_shape=output_shape2, strides=[1, 2, 2, 1])
    d2 = d2 + b2
    d2 = tf.nn.leaky_relu(batch_normalization(d2,name='bn_2'))
    #print(d2)
    #DECONVOLUTION LAYER 3
    W3 = tf.get_variable('g_w3', [5, 5, 256, 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b3 = tf.get_variable('g_b3', [256], initializer=tf.constant_initializer(1))
    input_shape3 = d2.get_shape().as_list()
    output_shape3 = [batch_size,int(input_shape3[1] * 2), int(input_shape3[2] * 2), 256]
    d3 = tf.nn.conv2d_transpose(d2, W3, output_shape=output_shape3, strides=[1, 2, 2, 1])
    d3 = d3 + b3
    d3 = tf.nn.leaky_relu(batch_normalization(d3,name='bn_3'))
    #print(d3)
    #DECONVOLUTION LAYER 4
    W4 = tf.get_variable('g_w4', [5, 5, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b4 = tf.get_variable('g_b4', [128], initializer=tf.constant_initializer(1))
    input_shape4 = d3.get_shape().as_list()
    output_shape4 = [batch_size,int(input_shape4[1] * 2), int(input_shape4[2] * 2), 128]
    d4 = tf.nn.conv2d_transpose(d3, W4, output_shape = output_shape4, strides=[1, 2, 2, 1])
    d4 = d4 + b4
    d4 = tf.nn.leaky_relu(batch_normalization(d4,name='bn_4'))      
    #print(d4)
    #DECONVOLUTION LAYER 5
    W5 = tf.get_variable('g_w5', [5, 5, 3, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b5 = tf.get_variable('g_b5', [3], initializer=tf.constant_initializer(1))
    input_shape5 = d4.get_shape().as_list()
    output_shape5 = [batch_size,int(input_shape5[1] * 2), int(input_shape5[2] * 2), 3]
    d5 = tf.nn.conv2d_transpose(d4, W5, output_shape = output_shape5, strides=[1, 2, 2, 1])
    #print(d5)
    d5 = tf.nn.tanh(d5)
    return d5

#Defining a placeholder for z
z_dimensions = 100
z_placeholder = tf.placeholder(tf.float32, shape=(None, 4, 4, z_dimensions))	
		

generated_image_output = generator(z_placeholder, bs)
z_batch = np.random.normal(0, 1, [bs, 4,4,z_dimensions])

print(generated_image_output)

#Generating a random image
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    generated_image = sess.run(generated_image_output,
                                feed_dict={z_placeholder: z_batch})
    generated_image = generated_image[0].reshape([64, 64,3])
    plt.imshow(generated_image)

#Resetting graph
tf.reset_default_graph()

#Defining Placeholders
z_placeholder = tf.placeholder(tf.float32, [None, 4, 4 , z_dimensions], name='z_placeholder') 

# x_placeholder is for feeding input images to the discriminator
x_placeholder = tf.placeholder(tf.float32, shape = [None,64,64,3], name='x_placeholder') 

# Gz holds the generated images
Gz = generator(z_placeholder, bs)

# Dx will hold discriminator prediction probabilities
Dx = discriminator(x_placeholder)

# Dg will hold discriminator prediction probabilities for generated images
Dg = discriminator(Gz, reuse_variables=True)

#Loss functions
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))

#Variables
tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

print([v.name for v in d_vars])
print([v.name for v in g_vars])

# Train the discriminator
d_trainer_fake = tf.train.AdamOptimizer(learning_rate= lr).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(learning_rate= lr).minimize(d_loss_real, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(learning_rate= lr).minimize(g_loss, var_list=g_vars)

# Use reuse_variables
tf.get_variable_scope().reuse_variables()

#For summary
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

#TensorBoard
images_for_tensorboard = generator(z_placeholder, bs)
tf.summary.image('Generated_images', images_for_tensorboard, 5)
merged = tf.summary.merge_all()
logdir = "tensorboard_dcgan/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

#Remaining image due to division operation of total images % batch_size
remaining = X.shape[0] % bs

print(remaining)

#Pre-train the discriminator
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Pre-train discriminator
for i in range(0,len(X)-remaining,bs):
    z_batch = np.random.normal(0, 1, [bs, 4, 4, z_dimensions])
    real_image_batch = np.array(X[i:i+bs]).reshape([bs, 64, 64, 3])
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch})

    if(i % 1000 == 0):
        print(i, "dLossReal: ", dLossReal, "dLossFake: ", dLossFake)

generated_photos = []

#Multiple images plottings parameters
w=64
h=64
columns = 4
rows = 4

# Train generator and discriminator together
for i in range(num_epoch):
    for j in range(0,X.shape[0] - int(remaining) , bs):
        real_image_batch = np.array(X[j:j+bs]).reshape([bs, 64, 64, 3])
        z_batch = np.random.normal(0, 1, [bs, 4, 4, z_dimensions])

        # Train discriminator on both real and fake images
        _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                               {x_placeholder: real_image_batch, z_placeholder: z_batch})

        # Train generator
        z_batch = np.random.normal(0, 1, [bs, 4, 4, z_dimensions])
        _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})
    
    # Printing once in 2 epochs
    if i % 2 == 0:
        print(i + 1, "dLossReal: ", dLossReal, "dLossFake: ", dLossFake)
        z_batch = np.random.normal(0, 1, [bs, 4, 4, z_dimensions])
        summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
        writer.add_summary(summary, i)
        # Every 100 iterations, show a generated image
        z_batch = np.random.normal(0, 1, [bs, 4, 4, z_dimensions])
        generated_images = generator(z_placeholder, bs)
        images = sess.run(generated_images, {z_placeholder: z_batch})
        fig = plt.figure(figsize=(8, 8))
        for m in range(1, columns*rows +1):
            img = images[m-1].reshape([64, 64, 3])
            generated_photos.append(img)
            fig.add_subplot(rows, columns, m)
            plt.imshow(img)
        plt.show()

        # Show discriminator's estimate
        im = images.reshape([bs, 64 ,64, 3])
        result = discriminator(x_placeholder)
        estimate = sess.run(result, {x_placeholder: im})
    #generated_estimates.append(estimate)
np.save('generated_flower_photos.npy', generated_photos)

