#!/usr/bin/env python3
# import packages
import os
import myconfig as config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D,Activation,Input,Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
import numpy as np
import random
import tensorflow as tf

# create CNN model
input_img=Input(shape=(None,None,3))

log_img = tf.math.log1p(input_img)
x1u = 1 - log_img
#x1u=tf.math.log(input_img)

x2u=Conv2D(32,(3,3),padding="same")(x1u)

xu=Conv2D(32,(3,3),padding="same")(x2u)
xu=Activation('LeakyReLU')(xu)

xu=Conv2D(32,(3,3),padding="same")(xu)
xu=Activation('LeakyReLU')(xu)

x_com1u = Add()([x2u, xu])

x3u=Conv2D(32,(3,3),padding="same")(x_com1u)

xu=Conv2D(32,(3,3),padding="same")(x3u)
x41u=Activation('LeakyReLU')(xu)

xu=Conv2D(32,(3,3),padding="same")(x3u)
x42u=Activation('LeakyReLU')(xu)

xu=Conv2D(32,(3,3),padding="same")(x3u)
x43u=Activation('LeakyReLU')(xu)

x_com2u = Add()([x41u, x42u])
x_com3u = Add()([x41u, x43u])
x_com4u = Add()([x42u, x43u])

xu=Conv2D(32,(3,3),padding="same")(x_com2u)
x51u=Activation('LeakyReLU')(xu)

xu=Conv2D(32,(3,3),padding="same")(x_com3u)
x52u=Activation('LeakyReLU')(xu)

xu=Conv2D(32,(3,3),padding="same")(x_com4u)
x53u=Activation('LeakyReLU')(xu)

x_com5u = Add()([x51u, x52u])
x_com6u = Add()([x_com5u, x53u])

xu=Conv2D(3,(3,3),padding="same")(x_com6u)
xu=Activation('LeakyReLU')(xu)

x_com7u = Add()([x1u, xu])

xu=Conv2D(32,(3,3),padding="same")(x_com7u)
#xupper = tf.math.exp(xu)
xupper = 1 - tf.math.exp(-1 * xu)

x1u=input_img

x2u=Conv2D(32,(3,3),padding="same")(x1u)

xu=Conv2D(32,(3,3),padding="same")(x2u)
xu=Activation('LeakyReLU')(xu)

xu=Conv2D(32,(3,3),padding="same")(xu)
xu=Activation('LeakyReLU')(xu)

x_com1u = Add()([x2u, xu])

x3u=Conv2D(32,(3,3),padding="same")(x_com1u)

xu=Conv2D(32,(3,3),padding="same")(x3u)
x41u=Activation('LeakyReLU')(xu)

xu=Conv2D(32,(3,3),padding="same")(x3u)
x42u=Activation('LeakyReLU')(xu)

xu=Conv2D(32,(3,3),padding="same")(x3u)
x43u=Activation('LeakyReLU')(xu)

x_com2u = Add()([x41u, x42u])
x_com3u = Add()([x41u, x43u])
x_com4u = Add()([x42u, x43u])

xu=Conv2D(32,(3,3),padding="same")(x_com2u)
x51u=Activation('LeakyReLU')(xu)

xu=Conv2D(32,(3,3),padding="same")(x_com3u)
x52u=Activation('LeakyReLU')(xu)

xu=Conv2D(32,(3,3),padding="same")(x_com4u)
x53u=Activation('LeakyReLU')(xu)

x_com5u = Add()([x51u, x52u])
x_com6u = Add()([x_com5u, x53u])

xu=Conv2D(3,(3,3),padding="same")(x_com6u)
xu=Activation('LeakyReLU')(xu)

x_com7u = Add()([x1u, xu])

xu=Conv2D(32,(3,3),padding="same")(x_com7u)
xlower = xu

x_com8 = Add()([xupper, xlower])
xm1 =Conv2D(32,(3,3),padding="same")(x_com8)

xu=Conv2D(32,(3,3),padding="same")(xm1)
xu=Activation('LeakyReLU')(xu)

xu=Conv2D(32,(3,3),padding="same")(xu)
xu=Activation('LeakyReLU')(xu)

x_com9 = Add()([xm1, xu])

xfinal=Conv2D(3,(3,3),padding="same")(x_com9)

model = Model(inputs=input_img, outputs=xfinal)

# load the data and normalize it
highImages=np.load(config.data)
print(highImages.dtype)
highImages=highImages.astype('float32')

lowImages=np.load(config.data_noisy)
print(lowImages.dtype)
lowImages=lowImages.astype('float32')

# define augmentor and create custom flow
aug = ImageDataGenerator(rotation_range=30, fill_mode="nearest")
def myFlow(generator,X,Y):
    for batchhigh, batchlow in generator.flow(x=X,y=Y,batch_size=config.batch_size,seed=0):
        yield (batchlow,batchhigh)

# create custom learning rate scheduler
def lr_decay(epoch):
    initAlpha=0.001
    factor=0.5
    dropEvery=5
    alpha=initAlpha*(factor ** np.floor((1+epoch)/dropEvery))
    return float(alpha)
callbacks=[LearningRateScheduler(lr_decay)]

# create custom loss, compile the model
print("[INFO] compilingTheModel")
#opt=optimizers.Adam(lr=0.001)
#def custom_loss(y_true,y_pred):
#    diff=y_true-y_pred
#    res=K.sum(diff*diff)/(2*config.batch_size)
#    return res

opt=optimizers.Adam(learning_rate=0.0001)
def custom_loss(y_true,y_pred):
    diff=K.abs(y_true-y_pred)
    l1=(diff)/(config.batch_size)
    return l1
model.compile(loss=custom_loss,optimizer=opt)

# train
model.fit_generator(myFlow(aug,highImages,lowImages),
epochs=config.epochs,steps_per_epoch=len(highImages)//config.batch_size,callbacks=callbacks,verbose=1)

# save the model
model.save('EnlightenNet.h5')
