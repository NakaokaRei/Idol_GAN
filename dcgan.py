from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout
import numpy as np
import os
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from PIL import Image
import glob


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1024*4*4))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape((4, 4, 1024), input_shape=(1024*4*4,)))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(512, (6, 6), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(256, (7, 7), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(128, (7, 7), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (7, 7), border_mode='same'))
    model.add(Activation('tanh'))
    return model

def discriminator_model():
    model = Sequential()
    model.add(Conv2D(256, (5, 5),
                     strides=(2, 2),
                     padding='same',
                     input_shape=(64, 64, 3))) # ここ注意
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, (5, 5), strides=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def combine_images(generated_images, GENERATED_IMAGE_PATH, epoch, index):
    plt.figure(figsize=(7, 7))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        img = generated_images[i]
        plt.tight_layout()
        img = (img + 1.0) * 127.5
        plt.imshow(img.astype(np.uint8))
        plt.axis("off")

    if not os.path.exists(GENERATED_IMAGE_PATH):
        os.mkdir(GENERATED_IMAGE_PATH)
    plt.savefig(GENERATED_IMAGE_PATH+"%04d_%04d.png" % (epoch, index))

BATCH_SIZE = 32
NUM_EPOCH = 30000
GENERATED_IMAGE_PATH = 'generaed_image/' # 生成画像の保存先


def train():

    files = glob.glob('訓練画像ディレクトリ/*.jpg') #訓練画像ディレクトリ
    images = []
    for f in files:
        img = Image.open(f)
        img = np.asarray(img) / 127.5 - 1.0
        images.append(img)
    images = np.asarray(images)
    noise_t = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])

    discriminator = discriminator_model()
    d_opt = Adam(lr=1e-6, beta_1=0.1)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)
    discriminator.summary()

    # generator+discriminator （discriminator部分の重みは固定）
    discriminator.trainable = False
    generator = generator_model()
    dcgan = Sequential([generator, discriminator])
    g_opt = Adam(lr=2e-5, beta_1=0.5)
    dcgan.compile(loss='binary_crossentropy', optimizer=g_opt)
    dcgan.summary()

    num_batches = int(images.shape[0] / BATCH_SIZE)
    print('Number of batches:', num_batches)
    for epoch in range(NUM_EPOCH):

        for index in range(num_batches):

            noise = np.random.uniform(-1, 1, size=[BATCH_SIZE, 100])
            image_batch = images[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)
            # 生成画像を出力
            if index % 500 == 0:
                generated_images_t = generator.predict(noise_t, verbose=0)
                combine_images(generated_images_t, GENERATED_IMAGE_PATH, epoch, index)


            # discriminatorを更新
            #image_batch = image_batch.transpose((0, 2, 3, 1))
            #print(image_batch.shape)
            X = np.concatenate((image_batch, generated_images))
            y = [1]*BATCH_SIZE + [0]*BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)

            # generatorを更新
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            g_loss = dcgan.train_on_batch(noise, [1]*BATCH_SIZE)
            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))

        generator.save_weights('generator.h5')
        discriminator.save_weights('discriminator.h5')

train()
