import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Flatten, Reshape
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Bidirectional, LSTM
from keras import models
from keras.models import Model, save_model, load_model
from keras import backend as K
from keras import metrics

from keras import layers
from tensorflow.keras.layers import Conv1D
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from tcn import TCN

import random
import keras
import tracemalloc
import time

global_size = 336   # Length of a sample
local_size = 240      # Length of missing values
batch_size = 8

building = 'Mouse_health_Estela'
# Using the building 'Mouse_health_Estela' as an example, you can also using other building for reproducing the result

def build_generator(input_dim, variable_num=3):
    encoder_input = Input(shape=(input_dim, variable_num))
    x = tf.keras.layers.Conv1D(32, 3, 1, activation='relu', padding='same', name="Encoder_1")(encoder_input)
    x = tf.keras.layers.Conv1D(64, 3, 2, activation='relu', padding='same', name="Encoder_2")(x)
    x = tf.keras.layers.Conv1D(128, 3, 2, activation='relu', padding='same', name="Encoder_3")(x)
    x = tf.keras.layers.Conv1D(128, 3, 2, activation='relu', padding='same', name="Encoder_4")(x)
    x = tf.keras.layers.Conv1D(128, 3, 1, activation='relu', padding='same', name="Encoder_5")(x)
    x = tf.keras.layers.Conv1D(128, 3, 1, activation='relu', padding='same', name="Encoder_6")(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = TCN(128, 3, 1, dilations=[1, 2, 4, 8], return_sequences=True, padding='causal', name="TCN_1")(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    encoded = TCN(128, 3, 1, dilations=[1, 2, 4, 8], return_sequences=True, padding='causal', name="TCN_2")(x)

    decoder_layer_1 = tf.keras.layers.Conv1D(128, 3, 1, activation='relu', padding='same', name="Decoder_1")(encoded)
    x = tf.keras.layers.Conv1D(128, 3, 1, activation='relu', padding='same', name="Decoder_2")(decoder_layer_1)
    x = tf.keras.layers.Conv1DTranspose(128, 3, 2, activation='relu', padding='same', name="Decoder_3")(x)
    x = tf.keras.layers.Conv1DTranspose(64, 3, 2, activation='relu', padding='same', name="Decoder_4")(x)
    x = tf.keras.layers.Conv1DTranspose(32, 3, 2, activation='relu', padding='same', name="Decoder_5")(x)
    decoder_output = tf.keras.layers.Conv1D(variable_num, 3, 1, activation='relu', padding='same', name="Decoder_Output")(x)

    generator = models.Model(encoder_input, decoder_output)
    return generator

generator = build_generator(336,3)
generator.summary()

def discriminator(input_dim_global=336, input_dim_local=240, variable_num=3):
    '''
    Discriminator get the output of the real and fake seperately and then using them to calculate the joint loss
    '''
    global_input = layers.Input(shape=(input_dim_global, variable_num))
    local_input = layers.Input(shape=(input_dim_local, variable_num))

    def global_discriminator(global_input):
        x = tf.keras.layers.Conv1D(16, 3, 2, activation='relu', padding='same', name="Global_Discriminator_1")(global_input)
        x = tf.keras.layers.Conv1D(32, 3, 2, activation='relu', padding='same', name="Global_Discriminator_2")(x)
        x = tf.keras.layers.Conv1D(64, 3, 2, activation='relu', padding='same', name="Global_Discriminator_3")(x)
        x = tf.keras.layers.Conv1D(128, 3, 2, activation='relu', padding='same', name="Global_Discriminator_4")(x)

        flatten_layer = tf.keras.layers.Flatten()
        result = flatten_layer(x)
        result = tf.keras.layers.Dense(1024)(result)

        return result

    def local_discriminator(local_input):
        x = tf.keras.layers.Conv1D(16, 3, 2, activation='relu', padding='same', name="Local_Discriminator_1")(local_input)  
        x = tf.keras.layers.Conv1D(32, 3, 2, activation='relu', padding='same', name="Local_Discriminator_2")(x)
        x = tf.keras.layers.Conv1D(64, 3, 2, activation='relu', padding='same', name="Local_Discriminator_3")(x)
        x = tf.keras.layers.Conv1D(128, 3, 2, activation='relu', padding='same', name="Local_Discriminator_4")(x)

        flatten_layer = tf.keras.layers.Flatten()
        result = flatten_layer(x)
        result = tf.keras.layers.Dense(1024)(result)

        return result

    global_d = global_discriminator(global_input)
    local_d = local_discriminator(local_input)

    output = tf.concat([global_d, local_d], axis=1)
    output = tf.keras.layers.Dense(1)(output)

    model = tf.keras.Model(inputs=[global_input, local_input], outputs=output)
    return model

def get_points():
    '''
    Randomly creating missing values in training samples for training the model. 
    Manually create data samples with up to 10 days of missing data, starting at a random point in time series data.
    '''
    points = []
    mask = []
    for i in range(batch_size):
        x1 = np.random.randint(0,95,1,'int')[0]
        x2 = x1 + local_size

        points.append([x1, x2])

        p1 = x1
        p2 = p1 + local_size

        m = np.zeros((global_size, 3), dtype=np.uint8)
        m[p1:p2+1, -2] = 1
        mask.append(m)
    return np.array(points), np.array(mask)

def calc_g_loss(x, completion):
    loss = tf.nn.l2_loss(x - completion)
    return tf.reduce_mean(loss)

discriminator = discriminator(variable_num=3)
generator.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss='mean_squared_error')

def joint_loss(real, fake):
    '''
    :param real: Output from discriminator for real samples
    :param fake: Output from discriminator for fake samples
    :return:
    '''
    alpha = 4e-4
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
    return tf.add(d_loss_real, d_loss_fake) * alpha

pre_trained_epoch = 100

adversarial_epoch = 50

memory_usage = []
time_usage = []

x_train = np.load(f'Data/Train_Data/data_train_7m_{building}.npy')

tracemalloc.start()
t0 = time.time()
for epoch in range(pre_trained_epoch+1):
    np.random.shuffle(x_train)
    step_num = int(len(x_train) * 0.8 / batch_size)
    for i in tqdm.tqdm(range(step_num)):
        x_batch = x_train[0:int(len(x_train) * 0.8)][i * batch_size:(i + 1) * batch_size]
        points_batch, mask_batch = get_points()

        generator_input = x_batch * (1 - mask_batch)
        loss = generator.train_on_batch(generator_input, x_batch)

    x_batch_test = x_train[int(len(x_train) * 0.8):][:batch_size]
    points_batch_test, mask_batch_test = get_points()
    generator_test_input = x_batch_test * (1 - mask_batch_test)
    test_loss = generator.evaluate(generator_test_input, x_batch_test)

    if epoch % 10 == 0:
        print(f"Autoencoder Pretrain Epoch [{epoch}/{pre_trained_epoch}] | Loss: {loss}| Test Loss: {test_loss}")

generator.save(f'Model/{building}/generator_{building}.h5')
generator = load_model(f'Model/{building}/generator_{building}.h5', custom_objects={"TCN": TCN})

optimizer_D = tf.keras.optimizers.Adam(0.0005)

for epoch in range(adversarial_epoch+1):
    np.random.shuffle(x_train)
    step_num = int(len(x_train) * 0.8 / batch_size)
    for i in range(step_num):
        x_batch = x_train[i * batch_size:(i + 1) * batch_size]
        points_batch, mask_batch = get_points()
        generator_input = x_batch * (1 - mask_batch)

        generator_output = generator.predict(generator_input)
        completion = generator_output * mask_batch + generator_input * (1 - mask_batch)

        local_x_batch = []
        local_completion_batch = []
        for i in range(batch_size):
            x1, x2 = points_batch[:, 0][0], points_batch[:, 1][0]
            local_x_batch.append(x_batch[i][x1:x2, :])
            local_completion_batch.append(completion[i][x1:x2, :])

        local_x_batch = np.array(local_x_batch)
        local_completion_batch = np.array(local_completion_batch)

        # Training the discriminator
        with tf.GradientTape() as tape:
            real_output = discriminator([x_batch, local_x_batch], training=True)
            fake_output = discriminator([completion, local_completion_batch], training=True)
            d_loss = joint_loss(real_output, fake_output)

        gradients = tape.gradient(d_loss, discriminator.trainable_variables)
        optimizer_D.apply_gradients(zip(gradients, discriminator.trainable_variables))

        # Test Discriminator
        x_batch_test = x_train[int(len(x_train) * 0.8):][:batch_size]
        points_batch_test, mask_batch_test = get_points()
        generator_test_input = x_batch_test * (1 - mask_batch_test)

        test_generator_output = generator.predict(generator_test_input)
        test_completion = test_generator_output * mask_batch_test + generator_test_input * (1 - mask_batch_test)

        local_x_batch_test = []
        local_completion_batch_test = []
        for i in range(batch_size):
            x1, x2 = points_batch[:, 0][0], points_batch[:, 1][0]
            local_x_batch_test.append(x_batch_test[i][x1:x2, :])
            local_completion_batch_test.append(test_completion[i][x1:x2, :])

        local_x_batch_test = np.array(local_x_batch_test)
        local_completion_batch_test = np.array(local_completion_batch_test)

        test_real_output = discriminator([x_batch_test, local_x_batch_test], training=False)
        test_fake_output = discriminator([test_completion, local_completion_batch_test], training=False)

        d_test_loss = joint_loss(test_real_output, test_fake_output)

        # Train Generator
        generator_input = x_batch * (1 - mask_batch)
        g_loss = generator.train_on_batch(generator_input, x_batch)

        # Test Generator
        x_batch_test = x_train[int(len(x_train) * 0.8):][:batch_size]
        points_batch_test, mask_batch_test = get_points()
        generator_test_input = x_batch_test * (1 - mask_batch_test)
        g_test_loss = generator.evaluate(generator_test_input, x_batch_test)

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{adversarial_epoch}] | D Loss: {d_loss} | G Loss: {g_loss}| D Test Loss: {d_test_loss} | G Test Loss: {g_test_loss}")

generator.save(f'Model/{building}/generator_{building}_after_discriminator.h5')

'''
# Figuring out the memory usage
print("Memory usage:", tracemalloc.get_traced_memory())
memory_usage.append(tracemalloc.get_traced_memory())
print("Training time:", time.time() - t0)
time_usage.append(time.time() - t0)
tracemalloc.stop()

np.save('memory_usage_elec.npy', np.array(memory_usage))
np.save('time_usage_elec.npy', np.array(time_usage))
'''
