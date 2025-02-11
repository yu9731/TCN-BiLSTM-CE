from keras.models import Model, save_model, load_model
import numpy as np
from matplotlib import pyplot as plt
from tcn import TCN
import tensorflow as tf

def test(building):
    generator = load_model(f'Model/{building}/generator_{building}_after_discriminator.h5', custom_objects={"TCN": TCN})
    generator.summary()
    x_test = np.load(f'Data/Train_Data/data_validation_7m_{building}.npy')

    step_num = int(len(x_test) / batch_size)

    min_train = np.load(f'Data/Train_Data/min_7m_{building}_train.npy')
    max_train = np.load(f'Data/Train_Data/max_7m_{building}_train.npy')

    cnt = 0
    np_nrmse = np.zeros((step_num,1))
    _, mask_batch = get_points()
    for i in range(step_num):
        x_batch = x_test[i * batch_size:(i + 1) * batch_size]

        generator_input = x_batch * (1 - mask_batch)
        generator_output = generator.predict(generator_input)
        completion = generator_output * mask_batch + generator_input * (1 - mask_batch)

        for j in range(batch_size):
            cnt += 1
            raw = x_batch[j]
            masked = raw * (1 - mask_batch[j]) + np.ones_like(raw) * mask_batch[j] * 0
            imputation = completion[j]

            imputation = imputation * (max_train - min_train) + min_train
            raw = raw * (max_train - min_train) + min_train

            plt.figure(figsize=(14/2.54, 8/2.54))
            plt.plot(imputation[:,1], c='red')
            plt.plot(raw[:,1], c='blue')
            plt.xlabel('Timestamp')
            plt.ylabel('Q_heat/kW')
            plt.grid(visible='True', linewidth=3.0)
            plt.legend(prop={'size': 20}, ncol=2, bbox_to_anchor=(0.68, 1.15))
            plt.tight_layout()
            plt.show()

            nrmse = np.sqrt(np.mean(np.square(imputation[48:288,1] - raw[48:288,1]))) / (max_train-min_train)
            np_nrmse[i] = nrmse
    print(np_nrmse, np.mean(np_nrmse))

global_size = 336
local_size = 240
batch_size = 1

def get_points():
    points = []
    mask = []
    for i in range(batch_size):
        x1 = global_size - local_size - 48
        # x1: start point of the continuos missing, in this case: 48 -> 336 - 240 (local_size) - 48 = 48
        x2 = x1 + local_size
        points.append([x1, x2])

        p1 = x1
        p2 = p1 + local_size

        m = np.zeros((global_size, 3), dtype=np.uint8)
        m[p1:p2+1, 1] = 1
        mask.append(m)
    return np.array(points), np.array(mask)

'''
# Testing for discrete missing values
def get_points():
    mask = []
    points = np.random.choice(336,270, False)
    for i in range(batch_size):
        m = np.zeros((global_size, 3), dtype=np.uint8)
        for j in range(len(points)):
            m[points[j], 1] = 1
        mask.append(m)
    return np.array(points), np.array(mask)
'''

building_name = 'Panther_retail_Kristina'
test(building_name)
