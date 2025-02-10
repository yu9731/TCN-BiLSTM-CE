from keras.models import Model, save_model, load_model
import numpy as np
from matplotlib import pyplot as plt
from tcn import TCN
import tensorflow as tf

def test(building, day):
    generator = load_model(f'TL_Model/Model_elec/{building}/generator_{building}_after_discriminator_100.h5', custom_objects={"TCN": TCN})
    generator.summary()
    x_test = np.load(f'Train_Data/Train_TL/data_validation_7m_{building}.npy')

    step_num = int(len(x_test) / BATCH_SIZE)

    min_train = np.load(f'Train_Data/Train_TL/min_7m_{building}_train.npy')
    max_train = np.load(f'Train_Data/Train_TL/max_7m_{building}_train.npy')

    cnt = 0
    np_nrmse = np.zeros((step_num,1))
    _, mask_batch = get_points()
    for i in range(step_num):
        x_batch = x_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        # _, mask_batch = get_points()

        generator_input = x_batch * (1 - mask_batch)
        generator_output = generator.predict(generator_input)
        completion = generator_output * mask_batch + generator_input * (1 - mask_batch)

        for j in range(BATCH_SIZE):
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
    print(np_nrmse)

IMAGE_SIZE = 336
LOCAL_SIZE = 264
# LOCAL_SIZE = 24 * (missing_day + 1), in this example, missing_day=10
HOLE = 240
# HOLE = 24 * missing_day, in this example, missing_day=10
BATCH_SIZE = 1


def get_points():
    points = []
    mask = []
    for i in range(BATCH_SIZE):
        x1 = IMAGE_SIZE - LOCAL_SIZE - 48
        x2 = x1 + LOCAL_SIZE
        points.append([x1, x2])

        w = HOLE
        p1 = x1 + (LOCAL_SIZE - w)
        p2 = p1 + w

        m = np.zeros((IMAGE_SIZE, 3), dtype=np.uint8)
        m[p1:p2+1, 1] = 1
        mask.append(m)
    return np.array(points), np.array(mask)

'''
Testing for discrete missing values
def get_points():
    mask = []
    points = np.random.choice(336,270, False)
    for i in range(BATCH_SIZE):
        m = np.zeros((IMAGE_SIZE, 3), dtype=np.uint8)
        for j in range(len(points)):
            m[points[j], 1] = 1
        mask.append(m)
    return np.array(points), np.array(mask)
'''

building_name = ['Estela']
for i, building in enumerate(building_name):
    test(building, '10d')
