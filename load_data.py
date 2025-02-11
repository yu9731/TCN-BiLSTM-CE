import numpy as np
import pandas as pd

data = pd.read_csv('Data/electricity.csv', index_col=[0], parse_dates=True)

train_start_dt = '2016-06-26 00:00:00'
train_end_dt = '2016-12-30 23:00:00'

test_start_dt = '2017-07-03 00:00:00'
test_end_dt = '2017-07-26 23:00:00'

building_name = 'Mouse_health_Estela'

def pd_time_information(Timestamp):
    '''
    Exracting all the temporal information that we need to consider
    : param Timestamp: All timestamps contained in the training and test data. dtype: list
    : return pd_time_information: A dataframe contained all needed temporal information. object: DataFrame
    '''
    pd_time_information = pd.DataFrame(columns=['DayOfWeek','hour'])
    for i in range(len(Timestamp)):
        day_of_week = pd.Timestamp(Timestamp[i]).dayofweek + 1
        hour = int(str(pd.Timestamp(Timestamp[i]))[11:13])
        pd_time_information.loc[i] = [day_of_week, hour]
    return pd_time_information

def filtering_by_hours(train_data, pd_time, Timestamp, save_file=False):
    '''
    Filtering outliers in measured data by the correlation between heat demand and specific hours
    :param df: Dataframe after data pre-processing
    '''
    pd_time_information_index = pd.to_datetime(Timestamp)
    pd_time_information.index = pd_time_information_index
    pd_time['Energy [KW]'] = train_data

    for i in range(24):
        hour = i
        energy_demand = pd_time[pd_time['hour'] == hour]['Energy [KW]'].values
        energy_demand_25 = np.percentile(energy_demand, 25)
        energy_demand_75 = np.percentile(energy_demand, 75)
        energy_demand_median = np.percentile(energy_demand, 50)
        lower_bound = energy_demand_25 - 1.5 * (energy_demand_75 - energy_demand_25)
        upper_bound = energy_demand_75 + 1.5 * (energy_demand_75 - energy_demand_25)
        for j in range(len(energy_demand)):
            if (lower_bound <= energy_demand[j] <= upper_bound) or np.isnan(energy_demand[j]):
                pass
            else:
                outlier_index = pd_time[pd_time['hour'] == hour]['Energy [KW]'].index[j]
                pd_time.loc[outlier_index, 'Energy [KW]'] = np.percentile(energy_demand, 50)

    return pd_time

train_data_time_features = pd_time_information(data.loc[train_start_dt:train_end_dt, f'{building_name}'].index)
test_data_time_features = pd_time_information(data.loc[test_start_dt:test_end_dt, f'{building_name}'].index)

train_data = data.loc[train_start_dt:train_end_dt, f'{building_name}'].astype('float32')
train_data = train_data.resample('60min').interpolate('linear').to_frame()
train_data = train_data.values

pd_train_data = filtering_by_hours(train_data, train_data_time_features, data.loc[train_start_dt:train_end_dt, f'{building_name}'].index)
train_data = pd_train_data['Energy [KW]'].values
train_data = train_data.reshape((len(train_data),1))

test_data = data.loc[test_start_dt:test_end_dt, f'{building_name}'].astype('float32')
test_data = test_data.resample('60min').interpolate('linear').to_frame()
# Interpolating all the missing values in training set
test_data = test_data.values
test_data = test_data.reshape((len(test_data),1))

def create_dataset(train_data_time_features, train_data):
    '''
    Combining all three variables (hour, electrictiy load, day of week) for further modeling of training samples
    :param: train_data_time_features: DataFrame comes from the function 'pd_time_information'. Object: DataFrame
    :param: train_data: Electricity load data. Object: Numpy Array
    :return: np_train: Numpy array combines all three variables across the whole training/testing time period. Object: Numpy array
    '''
    np_day_of_week = np.zeros((train_data_time_features.shape[0], 1))
    np_hour = np.zeros((train_data_time_features.shape[0], 1))

    day_of_week = train_data_time_features.loc[:,'DayOfWeek'].values
    hour = train_data_time_features.loc[:,'hour'].values

    for i in range(np_day_of_week.shape[0]):
        np_day_of_week[i] = np.sin(2 * np.pi * day_of_week[i]/7.0)
        np_hour[i] = np.sin(2 * np.pi * hour[i]/24.0)

    np_day_of_week = np_day_of_week.reshape((np_day_of_week.shape[0], 1))
    np_hour = np_hour.reshape((np_hour.shape[0], 1))

    np_train = np.hstack((np_day_of_week, train_data, np_hour))
    return np_train

np_train = create_dataset(train_data_time_features, train_data)
np_test = create_dataset(test_data_time_features, test_data)

def create_sequence(np_train, np_test, building):
    '''
    Modeling the training samples with rolling windows and normalizing the training and validation data. Default windows length: 336
    :param np_train: Training data represents as a numpy array. Object: Numpy array
    :param np_test: Testing data represents as a numpy array. Object: Numpy array
    :param building: Building name in DataFrame containing all electricity load data. Dtype: str
    :return data_train: Training set.
    :return data_validation: Validation set.
    '''

    np.save(f'Data/Train_data/min_7m_{building}_train.npy', np.min(np_train[:, 1]))
    np.save(f'Data/Train_data/max_7m_{building}_train.npy', np.max(np_train[:, 1]))

    for i in range(np_train.shape[1]):
        np_test[:, i] = (np_test[:, i] - np.min(np_train[:, i])) / (np.max(np_train[:, i]) - np.min(np_train[:, i]))
        np_train[:, i] = (np_train[:, i] - np.min(np_train[:, i])) / (np.max(np_train[:, i]) - np.min(np_train[:, i]))

    data_train = np.zeros((((np_train.shape[0]-336)//24)+1, 336, 3))
    for i in range(data_train.shape[0]):
        idx = i
        for j in range(3):
            data_train[idx, :, j] = np_train[i*24:(i*24+336), j]

    data_test = np.zeros((((np_test.shape[0] - 336) // 24) + 1, 336, 3))
    for i in range(data_test.shape[0]):
        idx = i
        for j in range(3):
            data_test[idx, :, j] = np_test[i * 24:(i * 24 + 336), j]

    np.save(f'Data/Train_data/data_train_7m_{building}.npy', data_train)
    np.save(f'Data/Train_data/data_validation_7m_{building}.npy', data_test)

    return data_train, data_test

data_train, data_test = create_sequence(np_train, np_test, building_name)
