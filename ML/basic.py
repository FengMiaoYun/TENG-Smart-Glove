import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "Images")
DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "Datasets")


# 保存生成的图片
# fig_id: string
# return: None
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# 读取csv文件
# file_name: string
# return: DataFrame
def load_data(file_name, path=DATA_PATH, file_extension="csv", skip_rows=True):
    file_path = os.path.join(path, file_name + "." + file_extension)
    if skip_rows:
        return pd.read_csv(file_path, skiprows=[1]).T
    else:
        return pd.read_csv(file_path).T


# 降低采样频率
# dataframe: DataFrame, frequency: int
# return: DataFrame
def lower_frequency(dataframe, frequency=50):
    data = dataframe.values
    second = data[0]
    voltage1 = data[1]
    voltage2 = data[2]
    voltage3 = data[3]
    voltage4 = data[4]

    T = 1.0 / frequency
    WHOLE_TIME = 5
    new_second = [i for i in range(int(WHOLE_TIME / T))]
    new_voltage1 = np.zeros(len(new_second))
    new_voltage2 = np.zeros(len(new_second))
    new_voltage3 = np.zeros(len(new_second))
    new_voltage4 = np.zeros(len(new_second))

    count = 0
    for i in range(len(second)):
        if i % (400 / frequency) == 0:
            new_second[count] /= frequency
            new_voltage1[count] = voltage1[i]
            new_voltage2[count] = voltage2[i]
            new_voltage3[count] = voltage3[i]
            new_voltage4[count] = voltage4[i]
            count += 1

    new_second = pd.DataFrame(new_second, columns=['second'], dtype=float)
    new_voltage1 = pd.DataFrame(new_voltage1, columns=['voltage1'], dtype=float)
    new_voltage2 = pd.DataFrame(new_voltage2, columns=['voltage2'], dtype=float)
    new_voltage3 = pd.DataFrame(new_voltage3, columns=['voltage3'], dtype=float)
    new_voltage4 = pd.DataFrame(new_voltage4, columns=['voltage4'], dtype=float)
    dataframe = pd.concat([new_second, new_voltage1, new_voltage2, new_voltage3, new_voltage4], axis=1)

    return dataframe.T


# 电压值归一化
# dataframe: DataFrame, voltage_limit: float
# return: DataFrame
def voltage_normalization(dataframe, voltage_limit=3.3):
    data = dataframe.values
    second = data[0]
    voltage1 = data[1]
    voltage2 = data[2]
    voltage3 = data[3]
    voltage4 = data[4]

    # max_voltage = np.zeros(5)
    # for i in range(len(second)):
    #     if abs(voltage1[i]) > max_voltage[0]:
    #         max_voltage[0] = abs(voltage1[i])
    #     if abs(voltage2[i]) > max_voltage[1]:
    #         max_voltage[1] = abs(voltage2[i])
    #     if abs(voltage3[i]) > max_voltage[2]:
    #         max_voltage[2] = abs(voltage3[i])
    #     if abs(voltage4[i]) > max_voltage[3]:
    #         max_voltage[3] = abs(voltage4[i])
    # for i in range(len(max_voltage) - 1):
    #     if max_voltage[i] > max_voltage[4]:
    #         max_voltage[4] = max_voltage[i]
    #
    # voltage_limit *= 0.95
    # for i in range(len(second)):
    #     voltage1[i] = voltage1[i] * max_voltage[4] / max_voltage[0]
    #     voltage2[i] = voltage2[i] * max_voltage[4] / max_voltage[1]
    #     voltage3[i] = voltage3[i] * max_voltage[4] / max_voltage[2]
    #     voltage4[i] = voltage4[i] * max_voltage[4] / max_voltage[3]
    voltage_limit *= 0.95
    voltage1 = [i * 1.1 for i in voltage1]
    voltage3 = [i * 5 for i in voltage3]
    voltage4 = [i * 25 for i in voltage4]

    second = pd.DataFrame(second, columns=['second'], dtype=float)
    voltage1 = pd.DataFrame(voltage1, columns=['voltage1'], dtype=float)
    voltage2 = pd.DataFrame(voltage2, columns=['voltage2'], dtype=float)
    voltage3 = pd.DataFrame(voltage3, columns=['voltage3'], dtype=float)
    voltage4 = pd.DataFrame(voltage4, columns=['voltage4'], dtype=float)
    dataframe = pd.concat([second, voltage1, voltage2, voltage3, voltage4], axis=1)

    return dataframe.T


# 信号裁剪
# dataframe: DataFrame
# return: DataFrame, int
def signal_cut_down(dataframe):
    data = dataframe.values
    second = data[0]
    volt = [data[1], data[2], data[3], data[4]]

    # 得到一列电压信号中有用成分的开始和结束时间
    # voltage: ndarray 1D
    # return: int, int
    def voltage_start_end(voltage):
        max_voltage = max(voltage)
        min_voltage = min(voltage)
        max_useful = 1
        min_useful = 1
        # max_count = 0
        # min_count = 1
        # for k in range(len(voltage)):
        #     if voltage[k] > max_voltage * 0.15:
        #         max_count += 1
        #     if voltage[k] < min_voltage * 0.15:
        #         min_count += 1
        # if max_count > 15:
        #     max_useful = 0
        # if min_count > 15:
        #     min_useful = 0
        #
        # voltage_positive = []
        # voltage_negative = []
        # for k in range(len(voltage)):
        #     if voltage[k] > 0:
        #         voltage_positive.append(voltage[k])
        #     if voltage[k] < 0:
        #         voltage_negative.append(voltage[k])
        # if max_voltage < np.mean(voltage_positive) * 7:
        #     max_useful = 0
        # if min_voltage > np.mean(voltage_negative) * 7:
        #     max_useful = 0
        if max_voltage < 0.4:
            max_useful = 0
        if min_voltage > -0.4:
            min_useful = 0

        if max_useful | min_useful:
            start = 0
            end = len(voltage) - 1
            for k in range(len(voltage)):
                if ((voltage[k] > max_voltage * 0.5) & max_useful) | ((voltage[k] < min_voltage * 0.5) & min_useful):
                    start = k
                    break
            for k in range(len(voltage)):
                if ((voltage[len(voltage) - 1 - k] > max_voltage * 0.5) & max_useful) | (
                        (voltage[len(voltage) - 1 - k] < min_voltage * 0.5) & min_useful):
                    end = len(voltage) - 1 - k
                    break
            start -= 15
            if start < 0:
                start = 0
            end += 15
            if end > len(voltage) - 1:
                end = len(voltage) - 1
        else:
            start = len(voltage) - 1
            end = 0
        return start, end

    cut_start = np.zeros(5)
    cut_end = np.zeros(5)
    for i in range(4):
        cut_start[i], cut_end[i] = voltage_start_end(volt[i])
    cut_start[4] = len(second) - 1
    cut_start[4] = min(cut_start)
    cut_end[4] = max(cut_end)
    length = int(cut_end[4]) - int(cut_start[4]) + 1

    new_second = [i * second[1] for i in range(length)]
    new_volt = [np.zeros(length), np.zeros(length), np.zeros(length), np.zeros(length)]
    for i in range(length):
        for j in range(4):
            new_volt[j][i] = volt[j][i + int(cut_start[4])]

    new_second = pd.DataFrame(new_second, columns=['second'], dtype=float)
    new_volt1 = pd.DataFrame(new_volt[0], columns=['voltage1'], dtype=float)
    new_volt2 = pd.DataFrame(new_volt[1], columns=['voltage2'], dtype=float)
    new_volt3 = pd.DataFrame(new_volt[2], columns=['voltage3'], dtype=float)
    new_volt4 = pd.DataFrame(new_volt[3], columns=['voltage4'], dtype=float)
    dataframe = pd.concat([new_second, new_volt1, new_volt2, new_volt3, new_volt4], axis=1)

    return dataframe.T, length


# 增加信号随机噪声
# dataframe: DataFrame
# return: ndarray DataFrame
def add_noise(dataframe, snr=15, multiple=5):
    data = dataframe.values
    second = data[0]
    second = pd.DataFrame(second, columns=['second'], dtype=float)
    volt = [data[1], data[2], data[3], data[4]]
    noise_volt = volt
    multiple_dataframe = []

    # 产生高斯白噪声
    # signal: ndarray 1D, snr: int (signal-noise-ratio)
    # return: ndarray 1D
    def wgn(signal, _snr):
        noise = np.random.randn(len(signal))
        _snr = 10 ** (_snr / 10)
        signal_power = np.mean(np.square(signal))
        noise_power = signal_power / _snr
        noise = noise * np.sqrt(noise_power)
        return noise

    for i in range(multiple):
        for j in range(len(volt)):
            noise_volt[j] += wgn(volt[j], snr)
        voltage1 = pd.DataFrame(noise_volt[0], columns=['voltage1'], dtype=float)
        voltage2 = pd.DataFrame(noise_volt[1], columns=['voltage2'], dtype=float)
        voltage3 = pd.DataFrame(noise_volt[2], columns=['voltage3'], dtype=float)
        voltage4 = pd.DataFrame(noise_volt[3], columns=['voltage4'], dtype=float)
        multiple_dataframe.append(pd.concat([second, voltage1, voltage2, voltage3, voltage4], axis=1).T)

    return multiple_dataframe


# 电压积分
# dataframe: DataFrame
# return: DataFrame
def voltage_integration(dataframe):
    data = dataframe.values
    second = data[0]
    voltage1 = data[1]
    voltage2 = data[2]
    voltage3 = data[3]
    voltage4 = data[4]

    voltage_sum = np.zeros(4)
    vol_integration1 = np.zeros(len(second))
    vol_integration2 = np.zeros(len(second))
    vol_integration3 = np.zeros(len(second))
    vol_integration4 = np.zeros(len(second))

    delta_time = second[1]
    for i in range(1, len(second)):
        voltage_sum[0] += voltage1[i] * delta_time
        vol_integration1[i] = voltage_sum[0]
        voltage_sum[1] += voltage2[i] * delta_time
        vol_integration2[i] = voltage_sum[1]
        voltage_sum[2] += voltage3[i] * delta_time
        vol_integration3[i] = voltage_sum[2]
        voltage_sum[3] += voltage4[i] * delta_time
        vol_integration4[i] = voltage_sum[3]

    vol_integration1 = pd.DataFrame(vol_integration1, columns=['integration1'], dtype=float)
    vol_integration2 = pd.DataFrame(vol_integration2, columns=['integration2'], dtype=float)
    vol_integration3 = pd.DataFrame(vol_integration3, columns=['integration3'], dtype=float)
    vol_integration4 = pd.DataFrame(vol_integration4, columns=['integration4'], dtype=float)
    dataframe = pd.concat([dataframe.T, vol_integration1, vol_integration2, vol_integration3, vol_integration4], axis=1)

    return dataframe.T


# 积分偏置
# dataframe: DataFrame
# # return: ndarray DataFrame
def integration_linear_bias(dataframe, multiple=5):
    data = dataframe.values
    second = data[0]
    volt = [data[1], data[2], data[3], data[4]]
    integration = [data[5], data[6], data[7], data[8]]
    dataframe_second = pd.DataFrame(second, columns=['second'], dtype=float)
    voltage1 = pd.DataFrame(volt[0], columns=['voltage1'], dtype=float)
    voltage2 = pd.DataFrame(volt[1], columns=['voltage2'], dtype=float)
    voltage3 = pd.DataFrame(volt[2], columns=['voltage3'], dtype=float)
    voltage4 = pd.DataFrame(volt[3], columns=['voltage4'], dtype=float)
    dataframe = pd.concat([dataframe_second, voltage1, voltage2, voltage3, voltage4], axis=1)

    bias_integration = integration
    multiple_dataframe = []
    max_bias = [data[5][-1], data[6][-1], data[7][-1], data[8][-1]]
    basic_bias = np.zeros(4)
    for t in range(multiple):
        for i in range(len(basic_bias)):
            basic_bias[i] = (max_bias[i] / second[-1]) * np.random.uniform(-2, 0)
            for j in range(len(second)):
                bias_integration[i][j] = integration[i][j] + basic_bias[i] * second[j]
        integration1 = pd.DataFrame(bias_integration[0], columns=['integration1'], dtype=float)
        integration2 = pd.DataFrame(bias_integration[1], columns=['integration2'], dtype=float)
        integration3 = pd.DataFrame(bias_integration[2], columns=['integration3'], dtype=float)
        integration4 = pd.DataFrame(bias_integration[3], columns=['integration4'], dtype=float)
        multiple_dataframe.append(pd.concat(
            [dataframe, integration1, integration2, integration3, integration4], axis=1).T)

    return multiple_dataframe


# 画出信号图并保存
# file_name: string
# return: None
def draw_signal(file_name):
    dataframe = voltage_integration(voltage_normalization(lower_frequency(load_data(file_name), frequency=200)))
    data = dataframe.values
    second = data[0]
    voltage1 = data[1]
    voltage2 = data[2]
    voltage3 = data[3]
    voltage4 = data[4]
    integration1 = data[5]
    integration2 = data[6]
    integration3 = data[7]
    integration4 = data[8]

    plt.figure(figsize=(40, 40))

    plt.subplot(8, 1, 1)
    plt.plot(second, voltage1, color='red')
    plt.title(file_name, fontsize=50)
    plt.xlabel("second", fontsize=30)
    plt.ylabel("voltage1", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplot(8, 1, 2)
    plt.plot(second, integration1, color='red')
    plt.xlabel("second", fontsize=30)
    plt.ylabel("integration1", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplot(8, 1, 3)
    plt.plot(second, voltage2, color='blue')
    plt.xlabel("second", fontsize=30)
    plt.ylabel("voltage2", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplot(8, 1, 4)
    plt.plot(second, integration2, color='blue')
    plt.xlabel("second", fontsize=30)
    plt.ylabel("integration2", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplot(8, 1, 5)
    plt.plot(second, voltage3, color='green')
    plt.xlabel("second", fontsize=30)
    plt.ylabel("voltage3", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplot(8, 1, 6)
    plt.plot(second, integration3, color='green')
    plt.xlabel("second", fontsize=30)
    plt.ylabel("integration3", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplot(8, 1, 7)
    plt.plot(second, voltage4, color='darkorange')
    plt.xlabel("second", fontsize=12)
    plt.ylabel("voltage4", fontsize=12)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplot(8, 1, 8)
    plt.plot(second, integration4, color='darkorange')
    plt.xlabel("second", fontsize=30)
    plt.ylabel("integration4", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    save_fig(file_name)


# 转换为一维矩阵
# dataframe: DataFrame
# return: ndarray 1D
def turn_to_1D(dataframe):
    dataframe = dataframe.drop('second', axis=0)
    return dataframe.values.flatten()
