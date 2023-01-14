from basic import *

PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "Datasets_Processed")


# 加载所有的原信号文件
# return: ndarray Dataframe
def load_all_data():
    dataframe = [load_data("scope_29"), load_data("scope_30"), load_data("scope_31"), load_data("scope_32"),
                 load_data("scope_34"), load_data("scope_35"), load_data("scope_36"), load_data("scope_37"),
                 load_data("scope_38"), load_data("scope_39"), load_data("scope_40"), load_data("scope_41"),
                 load_data("scope_42"), load_data("scope_43"), load_data("scope_44"), load_data("scope_45"),
                 load_data("scope_47"), load_data("scope_48"), load_data("scope_49"), load_data("scope_50"),
                 load_data("scope_51"), load_data("scope_52"), load_data("scope_53"), load_data("scope_54"),
                 load_data("scope_55"), load_data("scope_56"), load_data("scope_57"), load_data("scope_58"),
                 load_data("scope_61"), load_data("scope_62"), load_data("scope_63")]
    return dataframe


# 数据预处理pipeline
# dataframe: Dataframe
# return: ndarray 1D, ndarray 1D
def data_preprocessing(multiple1=11, multiple2=11, multiple3=11, usage="train"):
    dataframe0 = load_all_data()
    length = np.zeros(len(dataframe0))
    for i in range(len(dataframe0)):
        dataframe0[i], length[i] = signal_cut_down(voltage_normalization(lower_frequency(dataframe0[i])))
    max_length = max(length)

    # 将所有数据填充到统一长度，并进行移相
    # _dataframe: DataFrame, data_length: int
    # return: ndarray DataFrame
    def fill_in(_dataframe, data_length, times=5, bias=10):
        if times + 5 > bias:
            bias = times + 5
        data = _dataframe.values
        second = data[0]
        volt = [data[1], data[2], data[3], data[4]]
        new_second = [k * second[1] for k in range(data_length + bias)]
        new_second = pd.DataFrame(new_second, columns=['second'], dtype=float)
        new_volt = [np.zeros(data_length + bias), np.zeros(data_length + bias),
                    np.zeros(data_length + bias), np.zeros(data_length + bias)]
        times_volt = []
        start = np.zeros(times)
        for k in range(times):
            start[k] = np.random.randint(0, data_length - len(second) + bias)
            for p in range(len(second)):
                for q in range(len(new_volt)):
                    new_volt[q][p + int(start[k])] = volt[q][p]
            new_volt1 = pd.DataFrame(new_volt[0], columns=['voltage1'], dtype=float)
            new_volt2 = pd.DataFrame(new_volt[1], columns=['voltage2'], dtype=float)
            new_volt3 = pd.DataFrame(new_volt[2], columns=['voltage3'], dtype=float)
            new_volt4 = pd.DataFrame(new_volt[3], columns=['voltage4'], dtype=float)
            times_volt.append(pd.concat([new_second, new_volt1, new_volt2, new_volt3, new_volt4], axis=1).T)
        return times_volt

    dataframe1 = []
    for i in range(len(dataframe0)):
        multiple_volt = fill_in(dataframe0[i], int(max_length), times=multiple1)
        for t in range(multiple1):
            dataframe1.append(multiple_volt[t])

    dataframe2 = []
    for i in range(len(dataframe1)):
        multiple_volt = add_noise(dataframe1[i], multiple=multiple2)
        for t in range(multiple2):
            dataframe2.append(multiple_volt[t])

    dataframe3 = []
    for i in range(len(dataframe2)):
        multiple_volt = integration_linear_bias(voltage_integration(dataframe2[i]), multiple=multiple3)
        for t in range(multiple3):
            output_path = os.path.join(PROCESSED_DATA_PATH, usage, str(int(i / (multiple1 * multiple2)) * 100000 +
                                                                       (int(i / multiple2) % multiple1) * 1000 +
                                                                       (i % multiple2) * 10 + t) + ".csv")
            multiple_volt[t].T.to_csv(output_path, sep=',', index=False, header=True)
            dataframe3.append(multiple_volt[t])

    dataset = []
    for i in range(len(dataframe3)):
        dataset.append(turn_to_1D(dataframe3[i]))
    dataset = np.array(dataset)
    dataset.reshape(len(dataframe3), -1)

    label = []
    for i in range(4):
        if i < 3:
            for j in range(multiple1 * multiple2 * multiple3 * 8):
                label.append(i)
        else:
            for j in range(multiple1 * multiple2 * multiple3 * 7):
                label.append(i)
    label = np.array(label)
    label.reshape(1, -1)

    return dataset, label
