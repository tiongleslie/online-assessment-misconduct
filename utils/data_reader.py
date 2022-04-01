import pandas as pd
import numpy as np


def read_dataset(filename):
    data = pd.read_excel(open(filename, 'rb'), sheet_name='Sheet1')
    df = pd.DataFrame(data, columns=['Q1M', 'Q2M', 'Q3M', 'Q4M', 'Q5M', 'Q6M', 'Q7M', 'Q8M', 'Q9M', 'Q10M',
                                     'Q11M', 'Q12M', 'Q13M', 'Q14M', 'Q15M', 'Q16M', 'Q17M', 'Q18M', 'Q19M',
                                     'Q20M'])
    numpy_data = df.to_numpy()
    numpy_data = np.where(numpy_data >= 1, 1, 0)

    behaviour = np.zeros((numpy_data.shape[0], 3))

    df = pd.DataFrame(data, columns=['Scores', 'Time'])
    attitute = df.to_numpy()

    for i in range(attitute.shape[0]):
        if 10 < attitute[i][1] <= 30:
            behaviour[i][1] = 1
        elif attitute[i][1] <= 10:
            behaviour[i][0] = 1
        else:
            behaviour[i][2] = 1

    data_behaviour = np.concatenate((numpy_data, behaviour), axis=1)

    label = np.zeros((numpy_data.shape[0], 2))
    for i in range(attitute.shape[0]):
        if attitute[i][0] > 75 and behaviour[i][0] == 1:
            label[i][0] = 1
        elif attitute[i][0] > 75 and behaviour[i][2] == 1:
            label[i][0] = 1
        else:
            label[i][1] = 1

    return np.array(data_behaviour), np.array(label)


def read_ip_data(filename):
    data = pd.read_excel(open(filename, 'rb'), sheet_name='Sheet1')
    df = pd.DataFrame(data, columns=['IP'])

    numpy_data = df.to_numpy()
    IP = np.asarray(sorted(numpy_data))

    return IP
