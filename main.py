import tensorflow as tf
import os
import tensorflow.python.util.deprecation as deprecation
import sys
import argparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from model.DenseLSTM import DenseLSTM
from utils.data_reader import *
import config

deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='Dataset path for evaluation.', default='Data1-Spring2020')
    parser.add_argument('--mode', help='Evaluation mode (1: Behaviour classification, '
                                       '2: Internet Protocol Detection,'
                                       '3: Both', default=3)

    return parser.parse_args(argv)


def behaviour_classify(filename):
    batch_size = config.batch_size
    total_acc = 0

    test_data, test_label = read_dataset("dataset/" + filename + ".xlsx")

    with tf.Session(config=tf_config) as sess:
        model = DenseLSTM(sess, batch=batch_size, class_num=config.class_num, N=config.N)
        saver = tf.train.Saver()
        saver.restore(sess, config.model_path)
        batch_arr_test = range(0, len(test_data), batch_size)
        max_test = batch_arr_test[len(batch_arr_test) - 1]

        for k in range(len(batch_arr_test)):
            if batch_arr_test[k] == max_test:
                test_image_batch = test_data[batch_arr_test[k]:len(test_data)]
                test_label_batch = test_label[batch_arr_test[k]:len(test_data)]
                test_image_batch = np.reshape(test_image_batch, (len(test_image_batch), 1, config.N))
            else:
                test_image_batch = test_data[batch_arr_test[k]:batch_arr_test[k] + batch_size]
                test_label_batch = test_label[batch_arr_test[k]:batch_arr_test[k] + batch_size]
                test_image_batch = np.reshape(test_image_batch, (batch_size, 1, config.N))

            tacc, _ = model.test_step(test_image_batch, test_label_batch)
            total_acc += tacc

            if k == (len(batch_arr_test) - 1):
                print(
                    "\nAccuracy: %.2f(%%)" % (total_acc / len(batch_arr_test) * 100)
                )
        print("------------------------------------")
        sess.close()


def IP_detector(filename):
    test_data = read_ip_data("dataset/" + filename + ".xlsx")

    IP_records = []

    for i in range(test_data.shape[0]):
        IP = str(test_data[i]).replace("'", '')
        IP = IP.replace("[", "")
        IP = IP.replace("]", "").split(".")
        IP_records.append([int(IP[0]), int(IP[1]), int(IP[2]), int(IP[3])])

    IP_records = np.asarray(IP_records)
    IP_ = []

    for i in range(IP_records.shape[0]):
        for j in range(IP_records.shape[0]):
            if i == j:
                continue

            if IP_records[i, 2] == IP_records[j, 2] and IP_records[i, 1] == IP_records[j, 1] and \
                    IP_records[i, 0] == IP_records[j, 0]:
                IP_.append([IP_records[i, 0], IP_records[i, 1], IP_records[i, 2], IP_records[i, 3], 1])
            elif IP_records[i, 3] == IP_records[j, 3] and IP_records[i, 2] == IP_records[j, 2] and \
                    IP_records[i, 1] == IP_records[j, 1] and IP_records[i, 0] == IP_records[j, 0]:
                IP_.append([IP_records[i, 0], IP_records[i, 1], IP_records[i, 2], IP_records[i, 3], 2])
            else:
                IP_.append([IP_records[i, 0], IP_records[i, 1], IP_records[i, 2], IP_records[i, 3], 0])
    IP_ = np.asarray(IP_)

    Final_PCA = PCA(n_components=2)
    Final_PCA.fit(np.array(IP_))
    cluster_ = Final_PCA.transform(IP_)

    result_0 = []
    result_1 = []
    IP_suspect = []

    for i in range(IP_.shape[0]):
        if IP_[i, 4] == 1 or IP_[i, 4] == 2:
            result_0.append([cluster_[i, 0], cluster_[i, 1], IP_[i, 4]])
            IP_suspect.append([IP_[i, 0], IP_[i, 1], IP_[i, 2], IP_[i, 3]])
        elif IP_[i, 4] == 0:
            result_1.append([cluster_[i, 0], cluster_[i, 1], IP_[i, 4]])
    result_0 = np.asarray(result_0)
    result_1 = np.asarray(result_1)

    print("Suspected IP:")
    print("------------------------------------")
    IP_suspect = np.asarray(IP_suspect)
    for i in range(IP_suspect.shape[0]):
        print("\t{}: {}.{}.{}.{}".format((i+1), IP_suspect[i, 0], IP_suspect[i, 1], IP_suspect[i, 2], IP_suspect[i, 3]))

    plt.rcParams["figure.figsize"] = (6, 6)
    plt.scatter(result_1[:, 0], result_1[:, 1], s=10, marker="o", color='lime', label="Normal IP Addresses")
    plt.scatter(result_0[:, 0], result_0[:, 1], s=40, marker="^", color='red', label="Suspected IP Addresses")
    plt.legend(loc=1)
    plt.show()

    print("------------------------------------")


def main(args):
    if args.mode is 1:
        print("\n------------------------------------")
        print("Dataset - '{}' loaded!".format(args.dataset_path))
        print("------------------------------------")
        behaviour_classify(filename=args.dataset_path)
    elif args.mode is 2:
        print("\n------------------------------------")
        print("Dataset - '{}' loaded!".format(args.dataset_path))
        print("------------------------------------")
        IP_detector(filename=args.dataset_path)
    elif args.mode is 3:
        print("\n------------------------------------")
        print("Dataset - '{}' loaded!".format(args.dataset_path))
        print("------------------------------------")
        IP_detector(filename=args.dataset_path)
        behaviour_classify(filename=args.dataset_path)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    sys.exit(0)
