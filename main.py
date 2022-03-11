import tensorflow as tf
import os
import tensorflow.python.util.deprecation as deprecation
import sys
import argparse

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

    return parser.parse_args(argv)


def main(args):
    batch_size = config.batch_size
    total_acc = 0

    print("\n------------------------------------")
    print("Dataset - '{}' loaded!".format(args.dataset_path))

    test_data, test_label = read_dataset("dataset/" + args.dataset_path + ".xlsx")
    print("------------------------------------")

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


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    sys.exit(0)
