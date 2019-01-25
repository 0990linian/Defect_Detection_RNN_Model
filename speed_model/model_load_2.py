#-------------------------------------------------------------------------------
# Disable AVX warning.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# ------------------------------------------------------------------------------
import sys
import random
import pickle
import numpy as np
import tensorflow as tf

# Disable tensorflow warning about deprecated commands.
tf.logging.set_verbosity(tf.logging.ERROR)


def test_model():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('train_error_1_10/sample.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('train_error_1_10'))
        graph = tf.get_default_graph()
        batch_size = graph.get_tensor_by_name("batch_size:0")
        dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
        x = graph.get_tensor_by_name("X:0")
        for _ in range(10):
            sample_test = random.randint(0, len_test)
            feed_dict_sample = {batch_size: 1, dropout_keep_prob: 1, x: [X_test[sample_test]]}
            logits = graph.get_tensor_by_name("logits:0")
            prediction = sess.run(logits, feed_dict_sample)
            print(prediction[0][0])
            print(y_test[sample_test][0])
            print()


if __name__ == "__main__":
    with open("../pogo_data_generation/speed_model_stand.pickle", "rb") as pickle_save:
        X_train, X_test, y_train, y_test = pickle.load(pickle_save)

    batch_size = 1
    timestep = len(X_test[0])
    len_test = len(X_test)
    y_test = [[y / 10000] for y in y_test]
    network_type = ["LSTM", "GRU", "LSTM_LN", "customized_layer_norm_LSTM"]
    
    for i in range(4):
        if i != 0:
            continue
        chosen_network_type = network_type[i]
        test_model()
