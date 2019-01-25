################################################################################
# This script aims to load the trained model and test the effectiveness using 
# testing data.
################################################################################
import os
import sys
import random
import pickle
import numpy as np
import tensorflow as tf

# Disable AVX warning.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Disable tensorflow warning about deprecated commands.
tf.logging.set_verbosity(tf.logging.ERROR)


def test_model(save_dir):
    """
    Test the effectiveness of the built model without building the structure of 
    the model again.
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_dir + "/model.ckpt.meta")
        saver.restore(sess, tf.train.latest_checkpoint(save_dir))
        graph = tf.get_default_graph()
        batch_size = graph.get_tensor_by_name("batch_size:0")
        dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
        x = graph.get_tensor_by_name("X:0")
        
        for _ in range(10):
            sample_test = random.randint(0, len_test - 1)
            feed_dict_sample = {batch_size: 1, dropout_keep_prob: 1, x: [X_test[sample_test]]}
            logits = graph.get_tensor_by_name("logits:0")
            prediction = sess.run(logits, feed_dict_sample)
            print(prediction[0][0])
            print(y_test[sample_test][0])
            print()


if __name__ == "__main__":
    model_pickle = "../pogo_data_generation/speed_model_stand.pickle"
    with open(model_pickle, "rb") as pickle_save:
        X_train, X_test, y_train, y_test = pickle.load(pickle_save)

    batch_size = 1
    timestep = len(X_test[0])
    len_test = len(X_test)
    y_test = [[y / 10000] for y in y_test]
    network_type = ["LSTM", "GRU"]
    
    for i in range(2):
        if i != 0:
            continue
        chosen_network_type = network_type[i]
        save_dir = "1_25_01_test_error_9"
        test_model(save_dir)
