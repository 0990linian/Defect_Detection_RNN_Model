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
from speed_model import multilayer_lstm_graph_dynamic_rnn

# Disable tensorflow warning about deprecated commands.
tf.logging.set_verbosity(tf.logging.ERROR)


def test_model(g):
	with tf.Session() as sess:
		g["saver"].restore(sess, "train_error_1_10/sample.ckpt")
		for _ in range(10):
			sample_test = random.randint(0, len_test)
			feed_dict_sample = {g["batch_size"]: 1, g["dropout_keep_prob"]: 1, g["x"]: [X_test[sample_test]]}
			logit = sess.run(g["logits"], feed_dict_sample)
			print(logit[0][0])
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
        
        # ----------------------------------------------------------------------
        # Training and testing.
        # ----------------------------------------------------------------------
        g_test = multilayer_lstm_graph_dynamic_rnn(
            network_type=chosen_network_type,
            num_steps=timestep
        )
        test_model(
            g_test
        )