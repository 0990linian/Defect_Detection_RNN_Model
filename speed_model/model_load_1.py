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
from speed_model import rnn_model

# Disable tensorflow warning about deprecated commands.
tf.logging.set_verbosity(tf.logging.ERROR)


def test_model(model):
	with tf.Session() as sess:
		model["saver"].restore(sess, "1_25_test_error_9/model.ckpt")
		for _ in range(10):
			sample_test = random.randint(0, len_test)
			feed_dict_sample = {model["batch_size"]: 1, model["dropout_keep_prob"]: 1, model["x"]: [X_test[sample_test]]}
			logit = sess.run(model["logits"], feed_dict_sample)
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
        model_strcuture = rnn_model(
            cell_type=chosen_network_type,
            num_steps=timestep
        )
        test_model(model_strcuture)
