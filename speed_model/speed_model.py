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

def multilayer_lstm_graph_dynamic_rnn(
	    state_size=100,
	    num_inputs=3,
	    num_classes=1,
	    num_steps=500,
	    num_layers=1,
	    network_type="LSTM"
    ):

    x = tf.placeholder(tf.float32, [None, num_steps, num_inputs], name='X')
    y = tf.placeholder(tf.float32, [None, num_classes], name='Y')

    dropout_keep_prob = tf.placeholder(tf.float32, [], name='dropout_keep_prob')
    # Equal dropout on all inputs and outputs of a multi-layered RNN.
    # There are three types of networks implemented: LSTM, GRU and LSTM with 
    # Layer-Normalization.
    if network_type == "LSTM":
        cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout_keep_prob)
    elif network_type == "GRU":
        cell = tf.nn.rnn_cell.GRUCell(state_size)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout_keep_prob)
    elif network_type == "customized_layer_norm_LSTM":
        cell = LayerNormLSTMCell(state_size)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout_keep_prob)
    
    multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
    init_state = multi_cell.zero_state(batch_size, dtype=tf.float32)
    # rnn_outputs: [batch_size, num_steps, state_size].
    # final_state: [batch_size, hidden_layers * state_size].
    rnn_outputs, final_state = tf.nn.dynamic_rnn(multi_cell, x, initial_state=init_state)
    # Unstack the rnn_outputs to list [num_steps * [batch_size, state_size]].
    # Get the last element in the list, that is the last output.
    # last_output [batch_size, state_size].
    last_output = tf.unstack(rnn_outputs, axis=1)[-1]

    # Flatten the y matrix into a list.
    # y_reshaped [batch_size * num_classes].
    # y_reshaped = tf.reshape(y, [-1])

    z1 = tf.nn.relu(fc_layer(last_output, 100, 10))
    # z2 = tf.nn.relu(fc_layer(z1, 30, 6))

    with tf.variable_scope('regression'):
        W = tf.get_variable('W', [10, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(z1, W) + b

    cost = tf.reduce_mean(tf.square(logits - y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    error = tf.reduce_mean(abs(tf.truediv(logits, y) - 1))

    return dict(
        x = x,
        y = y,
        rnn_outputs = rnn_outputs,
        batch_size = batch_size,
        final_state = final_state,
        total_loss = cost,
        train_step = train_step,
        learning_rate = learning_rate,
        error = error,
        logits = logits,
        dropout_keep_prob = dropout_keep_prob
    )


def fc_layer(input, n_input_units, n_output_units):
    W = tf.Variable(tf.random_normal([n_input_units, n_output_units]))
    b = tf.Variable(tf.random_normal([n_output_units]))
    z = tf.matmul(input, W) + b
    return z


def train_network(
        g, 
        num_epochs,
        num_steps=200, 
        batch_size=32,
        verbose=True
    ):

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        learning_rate_change_1 = False
        learning_rate_change_2 = False
        # ----------------------------------------------------------------------
        # Training
        # ----------------------------------------------------------------------
        for epoch_count in range(num_epochs):
            training_loss = 0
            train_error = 0
            for index, (X, Y) in enumerate(zip(X_train, y_train)):
                feed_dict_train = {g["batch_size"]: batch_size, g["learning_rate"]: 5e-3, g["dropout_keep_prob"]: 0.8, g['x']: [X], g['y']: [[Y]]}
                training_loss_, _, train_error_ = \
                    sess.run(
                        [g['total_loss'], g["train_step"], g["error"]],
                        feed_dict_train
                    )
                
                if not learning_rate_change_1 and train_error_ < 0.1:
                    feed_dict_train[g["learning_rate"]] = 1e-3
                    learning_rate_change_1 = True
                if not learning_rate_change_2 and train_error_ < 0.01:
                    feed_dict_train[g["learning_rate"]] = 1e-4
                    learning_rate_change_2 = True

                training_loss += training_loss_
                train_error += train_error_

                if index % 15 == 0:
                    feed_dict_test = {g["batch_size"]: len_test, g["dropout_keep_prob"]: 1, g["x"]: X_test, g["y"]: y_test}
                    test_error = sess.run(g["error"], feed_dict_test)
                    print("Training error: {}".format(train_error / 20))
                    print("Testing error: {}".format(test_error))
                    train_error = 0

                    sample_test = random.randint(0, len_test)
                    feed_dict_sample = {g["batch_size"]: 1, g["dropout_keep_prob"]: 1, g["x"]: [X_test[sample_test]], g["y"]: [y_test[sample_test]]}
                    logit = sess.run(g["logits"], feed_dict_sample)
                    print(logit[0][0])
                    print(y_test[sample_test][0])
                    print()


def update_progress(percentage):
    barLength = 30
    status = ""
    if not isinstance(percentage, float):
        percentage = float(percentage)
    if percentage >= 1:
        percentage = 1
        status = "Done!\n"
    block = int(round(barLength * percentage))
    progress = "=" * block + " " * (barLength - block)
    # "\r" allows us to rewrite the output in the terminal.
    text = "\rProgress: [{0}] {1}% {2}"\
        .format(progress, round(percentage * 100, 4), status)
    sys.stdout.write(text)
    sys.stdout.flush()


if __name__ == "__main__":
    with open("../pogo_data_generation/speed_model.pickle", "rb") as pickle_save:
        X_train, X_test, y_train, y_test = pickle.load(pickle_save)

    epoch = 4
    batch_size = 1
    timestep = len(X_train[0])
    len_test = len(X_test)
    y_test = [[y] for y in y_test]
    network_type = ["LSTM", "GRU", "LSTM_LN", "customized_layer_norm_LSTM"]
    
    for i in range(4):
        if i != 0:
            continue
        chosen_network_type = network_type[i]
        save_directory = "save/{}/".format(chosen_network_type)
        
        # ----------------------------------------------------------------------
        # Training and testing.
        # ----------------------------------------------------------------------
        g_train = multilayer_lstm_graph_dynamic_rnn(
            network_type=chosen_network_type,
            num_steps=timestep
        )
        train_network(
            g_train,
            epoch,
            num_steps=timestep,
            batch_size=batch_size

        )
        summary = "Epoch {}".format(epoch)
        print(summary)
