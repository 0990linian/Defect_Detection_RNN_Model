#-------------------------------------------------------------------------------
# Disable AVX warning.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# ------------------------------------------------------------------------------
import sys
import random
import pickle
import sqlite3
import numpy as np
import tensorflow as tf

# Disable tensorflow warning about deprecated commands.
tf.logging.set_verbosity(tf.logging.ERROR)


def create_database(
        database
    ):
    db_connection = sqlite3.connect(database)
    cursor = db_connection.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS table_list ("
        "epoch INT, "
        "num_layers INT, "
        "state_size INT, "
        "cell_type VARCHAR(255), "
        "dropout_keep_prob real, "
        "batch_size INT, "
        "saving_directory VARCHAR(255), "
        "table_name VARCHAR(255))"
    )

    db_connection.commit()
    return db_connection


def add_table_in_database(
        database,
        epoch,
        num_layers,
        state_size,
        cell_type,
        dropout_keep_prob,
        batch_size,
        saving_directory,
        table_name
    ):
    db_connection = sqlite3.connect(database)
    cursor = db_connection.cursor()
    cursor.execute(
        "INSERT INTO table_list VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (epoch, num_layers, state_size, cell_type, dropout_keep_prob, batch_size, saving_directory, table_name)
    )
    cursor.execute(
        "CREATE TABLE {} (train_error real, test_error real)"
        .format(table_name)
    )
    db_connection.commit()


def multilayer_lstm_graph_dynamic_rnn(
        state_size=100,
        num_inputs=3,
        num_classes=1,
        num_steps=500,
        num_layers=1,
        cell_type="LSTM"
    ):

    x = tf.placeholder(tf.float32, [None, num_steps, num_inputs], name='X')
    y = tf.placeholder(tf.float32, [None, num_classes], name='Y')

    dropout_keep_prob = tf.placeholder(tf.float32, [], name='dropout_keep_prob')
    # Equal dropout on all inputs and outputs of a multi-layered RNN.
    # There are three types of networks implemented: LSTM, GRU and LSTM with 
    # Layer-Normalization.
    if cell_type == "LSTM":
        cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout_keep_prob)
    elif cell_type == "GRU":
        cell = tf.nn.rnn_cell.GRUCell(state_size)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout_keep_prob)
    elif cell_type == "customized_layer_norm_LSTM":
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

    z1 = tf.nn.tanh(fc_layer(last_output, 100, 10))
    # z2 = tf.nn.relu(fc_layer(z1, 30, 6))

    with tf.variable_scope('regression'):
        W = tf.get_variable('W', [10, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.add(tf.matmul(z1, W), b, name="logits")

    cost = tf.reduce_mean(tf.square(logits - y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    error = tf.reduce_mean(abs(tf.truediv(logits, y) - 1), name="error")
    saver = tf.train.Saver()

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
        dropout_keep_prob = dropout_keep_prob,
        saver = saver
    )


def fc_layer(input, n_input_units, n_output_units):
    with tf.variable_scope('fc_layer'):
        W = tf.Variable(tf.random_normal([n_input_units, n_output_units]), name="W")
        b = tf.Variable(tf.random_normal([n_output_units]), name="b")
    z = tf.matmul(input, W) + b
    return z


def train_network(
        g, 
        num_epochs,
        database,
        table,
        num_steps=200, 
        batch_size=32
    ):
    db_connection = sqlite3.connect(database)
    cursor = db_connection.cursor()
    learning_rate_change = False
    feed_dict_train = {g["batch_size"]: batch_size, g["learning_rate"]: 1e-3, g["dropout_keep_prob"]: 0.8}
    iter_num = 30

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch_count in range(num_epochs):
            train_error = 0
        
            for index, (X, Y) in enumerate(zip(X_train, y_train), 1):
                feed_dict_train[g['x']] = [X]
                feed_dict_train[g['y']] = [[Y / 10000]]
                _, train_error_ = \
                    sess.run(
                        [g["train_step"], g["error"]],
                        feed_dict_train
                    )
                train_error += train_error_

                if index % iter_num == 0:
                    feed_dict_test = {g["batch_size"]: len_test, g["dropout_keep_prob"]: 1, g["x"]: X_test, g["y"]: y_test}
                    test_error = sess.run(g["error"], feed_dict_test)
                    train_message = "Training error: {}".format(train_error / iter_num)
                    test_message = "Testing error: {}\n".format(test_error)
                    print(train_message)
                    print(test_message)
                    cursor.execute(
                        "INSERT INTO {} VALUES (?, ?)".format(table_name), 
                        (train_error / iter_num, test_error)
                    )
                    train_error = 0
                    
                    if not learning_rate_change and test_error < 0.15:
                        print("learning_rate changed to 1e-4")
                        feed_dict_train[g["learning_rate"]] = 1e-4
                        learning_rate_change = True
                        g["saver"].save(sess, saving_directory + "15/model.ckpt")

                    if test_error < 0.1 and not os.path.isdir(saving_directory + "10"):
                        g["saver"].save(sess, saving_directory + "10/model.ckpt")
                    elif test_error < 0.09 and not os.path.isdir(saving_directory + "9"):
                        g["saver"].save(sess, saving_directory + "9/model.ckpt")
                    elif test_error < 0.08 and not os.path.isdir(saving_directory + "8"):
                        g["saver"].save(sess, saving_directory + "8/model.ckpt")
                    elif test_error < 0.07 and not os.path.isdir(saving_directory + "7"):
                        g["saver"].save(sess, saving_directory + "5/model.ckpt")

            db_connection.commit()


if __name__ == "__main__":
    with open("../pogo_data_generation/speed_model_stand.pickle", "rb") as pickle_save:
        X_train, X_test, y_train, y_test = pickle.load(pickle_save)

    database = "speed_model_train.db"
    epoch = 5
    batch_size = 1
    num_layers = 1
    state_size = 100
    dropout_keep_prob = 1
    timestep = len(X_train[0])
    len_test = len(X_test)
    y_test = [[y / 10000] for y in y_test]
    cell_type = ["LSTM", "GRU", "LSTM_LN", "customized_layer_norm_LSTM"]
    saving_directory = "1_24_test_error_"

    create_database(database)

    for i in range(4):
        if i != 0:
            continue
        chosen_cell_type = cell_type[i]

        table_name = "1_24_00{}".format(i)
        add_table_in_database(
            database,
            epoch,
            num_layers,
            state_size,
            chosen_cell_type,
            dropout_keep_prob,
            batch_size,
            saving_directory,
            table_name
        )
        # ----------------------------------------------------------------------
        # Training and testing.
        # ----------------------------------------------------------------------
        g_train = multilayer_lstm_graph_dynamic_rnn(
            state_size=state_size,
            cell_type=chosen_cell_type,
            num_steps=timestep,
            num_layers=num_layers
        )
        train_network(
            g_train,
            epoch,
            database,
            table_name,
            num_steps=timestep,
            batch_size=batch_size
        )
