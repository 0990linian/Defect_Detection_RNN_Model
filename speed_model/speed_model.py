################################################################################
# This script aims to build an RNN model that could predict the ultrasonic speed 
# in the inclusion of the component.
################################################################################
import os
import sys
import random
import pickle
import sqlite3
import numpy as np
import tensorflow as tf

# Disable AVX warning.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Disable tensorflow warning about deprecated commands.
tf.logging.set_verbosity(tf.logging.ERROR)


def main():
    """
    Main function of this script

    Description:
        - Create database for the speed model if it does not exist.
        - Add table for specific training process to record training data.
        - Build the RNN model to predict speed in inclusion.
        - Train the model on provided dataset.
    
    Args:
        None

    Returns:
        None
    """
    create_database(database)
    cell_type = ["LSTM", "GRU"]

    for i in range(4):
        if i != 0:
            continue
        chosen_cell_type = cell_type[i]

        table_name = "run_1_25_00_{}".format(i)
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
        # Model building and training.
        # ----------------------------------------------------------------------
        speed_model = rnn_model(
            state_size=state_size,
            cell_type=chosen_cell_type,
            num_steps=timestep,
            num_layers=num_layers
        )
        train_network(
            speed_model,
            epoch,
            database,
            table_name,
            X_train,
            y_train,
            num_steps=timestep,
            batch_size=batch_size
        )


def create_database(database):
    """
    Create speed model database and the high level table.
    """
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
    """
    Add a table into database to record training process data through the 
    training stage.
    """
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


def rnn_model(
        state_size=100,
        num_inputs=3,
        num_classes=1,
        num_steps=500,
        num_layers=1,
        cell_type="LSTM"
    ):
    """
    Build the RNN model for the speed prediction task.

    Description:
        Build the RNN model based on different input cell_types.  
            Choosable options are LSTM and GRU.
        batch_size, learning_rate and dropout_keep_prob are set as 
            placeholders, so different values could be input during training and 
            testing stages.

            - batch_size: In this script, the training batch of 1 and testing
                batch of all test samples are used.
            - learning_rate: Due to the big change of error value during
                training, adaptive learning_rate is applied.  The initial
                learning rate is set to be 1e-3.  When the test error is less
                than 15%, reduce the error_rate to 1e-4.
            - dropout_keep_prob: dropout could be applied during training, but
                it must be 1 during testing.
    """
    x = tf.placeholder(tf.float32, [None, num_steps, num_inputs], name='X')
    y = tf.placeholder(tf.float32, [None, num_classes], name='Y')
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
    dropout_keep_prob = tf.placeholder(tf.float32, [], name='dropout_keep_prob')

    if cell_type == "LSTM":
        cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout_keep_prob)
    elif cell_type == "GRU":
        cell = tf.nn.rnn_cell.GRUCell(state_size)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout_keep_prob)
    
    multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    init_state = multi_cell.zero_state(batch_size, dtype=tf.float32)

    # rnn_outputs: [batch_size, num_steps, state_size].
    # final_state: [batch_size, hidden_layers * state_size].
    rnn_outputs, final_state = tf.nn.dynamic_rnn(multi_cell, x, initial_state=init_state)
    
    # Unstack the rnn_outputs to list [num_steps * [batch_size, state_size]].
    # Get the last element in the list, that is the last output.
    # last_output [batch_size, state_size].
    last_output = tf.unstack(rnn_outputs, axis=1)[-1]
 
    fc_output_1 = tf.nn.tanh(fc_layer(last_output, 100, 10, "fc_1"))
    fc_output_2 = fc_layer(fc_output_1, 10, num_classes, "fc_2")
    logits = tf.identity(fc_output_2, name="logits")

    cost = tf.reduce_mean(tf.square(logits - y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    error = tf.reduce_mean(abs(tf.truediv(logits, y) - 1), name="error")
    saver = tf.train.Saver()

    return dict(
        x = x,
        y = y,
        saver = saver,
        error = error,
        logits = logits,
        batch_size = batch_size,
        train_step = train_step,
        learning_rate = learning_rate,
        dropout_keep_prob = dropout_keep_prob
    )


def fc_layer(input, num_input, num_output, scope_name):
    """
    Define a fully connected layer.
    """
    with tf.variable_scope(scope_name):
        W = tf.Variable(tf.random_normal([num_input, num_output]), name="W")
        b = tf.Variable(tf.random_normal([num_output]), name="b")
        z = tf.add(tf.matmul(input, W), b, name="z")
    return z


def train_network(
        model, 
        num_epochs,
        database,
        table_name,
        X_train,
        y_train,
        num_steps=200, 
        batch_size=32
    ):
    """
    Train the built RNN model.

    Description:
        - Since the original data for input is at value around 1e-9 while the 
            speed is around 1e5, input data rescale is applied, which is done in 
            matlab function.  Reduce the output speed scale to between 0 and 1 
            is applied in this function, to provide more stable predictions.
        - For every certain number of iterations, the training and testing error 
            are shown to provide an insight into the training process.  These 
            data is also recorded in the database.
        - Save the model when test error is below 10%, 9%, 8% and 7%.
    """
    db_connection = sqlite3.connect(database)
    cursor = db_connection.cursor()
    
    iter_num = 5
    learning_rate_change = False
    
    feed_dict_train = {
        model["batch_size"]: batch_size,
        model["learning_rate"]: 1e-3,
        model["dropout_keep_prob"]: 0.8
    }
    feed_dict_test = {
        model["batch_size"]: len_test,
        model["dropout_keep_prob"]: 1
    }

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch_count in range(num_epochs):
            train_error = 0
        
            for index, (X, Y) in enumerate(zip(X_train, y_train), 1):
                feed_dict_train[model['x']] = [X]
                feed_dict_train[model['y']] = [[Y / 10000]]
                _, train_error_ = \
                    sess.run(
                        [model["train_step"], model["error"]],
                        feed_dict_train
                    )
                train_error += train_error_

                if index % iter_num == 0:
                    feed_dict_test[model["x"]] = X_test
                    feed_dict_test[model["y"]] = y_test
                    test_error = sess.run(model["error"], feed_dict_test)
                    print("Training error: {}".format(train_error / iter_num))
                    print("Testing error: {}\n".format(test_error))
                    cursor.execute(
                        "INSERT INTO {} VALUES (?, ?)".format(table_name), 
                        (train_error / iter_num, float(test_error))
                    )
                    db_connection.commit()
                    train_error = 0
                    
                    if not learning_rate_change and test_error < 0.15:
                        print("learning_rate changed to 1e-4")
                        feed_dict_train[model["learning_rate"]] = 1e-4
                        learning_rate_change = True

                    if test_error < 0.1 and not os.path.isdir(saving_directory + "10"):
                        model["saver"].save(sess, saving_directory + "10/model.ckpt")
                    elif test_error < 0.09 and not os.path.isdir(saving_directory + "9"):
                        model["saver"].save(sess, saving_directory + "9/model.ckpt")
                    elif test_error < 0.08 and not os.path.isdir(saving_directory + "8"):
                        model["saver"].save(sess, saving_directory + "8/model.ckpt")
                    elif test_error < 0.07 and not os.path.isdir(saving_directory + "7"):
                        model["saver"].save(sess, saving_directory + "5/model.ckpt")

            db_connection.commit()


if __name__ == "__main__":
    # Load data and define global variables.
    model_pickle = "../pogo_data_generation/speed_model_stand.pickle"
    with open(model_pickle, "rb") as pickle_save:
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
    saving_directory = "1_25_test_error_"

    main()
