################################################################################
# This script aims to build an RNN model that could predict the inclusion's 
# shape in the component.
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


def main(prev_sess=None):
    """
    Main function of this script

    Description:
        - Create database for the speed model if it does not exist.
        - Add table for specific training process to record training data.
        - Build the RNN model to predict speed in inclusion.
        - Train the model on provided dataset.
    """
    tf.reset_default_graph()
    create_database()
    add_table_in_database()

    shape_model = rnn_model()
    train_network(shape_model, prev_sess)


def create_database():
    """
    Create speed model database and the high level table.
    """
    db_connection = sqlite3.connect(database)
    cursor = db_connection.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS table_list ("
        "num_epochs INT, "
        "num_layers INT, "
        "state_size INT, "
        "cell_type VARCHAR(255), "
        "dropout_keep_prob real, "
        "batch_size INT, "
        "table_name VARCHAR(255))"
    )

    db_connection.commit()
    return db_connection


def add_table_in_database():
    """
    Add a table into database to record training process data through the 
    training stage.
    """
    db_connection = sqlite3.connect(database)
    cursor = db_connection.cursor()
    cursor.execute(
        "INSERT INTO table_list VALUES (?, ?, ?, ?, ?, ?, ?)",
        (num_epochs, num_layers, state_size, cell_type, dropout_keep_prob, batch_size, save_dir)
    )
    cursor.execute(
        "CREATE TABLE {} (train_error real, test_error real, cost real)"
        .format(save_dir)
    )
    db_connection.commit()


def reshape_data_for_batch(input_data_list, batch_size):
    
    def form_batch(input_data):
        result_data = []
        batch_data = []
        for index, one_batch in enumerate(input_data, 1):
            batch_data.append(one_batch)
            if index % batch_size == 0:
                result_data.append(batch_data[:])
                batch_data.clear()
        return result_data

    result_data_list = []
    for data in input_data_list:
        result_data_list.append(form_batch(data))
    return result_data_list


def fc_layer(input, num_input, num_output, scope_name):
    """
    Define a fully-connected layer.

    Args:
        input: The input data to this fully-connected layer.
        num_input: The number of nodes in the previous layer.
        num_output: The number of nodes in the current layer.
        scope_name: The scope name for this layer.  The adding of this scope is 
            for the easy management when visualizing on tensorboard in future.

    Returns:
        The result data passed through this layer.
    """
    with tf.variable_scope(scope_name):
        W = tf.Variable(
            tf.random_normal([num_input, num_output]), 
            name="weight"
        )
        b = tf.Variable(tf.random_normal([num_output]), name="bias")
        z = tf.add(tf.matmul(input, W), b, name="linear")
        if activation_type == "relu":
            a = tf.nn.relu(z, name="relu")
        elif activation_type == "tanh":
            a = tf.nn.tanh(z, name="tanh")
        elif activation_type == "leaky_relu":
            a = tf.nn.leaky_relu(z, alpha=0.2, name="leaky_relu")

        tf.summary.histogram("weight", W)
        tf.summary.histogram("bias", b)
        tf.summary.histogram("linear", z)
        tf.summary.histogram("activation", a)
        return a


def output_layer(input, num_input, num_output, scope_name):
    with tf.variable_scope(scope_name):
        W = tf.Variable(
            tf.random_normal([num_input, num_output]), 
            name="weight"
        )
        b = tf.Variable(tf.random_normal([num_output]), name="bias")
        z = tf.add(tf.matmul(input, W), b, name="linear")
        return z


def rnn_model():
    """
    Build the multi-layer RNN model for the speed prediction task.

    Description:
        Build the multi-layer RNN model based on different input cell_types.  
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
    # x: [batch_size, num_steps, num_inputs]
    # y: [batch_size, num_classes]
    x = tf.placeholder(tf.float32, [None, num_steps, num_inputs], name='X')
    y = tf.placeholder(tf.float32, [None, num_classes], name='Y')
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
    dropout_keep_prob = tf.placeholder(tf.float32, [], name='dropout_keep_prob')

    # x_fc_input: [batch_size * num_steps, num_inputs]
    x_fc_input = tf.reshape(x, [-1, num_inputs])
    # fc_output_0: [batch_size * num_steps, state_size]
    fc_output_0 = fc_layer(x_fc_input, num_inputs, state_size, "fc_0")
    # x_rnn_input: [batch_size, num_steps, state_size]
    x_rnn_input = tf.reshape(fc_output_0, [-1, num_steps, state_size])

    if cell_type == "LSTM":
        cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell, 
            input_keep_prob=dropout_keep_prob
        )
    elif cell_type == "GRU":
        cell = tf.nn.rnn_cell.GRUCell(state_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell, 
            input_keep_prob=dropout_keep_prob
        )
    
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(
        [cell] * num_layers, 
        state_is_tuple=True
    )
    init_state = multi_cell.zero_state(batch_size, dtype=tf.float32)

    # rnn_outputs: [batch_size, num_steps, state_size].
    # final_state: [batch_size, hidden_layers * state_size].
    rnn_outputs, final_state = tf.nn.dynamic_rnn(
        multi_cell, 
        x_rnn_input, 
        initial_state=init_state
    )
    
    # Unstack the rnn_outputs to list [num_steps * [batch_size, state_size]].
    # Get the last element in the list, that is the last output.
    # last_output [batch_size, state_size].
    last_output = tf.unstack(rnn_outputs, axis=1)[-1]
 
    fc_output_1 = fc_layer(last_output, state_size, 10, "fc_1")
    fc_output_2 = output_layer(fc_output_1, 10, num_classes, "fc_2")
    logits = tf.identity(fc_output_2, name="logits")

    with tf.name_scope("cost"):
        cost = tf.nn.l2_loss(tf.subtract(logits, y))
        tf.summary.scalar("cost", cost)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.name_scope("error"):
        error = tf.reduce_mean(abs(tf.truediv(logits, y) - 1), name="error")
        tf.summary.scalar("error", error)
    saver = tf.train.Saver()

    return dict(
        x = x,
        y = y,
        cost = cost,
        saver = saver,
        error = error,
        logits = logits,
        batch_size = batch_size,
        train_step = train_step,
        learning_rate = learning_rate,
        dropout_keep_prob = dropout_keep_prob
    )


def train_network(model, prev_sess=None):
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
        - Save the model parameters when test error reaches the minimum value 
            and when the training process is complete.
        - Enable continue training, allowing training from a saved checkpoint.

    Args:
        model: The built RNN model structure.
    """
    db_connection = sqlite3.connect(database)
    cursor = db_connection.cursor()
    
    test_error_min = 1
    learning_rate_change = False
    
    feed_dict_train = {
        model["batch_size"]: batch_size,
        model["learning_rate"]: 1e-3,
        model["dropout_keep_prob"]: dropout_keep_prob
    }
    feed_dict_test = {
        model["batch_size"]: len_test,
        model["dropout_keep_prob"]: 1
    }

    merged_summary = tf.summary.merge_all()
    tb_counter = 0

    with tf.Session() as sess:
        if prev_sess is None:
            sess.run(tf.initialize_all_variables())
            train_writer = tf.summary.FileWriter(
                "tensorboard/{}_train".format(save_dir),
                sess.graph
            )
            test_writer = tf.summary.FileWriter(
                "tensorboard/{}_test".format(save_dir),
                sess.graph
            )
        else:
            model["saver"].restore(sess, tf.train.latest_checkpoint(prev_sess))
        
        for epoch_count in range(num_epochs):
            train_error, cost = 0, 0

            for index, (X, Y) in enumerate(zip(x_train, y_train), 1):
                feed_dict_train[model['x']] = X
                feed_dict_train[model['y']] = Y

                _, train_error_, cost_, summary_train = sess.run(
                    [model["train_step"], model["error"], model["cost"], merged_summary],
                    feed_dict_train
                )
                train_writer.add_summary(summary_train, tb_counter)
                train_error += train_error_
                cost += cost_
                tb_counter += 1

                if index % iter_num == 0:
                    feed_dict_test[model["x"]] = x_val
                    feed_dict_test[model["y"]] = y_val

                    test_error, summary_test = sess.run(
                        [model["error"], merged_summary], 
                        feed_dict_test
                    )
                    test_writer.add_summary(summary_test, tb_counter)
                    logits, y = sess.run([model["logits"], model["y"]], feed_dict_test)
                    
                    print(logits, y)
                    print("Training error: {}".format(train_error / iter_num))
                    print("Testing error: {}\n".format(test_error))
                    
                    cursor.execute(
                        "INSERT INTO {} VALUES (?, ?, ?)".format(save_dir), 
                        (train_error / iter_num, float(test_error), cost / iter_num)
                    )
                    db_connection.commit()
                    train_error, cost = 0, 0
                    
                    # if not learning_rate_change and test_error < 0.15:
                    #     print("learning_rate changed to 1e-4")
                    #     feed_dict_train[model["learning_rate"]] = 1e-4
                    #     learning_rate_change = True

                    # if test_error < test_error_min:
                    #     model["saver"].save(sess, save_dir + "_min/model.ckpt")
                    #     test_error_min = test_error

            db_connection.commit()

        # model["saver"].save(sess, save_dir + "_final/model.ckpt")


if __name__ == "__main__":
    # Load data and define global variables.
    model_pickle = "shape_model.pickle"
    with open(model_pickle, "rb") as pickle_db:
        x_train, y_train, x_val, y_val, x_test, y_test = pickle.load(pickle_db)

    iter_num = 30
    num_epochs = 2
    batch_size = 5
    num_layers = 3
    num_inputs = 12
    num_classes = 2
    activation_type = "tanh"
    dropout_keep_prob = 1
    num_steps = len(x_train[0])
    len_test = len(x_val)
    database = "shape_model.db"
    y_train = [[y[0] / 400, y[1] / 100] for y in y_train]
    y_val = [[y[0] / 400, y[1] / 100] for y in y_val]
    state_size = 128
    [x_train, y_train] = reshape_data_for_batch([x_train, y_train], batch_size)
    cell_type = "LSTM"
    save_dir = "crack_03_10_13"

    # prev_sess = "test_error_1_28_10_final"
    prev_sess = None

    main(prev_sess)
