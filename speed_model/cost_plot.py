import matplotlib.pyplot as plt
import sqlite3


def plot_single(database, table):
	db_connection = sqlite3.connect(database)
	cursor = db_connection.cursor()

	test_error = cursor.execute("SELECT test_error FROM {}".format(table))
	test_error = test_error.fetchall()
	iteration = [i * 30 for i in range(0, len(test_error))]

	plt.plot(iteration, test_error, color="red")
	plt.xlabel("Ierations")
	plt.ylabel("Test error / %")
	plt.show()


def plot_double(database, table_1, table_2):
	db_connection = sqlite3.connect(database)
	cursor = db_connection.cursor()

	test_error_1 = cursor.execute("SELECT test_error FROM {}".format(table_1))
	test_error_1 = test_error_1.fetchall()
	test_error_2 = cursor.execute("SELECT test_error FROM {}".format(table_2))
	test_error_2 = test_error_2.fetchall()
	iteration_1 = [i * 30 for i in range(0, len(test_error_1))]
	iteration_2 = [i * 30 for i in range(0, len(test_error_2))]

	plt.plot(iteration_1, test_error_1, label="3_layers", color="red")
	plt.plot(iteration_2, test_error_2, label="2_layers", color="blue")
	plt.xlabel("Ierations")
	plt.ylabel("Test error / %")
	plt.legend(loc=1)
	plt.show()


def min_error(database, table):
	db_connection = sqlite3.connect(database)
	cursor = db_connection.cursor()

	test_error = cursor.execute("SELECT test_error FROM {}".format(table))
	test_error = test_error.fetchall()
	print(min(test_error))


if __name__ == "__main__":
	plot_single("cx1/speed_model_3.db", "result_LSTM")
	# plot_double("speed_model_train.db", "test_error_1_26_8", "test_error_1_26_9")
	min_error("cx1/speed_model_3.db", "result_LSTM")