import matplotlib.pyplot as plt
import sqlite3


def plot(database, table_1, table_2):
	db_connection = sqlite3.connect(database)
	cursor = db_connection.cursor()

	test_error_1 = cursor.execute("SELECT test_error FROM {}".format(table_1))
	test_error_1 = test_error_1.fetchall()
	test_error_2 = cursor.execute("SELECT test_error FROM {}".format(table_2))
	test_error_2 = test_error_2.fetchall()
	iteration_1 = [i * 30 for i in range(0, len(test_error_1))]
	iteration_2 = [i * 30 for i in range(0, len(test_error_2))]

	plt.plot(iteration_1, test_error_1, label="state_size_20", color="red")
	plt.plot(iteration_2, test_error_2, label="state_size_64", color="blue")
	plt.xlabel("Ierations")
	plt.ylabel("Test error / %")
	plt.legend(loc=1)
	plt.show()


if __name__ == "__main__":
	plot("speed_model_train.db", "test_error_1_26_9", "test_error_1_26_13")
