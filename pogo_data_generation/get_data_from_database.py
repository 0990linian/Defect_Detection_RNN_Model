import pickle
import sqlite3
from sklearn.model_selection import train_test_split


def create_training_testing_data(database):
	X, y = [], []
	db_connection = sqlite3.connect(database)
	cursor_speed = db_connection.cursor()
	cursor_data = db_connection.cursor()

	cursor_speed.execute("SELECT * FROM inclusion_speed")
	speed_info = cursor_speed.fetchone()
	while speed_info:
		(current_table, speed) = speed_info
		cursor_data.execute("SELECT * FROM {}".format(current_table))
		X.append(cursor_data.fetchall())
		y.append(speed)
		speed_info = cursor_speed.fetchone()

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	
	with open("speed_model.pickle", "wb") as pickle_save:
		pickle.dump([X_train, X_test, y_train, y_test], pickle_save)


if __name__ == "__main__":
	database = "pogo_circle.db"
	create_training_testing_data(database)




