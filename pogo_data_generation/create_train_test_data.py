################################################################################
# This script aims to generate training and testing data from the database.
################################################################################
import pickle
import sqlite3
from sklearn.model_selection import train_test_split


def create_training_testing_data(database):
	"""
	Create training and testing data and save them to pickle.

	Description:
		- Define two cursors of the database, one moves in table 
			inclusion_speed, one gathers data from other tables.
		- Use the table name recorded in the inclusion_speed to find 
			corresponding data.
		- Record the data and label in two lists, make sure they are in 
			corresponding positions.
		- Shuffle and create the training and testing data, save them to pickle.

	Args:
		database: A string input.  The name of the database.

	Returns:
		None.
	"""
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
	
	with open("speed_model_stand.pickle", "wb") as pickle_save:
		pickle.dump([X_train, X_test, y_train, y_test], pickle_save)


if __name__ == "__main__":
	database = "pogo_circle_stand.db"
	create_training_testing_data(database)




