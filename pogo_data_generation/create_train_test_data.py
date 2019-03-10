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
	x, y = [], []
	db_connection = sqlite3.connect(database)
	cursor_shape = db_connection.cursor()
	cursor_data = db_connection.cursor()

	cursor_shape.execute("SELECT * FROM crack_info")
	shape_info = cursor_shape.fetchone()
	while shape_info:
		(table_name, length, thickness, rotation) = shape_info
		print(table_name)
		cursor_data.execute("SELECT * FROM {}".format(table_name))
		x.append(cursor_data.fetchall())
		y.append([length, thickness])
		shape_info = cursor_shape.fetchone()

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
	
	with open("../shape_model/shape_model.pickle", "wb") as pickle_save:
		pickle.dump([x_train, y_train, x_val, y_val, x_test, y_test], pickle_save)


if __name__ == "__main__":
	database = "pogo_crack.db"
	create_training_testing_data(database)




