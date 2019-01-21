################################################################################
# This script aims to use python to call matlab function view_hist, in order to 
# get time trace of Pogo-generated data and log them into a database.
################################################################################
import matlab.engine
import sqlite3


def load_pogo_into_database(database, matlab_engine, pogo_file_list):
	"""
	Load emitter and receiver data generated from Pogo into a database.

	Description:
		- For each Pogo history file generated, using matlab function view_hist 
			to derive the time-magnitude data at emitter and receiver.
		- Create a table in the database for every Pogo file generated, directly 
			using their names.
		- Log the data into corresponding tables.

	Args:
		database: A string input.  The name of the database.
		matlab_engine: A string input.  The engine used to run matlab functions.
		pogo_file_list: A list input.  The list contains all generated Pogo 
			files.

	Returns:
		None
	"""
	db_connection = sqlite3.connect(database)
	cursor = db_connection.cursor()

	for pogo_file in pogo_file_list:
		# Run matlab file view_hist_function.m.
		# data_info has elements representing data at every single time frame, 
		# with structure [time, emitter_magnitude, receiver_magnitude]
		data_info = matlab_engine.view_hist(pogo_file)

		cursor.execute(
			"CREATE TABLE IF NOT EXISTS {}"
			"(time real, emitter real, receiver real, inclusion_speed real)".format(pogo_file)
		)

		for i in data_info:
			cursor.execute(
				"INSERT INTO {} (time, emitter, receiver) VALUES (?,?,?)"
					.format(pogo_file), 
				(i[0], i[1], i[2])
			)

		db_connection.commit()


if __name__ == "__main__":
	database = "circle_database.db"
	matlab_engine = matlab.engine.start_matlab()
	pogo_file_list = []
	for num in range(9, 22):
		pogo_inp = "struct2d_circle_E{}".format(num * 10)
		pogo_file_list.append(pogo_inp)
	load_pogo_into_database(database, matlab_engine, pogo_file_list)

