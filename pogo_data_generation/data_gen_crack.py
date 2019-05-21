################################################################################
# This script aims to generate data (emitter, receiver and timeline) and label 
# (inclusion_speed) and store them into a database.
################################################################################
import os
import sqlite3
import matlab.engine
from get_hist_field_pogo import pogo_get_hist_field


def main():
	"""
	Main function.

	Description:
		- Define the range of Young's modulus and circle radius.
		- Define name of the database.
		- Generate data and label.
	"""
	start_len, end_len = 122, 300
	start_thick, end_thick = 10, 80
	start_rot, end_rot = 0, 1
	database = "pogo_crack.db"
	db_connection = create_database(database, False)
	matlab_engine = matlab.engine.start_matlab()
	generate_pogo_data(
		matlab_engine,
		db_connection,
		start_len,
		end_len,
		start_thick,
		end_thick,
		start_rot,
		end_rot
	)


def create_database(database, start_new_db=True):
	"""
	Create a new database with the default inclusion_speed table.

	Args:
		database: a string input.  The name of the newly created database.

	Returns:
		db_connection: The connection to the database.
	"""
	if start_new_db:
		if os.path.isfile(database):
			os.remove(database)

		db_connection = sqlite3.connect(database)
		cursor = db_connection.cursor()
		
		cursor.execute(
			"CREATE TABLE crack_info" 
			"(table_name varchar(255), length real, thickness real, rotation int)"
		)
		db_connection.commit()
	else:
		db_connection = sqlite3.connect(database)

	return db_connection


def generate_pogo_data(
		matlab_engine, 
		db_connection, 
		start_len, 
		end_len, 
		start_thick, 
		end_thick, 
		start_rot, 
		end_rot
	):
	"""
	Generate ultrasonic data using Pogo

	Description:
		- Run matlab function "gen_varying_velocity" with Young's modulus and 
			radius inputs to generate pogo-inp files and corresponding 
			inclusion_speed.
		- Insert inclusion_speed into database, using pogo-inp name as index.
		- Run Pogo remotely to get pogo-hist files.
		- Extract ultrasonic data from pogo-hist files and write them into 
			database.
		- The input Young's modulus and radius are scaled, this is done for 
			easier range running.

	Args:
		matlab_engine: The engine of matlab, used to call matlab functions.
		db_connection: The connection to the database.
		start_E: The scaled low-end Young's modulus.
		end_E: The scaled high-end Young's modulus.
		start_R: The scaled low-end circle radius.
		end_R: The scaled high-end circle radius.

	Returns:
		None
	"""
	cursor = db_connection.cursor()

	for length in range(start_len, end_len):
		for thick in range(start_thick, end_thick):
			for rot in range(start_rot, end_rot):
				pogo_name = "crack_l" + str(length) + "_t" + str(thick) + "_r" + str(rot)
				matlab_engine.gen_crack_multi(
					length * 10e-5, thick * 10e-5, float(rot), pogo_name, nargout=0
				)
				cursor.execute(
					"INSERT INTO crack_info "
					"(table_name, length, thickness, rotation) VALUES (?, ?, ?, ?)", 
					(pogo_name, length, thick, rot)
				)
				pogo_get_hist_field(pogo_name)
				write_pogo_into_db(cursor, matlab_engine, pogo_name)
				pogo_inp = "pogo_gen/" + pogo_name + ".pogo-inp"
				if os.path.isfile(pogo_inp):
					os.remove(pogo_inp)
		
			db_connection.commit()


def write_pogo_into_db(cursor, matlab_engine, pogo_name):
	"""
	Load emitter and receiver data generated from Pogo into a database.

	Description:
		- For each Pogo history file generated, using matlab function view_hist 
			to derive the ultrasonic data at emitter and receiver.
		- Create a table in the database for every Pogo file generated, directly 
			using their names.
		- Log the data into corresponding tables.

	Args:
		cursor: The database cursor, used to execute sql commands.
		matlab_engine: The engine used to run matlab functions.
		pogo_name: A string input. The name of pogo-related files.

	Returns:
		None
	"""
	# data_info has elements representing data at every single time frame, 
	# with structure [time, emitter_magnitude, receiver_magnitude]
	cursor.execute(
		"CREATE TABLE {}"
		"(receiver_1_1 real, receiver_1_2 real, receiver_1_3 real,"
		" receiver_2_1 real, receiver_2_2 real, receiver_2_3 real,"
		" receiver_3_1 real, receiver_3_2 real, receiver_3_3 real,"
		" receiver_4_1 real, receiver_4_2 real, receiver_4_3 real)"
		.format(pogo_name)
	)

	data_info = matlab_engine.view_hist(pogo_name)
	for i in data_info:
		cursor.execute(
			"INSERT INTO {} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
				.format(pogo_name), 
			(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11])
		)


if __name__ == "__main__":
	main()
