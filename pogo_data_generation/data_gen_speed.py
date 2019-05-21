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
	start_E, end_E = 50, 210
	start_R, end_R = 5, 15
	database = "pogo_circle_stand.db"
	db_connection = create_database(database)
	matlab_engine = matlab.engine.start_matlab()
	generate_pogo_data(matlab_engine, db_connection, start_E, end_E, start_R, end_R)


def create_database(database):
	"""
	Create a new database with the default inclusion_speed table.

	Args:
		database: a string input.  The name of the newly created database.

	Returns:
		db_connection: The connection to the database.
	"""
	if os.path.isfile(database):
		os.remove(database)

	db_connection = sqlite3.connect(database)
	cursor = db_connection.cursor()
	
	cursor.execute("DROP TABLE IF EXISTS inclusion_speed")
	cursor.execute(
		"CREATE TABLE inclusion_speed (table_name varchar(255), speed real)"
	)
	db_connection.commit()
	
	return db_connection


def generate_pogo_data(
		matlab_engine, 
		db_connection, 
		start_E, 
		end_E, 
		start_R, 
		end_R
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

	for E in range(start_E, end_E):
		for R in range(start_R, end_R):
			pogo_name = "struct2d_circle_E" + str(E) + "_R" + str(R)
			inclusion_speed = matlab_engine.gen_varying_velocity(
				E * 10e8, R * 10e-4, pogo_name
			)
			cursor.execute(
				"INSERT INTO inclusion_speed (table_name, speed) VALUES (?, ?)", 
				(pogo_name, inclusion_speed)
			)
			pogo_get_hist_field(pogo_name + ".pogo-inp")
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
	data_info = matlab_engine.view_hist(pogo_name)

	cursor.execute(
		"CREATE TABLE {}"
		"(time real, emitter real, receiver real)"
		.format(pogo_name)
	)

	for i in data_info:
		cursor.execute(
			"INSERT INTO {} (time, emitter, receiver) VALUES (?,?,?)"
				.format(pogo_name), 
			(i[0], i[1], i[2])
		)


if __name__ == "__main__":
	main()
