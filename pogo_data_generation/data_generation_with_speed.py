################################################################################
# This script aims to use python to call matlab function gen_varying_velocity, 
# in order to generate related pogo-inp files with different input parameters.
################################################################################
import os
import sqlite3
import matlab.engine
from get_hist_field_pogo import pogo_get_hist_field


def main():
	start_E = 150
	end_E = 155
	start_R = 5
	end_R = 7
	database = "pogo_circle.db"
	db_connection = create_database(database)
	matlab_engine = matlab.engine.start_matlab()
	gen_pogo_inp_youngs_modulus_list(matlab_engine, db_connection, start_E, end_E, start_R, end_R)


def create_database(database):
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


def gen_pogo_inp_youngs_modulus_list(matlab_engine, db_connection, start_E, end_E, start_R, end_R):
	cursor = db_connection.cursor()

	for E in range(start_E, end_E):
		for R in range(start_R, end_R):
			pogo_name = "struct2d_circle_E" + str(E) + "_R" + str(R)
			inclusion_speed = matlab_engine.gen_varying_velocity(
				E * 10e8, R * 10e-4, pogo_name
			)
			cursor.execute("INSERT INTO inclusion_speed (table_name, speed) VALUES (?, ?)", (pogo_name, inclusion_speed))
			pogo_get_hist_field(pogo_name + ".pogo-inp")
			write_pogo_into_db(cursor, matlab_engine, pogo_name)

	db_connection.commit()


def write_pogo_into_db(cursor, matlab_engine, pogo_name):
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
	# Run matlab file view_hist_function.m.
	# data_info has elements representing data at every single time frame, 
	# with structure [time, emitter_magnitude, receiver_magnitude]
	data_info = matlab_engine.view_hist(pogo_name)

	cursor.execute(
		"CREATE TABLE {}"
		"(time real, emitter real, receiver real, inclusion_speed real)"
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
