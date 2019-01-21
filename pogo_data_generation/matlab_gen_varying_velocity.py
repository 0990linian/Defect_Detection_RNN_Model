################################################################################
# This script aims to use python to call matlab function gen_varying_velocity, 
# in order to generate related pogo-inp files with different input parameters.
################################################################################
import matlab.engine
import sqlite3
from get_hist_field_pogo import get_hist_field


def main():
	start_E = 150
	end_E = 151
	start_R = 5
	end_R = 6
	database = "Nian.db"
	db_connection = create_database(database)
	matlab_engine = matlab.engine.start_matlab()
	gen_pogo_inp_youngs_modulus_list(matlab_engine, db_connection, start_E, end_E, start_R, end_R)


def create_database(database):
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
			pogo_inp_name = "struct2d_circle_E" + str(E) + "_R" + str(R)
			inclusion_speed = matlab_engine.gen_varying_velocity(
				E * 10e8, R * 10e-4, pogo_inp_name
			)
			cursor.execute("INSERT INTO inclusion_speed (table_name, speed) VALUES (?, ?)", (pogo_inp_name, inclusion_speed))
			get_hist_field(pogo_inp_name + ".pogo-inp")

	db_connection.commit()


if __name__ == "__main__":
	main()
