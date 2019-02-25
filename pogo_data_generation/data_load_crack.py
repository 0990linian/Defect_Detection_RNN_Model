################################################################################
# This script aims to take the generated ultrasonic data from Pogo using crunch5 
# machine and record them into the database.
################################################################################
import os
import re
import sqlite3


def main():
	database = "pogo_crack.db"
	db_connection = sqlite3.connect(database)
	cursor = db_connection.cursor()
	
	for file in os.listdir("pogo_gen"):
		pogo_name = file[:-4]
		pogo_info = re.search(r"crack_l(?P<length>\d+)_t(?P<thick>\d+)_r(?P<rot>\d+)", pogo_name)
		cursor.execute(
			"INSERT INTO crack_info "
			"(table_name, length, thickness, rotation) VALUES (?, ?, ?, ?)", 
			(pogo_name, pogo_info.group("length"), pogo_info.group("thick"), pogo_info.group("rot"))
		)

		cursor.execute(
			"CREATE TABLE {}"
			"(receiver_1_1 real, receiver_1_2 real, receiver_1_3 real,"
			" receiver_2_1 real, receiver_2_2 real, receiver_2_3 real,"
			" receiver_3_1 real, receiver_3_2 real, receiver_3_3 real,"
			" receiver_4_1 real, receiver_4_2 real, receiver_4_3 real)"
			.format(pogo_name)
		)
		
		with open(os.path.join("pogo_gen", file), "r") as data:
			line = data.readline()
			while line:
				i = [float(j) for j in line.split()]
				cursor.execute(
					"INSERT INTO {} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
						.format(pogo_name), 
					(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11])
				)
				line = data.readline()

		db_connection.commit()


if __name__ == "__main__":
	main()