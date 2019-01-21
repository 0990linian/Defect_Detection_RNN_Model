import sqlite3


def get_data_from_dtaabase(database):
	db_connection = sqlite3.connect(database)
	cursor = db_connection.cursor()

	cursor.execute("SELECT * FROM Nian")
	a = cursor.fetchall()
	print(a)
	print(len(a))
	print(len(a[0]))


def create_new_database(database):
	db_connection = sqlite3.connect(database)
	cursor = db_connection.cursor()

	cursor.execute("CREATE TABLE Nian (thing1 real, thing2 real, speed real)")
	cursor.execute("INSERT INTO Nian (thing1, thing2, speed) VALUES (?, ?, ?)", (3.4, 4.5, 6.7))
	for i in range(20):
		cursor.execute("INSERT INTO Nian (thing1, thing2) VALUES (?, ?)", (i + 0.9, i + 20))
	db_connection.commit()


if __name__ == "__main__":
	create_new_database("Nian.db")
	get_data_from_dtaabase("Nian.db")




