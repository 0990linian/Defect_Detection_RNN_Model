################################################################################
# This script aims to provide an easy way to change the content of the result 
# database.
################################################################################
import sqlite3


db_connection = sqlite3.connect("speed_model_train.db")
cursor = db_connection.cursor()
for i in range(3, 4):
	cursor.execute("DROP TABLE test_error_1_25_0" + str(i))
	cursor.execute("DELETE FROM table_list WHERE num_epochs=3")
	cursor.execute("DELETE FROM table_list WHERE table_name=\"test_error_1_25_0{}\"".format(i))
db_connection.commit()
