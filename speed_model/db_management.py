################################################################################
# This script aims to provide an easy way to change the content of the result 
# database.
################################################################################
import sqlite3


db_connection = sqlite3.connect("speed_model_train.db")
cursor = db_connection.cursor()
date = "29"

for i in range(10, 11):
	cursor.execute("DROP TABLE test_error_1_" + date + "_" + str(i))
	cursor.execute("DELETE FROM table_list WHERE table_name=\"test_error_1_{}_{}\"".format(date, i))

db_connection.commit()
