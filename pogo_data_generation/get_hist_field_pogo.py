################################################################################
# This script aims to get pogo-hist and pogo-field files from crunch machine by 
# running commands and copying files between local machine and remote machine.
################################################################################
import logging
import subprocess

# ------------------------------------------------------------------------------
# Construct logger for this script.
# ------------------------------------------------------------------------------ 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Set the mode to "writing" so that every time we re-run the script, the 
# previous log does not remain.
file_handler = logging.FileHandler("get_hist_field_pogo.log", mode="w")
formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
file_handler.setFormatter(formatter)

# StreamHandler enables the logging message to be also output on the console.
# It can also set thee output event level.
stream_handdler = logging.StreamHandler()
stream_handdler.setLevel(logging.INFO)
stream_handdler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handdler)


def get_hist_field(pogo_inp_name):
	"""
	This function aims to get history and field files run from Pogo on the 
	remote machine while hiding the relative long system output from Pogo 
	package.

	Description:
		- Derive the names of pogo-hist and pogo-field files.
		- Copy the pogo-inp file onto the remote machine.
		- Run pogoBlock and pogoSolve on the remote machine to generate history 
			and field files.  Flag "-o" is used to overwrite existing same-name 
			files if the previous operation did not clean relative generated 
			files.
		- Copy generated pogo-hist and pogo-field files from remote machine to 
			the local machine.
		- Delete the generated pogo-hist pogo-field and pogo-block file on the 
			remote machine.
	
	Notes:
		- The entire process could go wring at any stage of this function, which 
			could lead to incomplete pogo files on the local machine.  In such 
			case, we would output an error message and clean up the possible 
			copied file on the local machine.
		- For executing command on the remote machine, the flag "-T" is chosen 
			rather than "-t".  The reason for this is flag "-t" would results in 
			the local machine requests a pty from the remote machine, which 
			would generates an error message "Pseudo-terminal will not be 
			allocated because stdin is not a terminal".  Using the flag "-T" 
			would resolve this issue and let us examine if the error message 
			really matters.

	Args:
		pogo_inp_name: The name of pogo-inp file that would run on the remote 
			machine.
	
	Returns:
		True if all command calls have no error, False otherwise.
	"""
	pogo_field_name = pogo_inp_name[:-3] + "field"
	pogo_hist_name = pogo_inp_name[:-3] + "hist"
	pogo_block_name = pogo_inp_name[:-3] + "block"
	
	cp_inp = remote_command_execute(
		"scp pogo_gen/{} nl2314@crunch5.me.ic.ac.uk:~/fyp".format(pogo_inp_name),
		"copying pogo-inp file to remote machine..."
	)

	pogo_cmd = remote_command_execute(
		"ssh -T nl2314@crunch5.me.ic.ac.uk \"pogoBlock ~/fyp/{0}; pogoSolve ~/fyp/{0} -o\"".format(pogo_inp_name),
		"running pogoBlock and pogoSolve on the crunch machine..."
	)

	cp_hist = remote_command_execute(
		"scp nl2314@crunch5.me.ic.ac.uk:~/fyp/{} pogo_gen".format(pogo_hist_name),
		"copying pogo-hist file back to local machine..."
	)

	cp_field = remote_command_execute(
		"scp nl2314@crunch5.me.ic.ac.uk:~/fyp/{} pogo_gen".format(pogo_field_name),
		"copying pogo-field file back to local machine..."
	)

	rm_pogo = remote_command_execute(
		"ssh -T nl2314@crunch5.me.ic.ac.uk \"rm ~/fyp/{0}*\"".format(pogo_inp_name[:-3]),
		"removing pogo-related files on the remote machine..."
	)

	command_result = cp_inp and pogo_cmd and cp_hist and cp_field and rm_pogo	
	return command_result


def remote_command_execute(command, process_description):
	"""
	Communicate from local machine to remote machine using subprocess.Popen.

	Description:
		- Run the input command. (Output only the process description but hide 
			the command output)
		- If the command runs OK (without errors), returns True.  If there are 
			some error occurs during running the command, returns False.
	
	Note:
		We are using subprocess.Popen here rather than subprocess.call or 
		os.system to execute commands on the command line.  The reason for this 
		is the command line output from Pogo is relative long, and it's 
		unnecessary for the user to see the output. Therefore, subprocess.Popen 
		was chosen to hide them.

	Args:
		command: A string input.  The command that would run on the local 
			machine and communicate with the remote machine.
		process_description: A string input.  Shows the process overview of 
			current input command.

	Returns:
		True if the command is run successfully without errors, False otherwise.
	"""
	logger.info(process_description)

	command_popen = subprocess.Popen(
		command, 
		stdout=subprocess.PIPE, 
		stderr=subprocess.PIPE
	)
	_, command_err = command_popen.communicate()

	if not command_err:
		logger.debug("Command runs OK.")
		return True
	else:
		logger.error(
			"The process of {0} has error \"{1}\""
			.format(process_description[:-3], command_err)
		)
		return False


if __name__ == "__main__":
	for num in range(9, 22):
		pogo_inp = "struct2d_circle_E{}.pogo-inp".format(num * 10)
		get_hist_field(pogo_inp)
