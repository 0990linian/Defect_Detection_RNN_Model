################################################################################
# This script aims to use python to call matlab function gen_varying_velocity, 
# in order to generate related pogo-inp files with different input parameters.
################################################################################
import matlab.engine


def gen_pogo_inp_youngs_modulus_list(start_E, end_E):
	youngs_modulus_list, pogo_inp_list = [], []
	for num in range(start_E, end_E):
		pogo_inp_name = "struct2d_circle_E" + str(num * 10)
		pogo_inp_list.append(pogo_inp_name)
		youngs_modulus_list.append(num * 10e9)
	return pogo_inp_list, youngs_modulus_list


def generate_pogo_inp(matlab_engine, youngs_modulus_list, pogo_inp_list):
	for E, pogo_inp in zip(youngs_modulus_list, pogo_inp_list):
		matlab_engine.gen_varying_velocity(E, pogo_inp, nargout=0)


if __name__ == "__main__":
	start_E = 9
	end_E = 22
	matlab_engine = matlab.engine.start_matlab()
	pogo_inp_list, youngs_modulus_list = \
		gen_pogo_inp_youngs_modulus_list(start_E, end_E)
	generate_pogo_inp(matlab_engine, youngs_modulus_list, pogo_inp_list)
