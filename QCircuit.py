import qubit_class as q
import Gates as g
import Operations as op
import inspect
from itertools import zip_longest

"""
The purpose of this class is to abstract most of the job of making quantum circuits.
The user is supposed to merely use strings for gates and corresponding list of arguments
for gates and get back the unitary operator for that gate
"""


class Circuit(object):

    def __init__(self, state):
        """
        :param state: A string for the initial density matrix which will be used for the circuit
        """
        self.qubits = q.MultiQubit(state)
        self.bucket = {}  # contains the string of functions for each step
        self.gate_list = {}  # contains a list of functions from the Gates module
        self.get_gates()  # adds "g." to string representation of function
        self.arg_bucket = {}  # For each step we have a list of arguments. An element in a list tells us the
        # For the specific function

    def get_circuit(self):
        return self.bucket

    def apply_circuit(self):
        """
         Applies all the steps in the bucket
        :return:
        """
        for i in range(0, len(self.bucket)):
            self.apply_step(str(i))

    def add_step(self, step, gate_string, arg_list=[]):
        """
        :param step: Should be a string denoting the step in the circuit
        :param gate_string: A string of gates with gate names separated by a comma
        :param arg_list: The is a list of arguments each element contains arguments for a specific function
        i.e the first element of list contains argument for first function
        :return:
        """
        if isinstance(step, str) and isinstance(gate_string, str):
            self.arg_bucket[step] = {}
            self.bucket[step] = list(gate_string.split(','))
            if arg_list:
                for m, i in zip_longest(self.bucket[step], arg_list):
                    self.arg_bucket[step][m] = i
        else:
            return 'Please use strings for step and gate_string arguments'

    def delete_step(self, step):
        """
        :param step: Should be string deletes the step in the bucket you do not want
        :return:
        """
        del self.bucket[step]

    def get_gates(self):
        """
        :return: Returns a dictionary of  from Gates module
        """
        list = dir(g)
        for i in list:
            if i[0] != '_' and i != 'array':
                self.gate_list[i] = 'g.' + i

    def apply_step(self, step):
        """
        :param step: The step in the circuit
        :return:  Applies the operator for a particular step in the circuit to the qubit 
        density matrix
        """
        if isinstance(step, str):
            o = self.step_operator(step)
            self.qubits.operator(o)
        else:
            return 'Argument should be a string'

    def step_operator(self, step):
        """
        :param step: The step in the circuit
        :return: Returns a the operator for the particular step in the circuit
        """
        op_list = []
        if isinstance(step, str):
            for s in self.bucket[step]:
                op_list.append(self.gate_list[s])
            operators = list(map(eval, op_list))
            for i in range(len(operators)):
                if self.check_signature(operators[i]) is not []:
                    arg = self.arg_bucket[step][operators[i].__name__]
                    operators[i] = operators[i](arg)
                else:
                    operators[i] = operators[i]()
        o = op.superkron(*operators)
        return o

    def check_signature(self, func):
        """
        This function checks to see if func requires arguments
        :param func: The function we would like to examine
        :return: List of the arguments
        """
        sig = inspect.signature(func)
        arg_list = list(sig.parameters)
        return arg_list


if __name__ == '__main__':
    c = Circuit('')
    c.add_step('0', 'z,x')
    c.add_step('1', 'r_x,r_y', arg_list=[90, 75])
    cl = c.step_operator('1')

    print(cl)




