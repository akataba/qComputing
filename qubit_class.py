import NoisyEvolution as ne
import numpy as np
import Operations as op
from scipy.linalg import expm
import Gates as g
from lea import *


class Qubit(object):

    def __init__(self, value):
        """
        :param state: Should be 0 or 1. If 0 qubit initialized to ground state if 1 qubit initialized to excited state
        :return: Returns the state of the qubit
        """
        if value == 0:
            self.state = np.array([[1, 0], [0, 0]])
        elif value == 1:
            self.state = np.array([[0, 0], [0, 1]])
        else:
            print("Please enter 0 or 1 only !")

    def q_evolve(self, h, dt):
        """
        :param h: Hamiltonian by which to evolve the system
        :param dt: time step to evolve by
        :return: returns the state of qubit after evolution
        """
        U = expm(-1j*h*dt)
        self.state = np.dot(U, np.dot(self.state, op.ctranspose(U)))

    def q_decohere(self, k):
        """
        :param k: A set of kraus operators for decoherent evolutioin
        :param n: This is the number of qubits
        :return: Returns the state of the qubit after application of kraus operators
        """
        self.state = ne.decohere(k, self.state, 1)

    def operator(self, o):
        """
        :param o: The operator you want applied to the qubit
        :return:  Returns the transformed density matrix after the operation
        """
        self.state = np.dot(o, np.dot(self.state, o))

    def measure(self):
        """
        :return: Density matrix after a measurement
        """
        p0 = np.trace(np.dot(self.state, g.b1()))
        p1 = np.trace(np.dot(self.state, g.b4()))
        outcome = {'0': p0*100, '1': p1*100}
        picked_obj = Lea.fromValFreqsDict(outcome)
        picked_state = picked_obj.random()

        if picked_state == '0':
            self.state = g.b1()
        else:
            self.state = g.b4()


class MultiQubit(Qubit):
    def __init__(self, string):
        """
        :param string: This should be a string of 1's and 0's where 1 is the |1> and
        0 is |0>
        :return: Returns the
        """
        tmp = 1
        for i in range(0, len(string)):
            if string[i] == '0':
                tmp = np.kron(tmp, np.array([[1, 0], [0, 0]]))
            else:
                tmp = np.kron(tmp, np.array([[0, 0], [0, 1]]))
        self.state = tmp

    def q_decohere(self, k, n):
        self.state = ne.decohere(k, self.state, n)








