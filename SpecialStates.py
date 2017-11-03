import numpy as np
import qubit_class as qubit
import Operations as op
import NoisyEvolution as ne
import Gates as g
# from sympy import*
# from sympy.physics.quantum import *


def clusterstate(n, string, ar, noisy=False, kraus=None):
    """
    This creates a cluster state
    :param n: This is the number of qubits in the system
    :param string: This is the string that determines the state of the initial system. e.g
    '0000' produces 4 qubit state being all in the ground state while '01010' produces a five qubits with
    qubits being in ground and excited states alternately
    :param ar: This is a list of numbers that specifies the various control and target position
    e.g clusterstate(4, '0000', [1,3,2,4]) creates two control operations with first qubit being the control and the third qubit
    being the target and the second operation has second being the control with the fourth qubit being the target.
    :param noisy: If true decoherence is added between gate applications
    :param kraus: This will be a 3 dimensional array of kraus matrices
    :return: returns the state of the qubit after the controlled z operations. This should be a cluster state

    """

    q = qubit.MultiQubit(string)
    q.state = np.dot(g.multi_hadamard(n), np.dot(q.state, g.multi_hadamard(n)))
    if noisy is False:
        for i in range(0, len(ar), 2):
            controlgate = g.c_u(g.z(), n, ar[i], ar[i+1])
            q.state = np.dot(controlgate, np.dot(q.state, op.ctranspose(controlgate)))
    else:
        for i in range(0, len(ar), 2):
            controlgate = g.c_u(g.z(), n, ar[i], ar[i+1])
            q.q_decohere(kraus, n)
            q.state = np.dot(controlgate, np.dot(q.state, op.ctranspose(controlgate)))


    return q


def ghz_state(n, string, ar, noisy=False, kraus=None):
    """
    This creates an n qubit ghz state
    :param n:  The number of qubits in the state
    :param string: The string for the initial state of the density matrix e.g '000' produces a state where all
    the three qubits are in the ground state while '111' produces a state where all the qubits are the excited state
    :param ar: This is a list of numbers that specifies the various control and target position
    e.g ghz_state(4, '0000', [1,2,1,3,1,4]) creates two control operations with first qubit being the
    control and the second qubit being the target and the second operation has first being the control
    with the third qubit being the target third operation has the first qubit being the control and the
    fourth qubit being the target
     :param noisy: If true decoherence is added between gate applications
    :param kraus: This will be a 3 dimensional array of kraus matrices
    :return: returns the state of the qubit after the controlled x operations. This should be a ghz state.
    """

    q = qubit.MultiQubit(string)
    h_gate = op.superkron(g.h(), np.eye(pow(2, n-1)))
    q.state = np.dot(h_gate, np.dot(q.state, h_gate))
    if noisy is False:
        for i in range(0, len(ar), 2):
            controlgate = g.c_u(g.x(), n, ar[i], ar[i + 1])
            q.state = np.dot(controlgate, np.dot(q.state, op.ctranspose(controlgate)))
    else:
        for i in range(0, len(ar), 2):
            controlgate = g.c_u(g.x(), n, ar[i], ar[i + 1])
            q.q_decohere(kraus, n)
            q.state = np.dot(controlgate, np.dot(q.state, op.ctranspose(controlgate)))

    return q


def purestate(string):
    """
    Creates a simple computational basis pure state using a string variable. e.g
    '000' produces a three qubit states with all being in the zero state while
    '100' produces three qubits with the first being in the excited state and the rest in the
    ground state.
    :param string:
    :return:
    """
    try:
        q = qubit.MultiQubit(string)
        return q
    except TypeError:
        return 'Please input string'


if __name__ == "__main__":

    state = clusterstate(3, '000', [1, 2, 2, 3])
    ghzstate = ghz_state(3, '000', [1, 2, 1, 3])
    # h = Matrix([[1, 1], [1, -1]])*1/sqrt(2)
    # o = Matrix([[1, 0], [0, 0]])
    # i = Matrix([[0, 0], [0, 1]])
    # id_gate = Matrix([[1, 0], [0, 1]])
    # z = Matrix([[1, 0], [0, -1]])
    # x = Matrix([[0, 1], [1, 0]])
    # cz = TensorProduct(o, id_gate, id_gate) + TensorProduct(i, z, id_gate)
    # cx = TensorProduct(o, id_gate, id_gate) + TensorProduct(i, x, id_gate)
    # cz_23 = TensorProduct(id_gate, o, id_gate) + TensorProduct(id_gate, i, z)
    # cx_13 = TensorProduct(o, id_gate, id_gate) + TensorProduct(i, id_gate, x)
    # multi_h = TensorProduct(h, h, h)
    # h_g = TensorProduct(h, id_gate, id_gate)
    # # init_state = TensorProduct(o, o, o)
    # cluster = cz_23*(cz*(multi_h*init_state*multi_h**-1)*cz**-1)*cz_23**-1
    # ghz = cx_13*(cx * (h_g * init_state * h_g ** -1) * cx ** -1) * cx_13 ** -1
    # print(" The cluster states is: ", state.state)
    # print("The ghz state is : ", ghzstate.state)
    # pprint(ghz)



