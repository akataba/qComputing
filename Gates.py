from numpy import array, sqrt, kron, zeros, pi
from Operations import generatehamiltoniantring, generatetensorstring, controlgatestring
from scipy.linalg import expm
from cmath import exp


def x():
    out = array([[0, 1], [1, 0]])
    return out


def h():
    out1 = 1/sqrt(2)*array([[1, 1], [1, -1]])
    return out1


def id():
    out2 = array([[1, 0], [0, 1]])
    return out2


def y():
   out3 = array([[0, -1j], [1j, 0]])
   return out3


def z():
    out4 = array([[1, 0], [0, -1]])
    return out4


def cnot():
    out5 = array([[1, 0, 0, 0, ], [0, 1, 0, 0, ], [0, 0, 1, 0], [0, 0, 0, 1]])
    return out5

def cz():
    out6 = array([[1, 0, 0, 0, ], [0, 1, 0, 0, ], [0, 0, 1, 0], [0, 0, 0, -1]])
    return out6


def r_x(theta):
    out7 = expm(-1j*theta*x())
    return out7


def r_y(theta):
    out8 = expm(-1j*theta*y())
    return out8


def r_z(theta):
    out8 = expm(-1j*theta*z())
    return out8


def r_i(theta):
    out9 = expm(-1j*theta*id())
    return out9

def b1():
    out = array([[1, 0], [0, 0]])
    return out


def b2():
    out = array([[0, 1], [0, 0]])
    return out


def b3():
    out = array([[0, 0], [1, 0]])
    return out


def b4():
    out = array([[0, 0], [0, 1]])
    return out


def q_Hamiltonian(ham, n, s):
    """
    :param ham: hamiltonian by which the qubits will evolve by
    :param s : must be a string of ones e.g 010 represents id (tensor)ham(tensor)id while 10 represents ham(tensor) id
    :param n : The length of the string. This determines how many zeros there will be
    :return:
    """

    label = generatehamiltoniantring(n, s)
    a = zeros((pow(2, n), pow(2, n)), dtype=complex)
    terms = {
         "0": id(),
         "1": ham
     }
    for qubit in range(len(label)):
        tmp = 1
        for digit in label[qubit]:
            tmp = kron(tmp, terms[digit])
        a += tmp
    return a


def e_ij(tup, i, j):
    k = zeros(tup)
    k[i-1, j-1] = 1
    return k


def c_u(u, n, i, j):
    """
    This creates a controlled unitary operation on n qubits
    :param u: Unitary matrix
    :param n: The number of qubits to be used
    :param i: the position of control qubit for the controlled operation
    :param j: the position of target qubit for the controlled operation
    :return:  the controlled operation
    """
    tmp = 1
    tmp1 = 1
    term_1 = {
        "0": id(),
        "1": e_ij((2, 2), 1, 1)
    }
    # What happens when the control qubit is in the zero state
    label_1 = generatetensorstring(n, i)
    print(label_1)
    for qubit in range(len(label_1)):
        key = label_1[qubit]
        tmp = kron(tmp, term_1[key])
    cu_1 = tmp
    print('cu_1: ', cu_1)

    # What happens when the control bit is in the one state

    term_2 = {
        "0": id(),
        "2": u,
        "1": e_ij((2, 2), 2 , 2)
    }

    label_2 = controlgatestring(n, ('1', i), ('2', j))
    print(label_2)
    for qubit in range(len(label_2)):
        for digit in label_2[qubit]:
            tmp1 = kron(tmp1, term_2[digit])
    cu_2 = tmp1
    print('cu_2: ', cu_2)

    return cu_1 + cu_2


def qft(n):
    """
    :param n: The number of qubits
    :return:  outputs the quantum fourier transform for n qubits
    """
    w = exp(1j*2*pi/n)
    dft = zeros((n, n), dtype=complex)
    for i in range(0, n):
        for k in range(0, n):
            dft[i, k] = pow(w, i*k)
    return dft*1/sqrt(n)






