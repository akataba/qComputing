from numpy import array,sqrt,kron,zeros
from scipy.linalg import expm
import Operations as op

def x():
    out = array([[0,1],[1,0]])
    return out


def h():
    out1 = 1/sqrt(2)*array([[1,1],[1,-1]])
    return out1


def id():
    out2 = array([[1,0],[0,1]])
    return out2

def y():
   out3 = array([[0,-1j],[1j,0]])
   return out3


def z():
    out4 = array([[1,0],[0,-1]])
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
    out = array([[1,0],[0,0]])
    return out
def b2():
    out =array([[0,1],[0,0]])
    return out
def b3():
    out =array([[0,0],[1,0]])
    return out
def b4():
    out = array([[0,0],[0,1]])
    return out

def q_Hamiltonian(ham,n,s):
    """
    :param ham: hamiltonian by which the qubits will evolve by
    :param s : must be a string of ones e.g 010 represents id (tensor)ham(tensor)id while 10 represents ham(tensor) id
    :return:
    """

    label =[]
    A = zeros((pow(2,n),pow(2,n)),dtype = complex)
    for i in range(0,n):
        str = s
        str = str.ljust(n-i,'0')
        str = str.rjust(n,'0')
        label.append(str)

    terms = {
         "0" : id(),
         "1" : ham
     }
    for qubit in range(len(label)):
        tmp = 1
        for digit in label[qubit]:
            tmp = kron(tmp,terms[digit])
        A += tmp

    return A