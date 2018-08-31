import Gates as g
import Operations as op
import numpy as np
from decimal import *


def decohere(K, rho, n):
    """
    Carries out the kraus evolution of density matrix rho
    :param K: An array containing the Kraus Matrices
    :param rho: The density matrix to be evolved
    :param n: The number of qqubits
    :return: The evolved density matrix
    """

    out = np.zeros((pow(2, n), pow(2, n)), dtype=complex)
    for x in range(len(K)):
        out += np.dot(K[x], np.dot(rho, op.ctranspose(K[x])))
    return out


def kraus_pta(n, t, t1, t2):
    """
    Produces the kraus matrices for the pta channel
    :param n: number of qubit
    :param t: time step for evolution
    :param t1: relaxation time
    :param t2: dephasing time
    :return: returns a 3 dimensinal array of pta kraus matrices
    """
    gamma = 1 - np.exp(-t / t1)
    t_phi = Decimal(1 / t2) - Decimal(1 / (2 * t1))
    # lambda1 = np.exp(-t/t1)*(1-np.exp(-2*(t/t_phi)))
    px = py = gamma / 4.0
    pz = 1.0 / 2.0 - py - np.exp(-t / (2 * t1)) * np.exp(-(t / t_phi)) / 2
    pi = 1 - (px + py + pz)
    print('px: ', py, 'pz: ', pz, 'pi: ', pi)
    A = np.zeros((pow(4, n), pow(2, n), pow(2, n)), dtype=complex)  # 3 dimensional array to store kraus matrices
    ptaOperators = {
        '0': np.sqrt(pi) * g.id(),
        '1': np.sqrt(px) * g.x(),
        '2': np.sqrt(py) * g.y(),
        '3': np.sqrt(pz) * g.z()
    }

    # get labels
    labels = op.createlabel(n, 4)

    for i in range(len(labels)):
        temp = 1
        for digit in labels[i]:
            temp = np.kron(temp, ptaOperators[digit])
        A[i] = temp
    return A


def pta_ad(n, t, t1):
    """
    Produces the kraus matrices for the pta channel
    :param n: number of qubit
    :param t: time step for evolution
    :param t1: relaxation time
    :return: returns a 3 dimensinal array of pta kraus matrices
    """
    gamma = 1 - np.exp(-t / t1)
    px = py = gamma / 4.0
    pz = 1.0 / 2.0 - py - np.sqrt(1 - gamma) / 2
    pi = 1 - (px + py + pz)
    A = np.zeros((pow(4, n), pow(2, n), pow(2, n)), dtype=complex)  # 3 dimensional array to store kraus matrices
    ptaOperators = {
        '0': np.sqrt(pi) * g.id(),
        '1': np.sqrt(px) * g.x(),
        '2': np.sqrt(py) * g.y(),
        '3': np.sqrt(pz) * g.z()
    }

    # get labels
    labels = op.createlabel(n, 4)

    for i in range(len(labels)):
        temp = 1
        for digit in labels[i]:
            temp = np.kron(temp, ptaOperators[digit])
        A[i] = temp
    return A


def kraus_ad(n, t, t1):
    """
      Produces the kraus matrices for the amplitude damping channel
        :param n: number of qubit
        :param t: time step for evolution
        :param t1: relaxation time
        :return: returns a 3 dimensinal array of amplitude damping kraus matrices
      """

    A = np.zeros((pow(2, n), pow(2, n), pow(2, n)), dtype=complex)  # 3 dimensional array to store kraus matrices
    gamma = 1 - np.exp(-t / t1)
    adOperators = {
        "0": np.array([[1, 0], [0, np.sqrt(1 - gamma)]]),
        "1": np.array([[0, np.sqrt(gamma)], [0, 0]])
    }

    labels = op.createlabel(n, 2)

    for i in range(len(labels)):
        temp = 1
        for digit in labels[i]:
            temp = np.kron(temp, adOperators[digit])
        A[i] = temp
    return A


def kraus_exact(n, t, t1, t2, markovian=False, alpha=None):
    """
    Produces the kraus matrices for the exact evolution of amplitude damping with dephasing channel
        :param n: number of qubit
        :param t: time step for evolution
        :param t1: relaxation time
        :param t2: dephasing time must be smaller than t1
        :param markovian: If true the kraus matrices are for non markovian evolution and
        t2 takes the role of  t_phi
        :param alpha: The power of 1/f^{alpha} flux noise
        :return:  a 3 dimensinal array of kraus matrices with amplitude damping and dephasing

    """

    A = np.zeros((pow(3, n), pow(2, n), pow(2, n)), dtype=complex)  # 3 dimensional array to store kraus matrices
    gamma = 1 - np.exp(-t / t1)
    if markovian:
        t_phi = 1 / t2 - 1 / (2 * t1)
        lambda1 = np.exp(-t / t1) * (1 - np.exp(-2 * (t / t_phi)))
        print('We are markovian')
    else:
        t_phi = t2
        lambda1 = np.exp(-t / t1) * (1 - np.exp(-2 * (t / t_phi)**(1 + alpha)))
        print('We are non markovian')

    krausOperators = {
        "0": np.array([[1, 0], [0, np.sqrt(1 - gamma - lambda1)]]),
        "1": np.array([[0, np.sqrt(gamma)], [0, 0]]),
        "2": np.array([[0, 0], [0, np.sqrt(lambda1)]]),
    }

    labels = op.createlabel(n, 3)

    for i in range(len(labels)):
        temp = 1
        for digit in labels[i]:
            temp = np.kron(temp, krausOperators[digit])
        A[i] = temp
    return A

