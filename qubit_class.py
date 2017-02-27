import NoisyEvolution as ne
import  numpy as np
import Operations as op
from scipy.linalg import expm


class Qubit(object):

    def __init__(self,value):
        """
        :param state: Should be 0 or 1. If 0 qubit initialized to ground state if 1 qubit initialized to excited state
        :return: Returns the state of the qubit
        """
        if value == 0:
            self.state = np.array([[1,0],[0,0]])
        elif value == 1:
            self.state = np.array([[0,0],[0,1]])
        else:
            print("Please enter 0 or 1 only !")

    def q_evolve(self,h,dt):
        """
        :param h: Hamiltonian by which to evolve the system
        :param dt: time step to evolve by
        :return: returns the state of qubit after evolution
        """
        U = expm(-1j*h*dt)
        self.state = np.dot(U,np.dot(self.state,op.ctranspose(U)))

    def q_decohere(self,k):
        """
        :param k: A set of kraus operators for decoherent evolutioin
        :return: Returns the state of the qubit after application of kraus operators
        """
        self.state = ne.decohere(k,self.state,1)

    def operator(self,o):
        """
        :param o: The operator you want applied to the qubit
        :return:  Returns the transformed density matrix after the operation
        """
        self.state = np.dot(o,np.dot(self.state,o))



class Multi_Qubit(Qubit):
    def __init__(self,value,n):
        """
        :param value:Should be 0 or 1. If 0 all qubits are initialized in in ground state, if 1 all qubits
        initialized to excited state.
        :return: Returns the state of the qubit
        """
        tmp =1
        if value == 0:
            for i in range(0,n):
                tmp = np.kron(tmp,np.array([[1,0],[0,0]]))
            self.state =  tmp
        elif value == 1:
            for i in range(0,n):
                tmp = np.kron(tmp,np.array([[0,0],[0,1]]))
            self.state = tmp
        else:
            print("Please eneter 0 or 1 only !")

    def q_decohere(self,k,n):
        self.state = ne.decohere(k,self.state,n)






