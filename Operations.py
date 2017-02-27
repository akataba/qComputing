from numpy import zeros, dot,savetxt,array,loadtxt
import matplotlib.pyplot as plt
import sys

sys.path.insert(1,'/home/amara/Documents/"Python Files"/PauliChannelPaper')




def checknormkraus(k,n):
    """
    Makes sure that the kraus matrices are properly normalized and therefore preserve probability
    :param k: A 3 dimensional array of shape(m,2^n,2^n). m is the number of kraus matrices
    :param n: The number of qubits
    :return: Should return identity
    """
    out = zeros((pow(2,n),pow(2,n)),dtype = complex)
    for x in range(len(k)):
        out += dot(ctranspose(k[x]), k[x])
    return out


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array looks like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

def dec2base(n,base):
    """
    Gets gives you a string in base less than 20
    :param n: The number you have
    :param base: The base you want
    :return:  The string containing giving the base m representation
    """
    convertString = "0123456789ABCDEF"
    if n < base:
        return convertString[n]
    else:
        return dec2base(n // base, base) + convertString[n % base]

def createlabel(q,n):
    """
    Create a string of labels for making kraus matrices
    :param q: Number of qubits also the length of the label string
    :param n: Number of distinct kraus matrices for one qubit
    :return: A list of labels
    """
    # When using dec2base function make sure to pad the string with the right number of zeros e.g for base 3 dec2base
    # gives 1 rather than 01 if we were dealing with 2 qubits.
    # The number of krause kraus matrices or labels is  n^q

    label =[]
    for i in range(pow(n,q)):
        label.append(dec2base(i,n))

    # Next we make sure that each element in the label list has length the number of qubits if not add a zero
    for x in range(len(label)):
        if len(label[x]) < q:
            label[x]=label[x].zfill(q)
        else:
            break
    return label

def ctranspose(A):
    """
    :param A: Matrix A
    :return: Conjugate transpose of A
    """
    out = A.conjugate().transpose()
    return out

def frange(start, end, step):
    tmp = start
    while(tmp < end):
        yield tmp
        tmp += step



def write_to_file(name,*args):
    """
    :param name: A string that is the name of the file to which the data will be written to
    :param args: list of arguments with entries being data e.g arg[0] = time where time is a list of times
    :return:
    """

    data = array(args).T
    file = open(name,'w+')
    savetxt(name,data,fmt=['%.5f','%.5f'])
    file.close()

def load_data(name,marker1,xlabel='',ylabel='',val=0):
    """
    :param name: String varible containing name of the file
    :param val: Should be 1 to load and plot data else throw error message else just loads data
    :param xlabel: xlabel for the graph
    :param ylabel: ylabel for the graph
    :return:
    """
    data_1,data_2 = loadtxt(name,dtype=float,unpack=True)

    if val == 1:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.scatter(data_1,data_2, label = marker1,color ='k')
        plt.legend()
        plt.show()

def generateDictKeys(string,n):
    """
    This was written as a quick and easy way of producing keys for a dictionary.
    :param string: Base String on which to attach numerals
    :param n: Number of strings to generate with '{}{}'.format(string,n) being the last string generated
    :return: yields strings like "a1" or "a2" if string= 'a' and n>2
    """
    if type(string) != str or  type(n) != int:
        raise ValueError('Please input string and integer for first and second argument')
    else:
       keylist=[string+str(i) for i in range(n)]
       return keylist

def generatehamiltoniantring(n, s):
    """
    This generates a string that is used to contruct tensor products terms in Hamiltonian with a certain kind of
    interaction term. For example suppose we have an ising Hamiltonian with 4 qubits. We generate strings in a
    list 1100, 0110,0011. The zero will be used to tensor an identity and the 1 will be used to tensor the pauli
    matrix.
    :param n: The number of qubits we have in our system
    :param s: This is a string of 1's that determines how local our hamiltonian is. So a three body interaction
    means we need '111' or a two body interaction will be '11'
    :return: Returns a list of strings that will be used to create the Hamiltonian
    """
    label = []
    if isinstance(s, str):
        for i in range(0, n):
            strs = s
            strs = strs.ljust(n-i, '0')
            strs = strs.rjust(n, '0')
            label.append(strs)
    else:
        print('Please enter string for second variable and integer for first')
    return label


def controlgatestring(n, control_pos1, target_pos2):
    """
    :param n: The length of the string
    :param control_pos1: This must be a tuple that specifies which qubit is control and its position
     (string,number)
    :param target_pos2:  This must be a tuple that specifies which qubit is target and its position
    :return:
    """
    out = ''

    if isinstance(control_pos1, tuple) and isinstance(target_pos2, tuple):
        control, pos1 = control_pos1
        target, pos2 = target_pos2
        for i in range(0, n):
            if i == pos1-1:
                out += control
            elif i == pos2-1:
                out += target
            else:
                out += '0'
    else:
        print("Pleas make sure that you have provided a tuple with the 1st arg being a string and 2nd being a number")

    return out


def generatetensorstring(n, *args):
    """
    This is a function like generatehamiltonianstring except it allows for the possibility that we could have
    different operators in the interaction placed at arbitrary place. Plus this does not generate a list
    but just one string. IZIIXIY needs string '0100203'. Can't handle a term like 'IZZZ' because we need
    string '0111' because for each  operator other than identity label is incremented.
    :param n: The number of qubits in the system
    :param args: list of arguments stating positions for where numbers other than 0 should go.
    :return: returns a string
    """
    out = ''
    label = 0
    arg = array(args) - 1

    for i in range(0, n):
        if i in arg:
            label += 1
            out += str(label)
        else:
            out += '0'
    return out


def partial_trace(n, m, k):
    """
    :param n: Number of qubits in the system
    :param m: The position of the qubit to trace
    :param k: label for the kth basis for the traced out qubit
    :return: Returns the matrix that helps trace out the mth qubit
    """
    tmp = 1
    tensor_label = generatetensorstring(n, m)

    terms = {"0": g.id(), "1": g.e_ij((2, 1), k, 1)}

    for i in tensor_label:
        if tensor_label[int(i)] == '1':
            tmp = kron(tmp, terms[i])
        else:
            tmp = kron(tmp, terms[i])
    return tmp




