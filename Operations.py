from numpy import zeros, dot, savetxt, array, loadtxt, kron, pi, exp, conjugate
from numpy.linalg import eig
import matplotlib.pyplot as plt
import SpecialStates as ss
import qutip as qt
import Gates as g
from numpy import trace, cos, sin, arcsin, random, sqrt
from quantum_CSD_compiler.DiagUnitaryExpander import *
from quantum_CSD_compiler.MultiplexorExpander import *
from quantum_CSD_compiler.Tree import *
from for_IBM_devices.Qubiter_to_IBMqasm2_5q import *
import os as opsys


def checknormkraus(k, n):
    """
    Makes sure that the kraus matrices are properly normalized and therefore preserve probability
    :param k: A 3 dimensional array of shape(m,2^n,2^n). m is the number of kraus matrices
    :param n: The number of qubits
    :return: Should return identity
    """
    out = zeros((pow(2, n), pow(2, n)), dtype=complex)
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


def dec2base(n, base):
    """
    Gets gives you a string in base less than 20
    :param n: The number you have
    :param base: The base you want
    :return:  The string containing giving the base m representation
    """
    convertstring = "0123456789ABCDEF"
    if n < base:
        return convertstring[n]
    else:
        return dec2base(n // base, base) + convertstring[n % base]


def createlabel(q, n):
    """
    Create a string of labels for making kraus matrices
    :param q: Number of qubits also the length of the label string
    :param n: Number of distinct kraus matrices for one qubit
    :return: A list of labels
    """
    # When using dec2base function make sure to pad the string with the right number of zeros e.g for base 3 dec2base
    # gives 1 rather than 01 if we were dealing with 2 qubits.
    # The number of kraus matrices or labels is  n^q

    label = []
    for i in range(pow(n, q)):
        label.append(dec2base(i, n))

    # Next we make sure that each element in the label list has length the number of qubits if not add a zero
    for x in range(len(label)):
        if len(label[x]) < q:
            label[x] = label[x].zfill(q)
        else:
            break
    return label


def padded_dec2base(n, q, base):
    """
       Gets gives you a string in base less than 20
       :param n: The number you have
       :param q: The number of qubits in system or determines the number of zeros to pad i.e
       if n=3 and q=3 the function will return 011 rather than 11
       :param base: The base you want
       :return:  The string containing giving the base m representation
       """
    convertstring = "0123456789ABCDEF"
    if n < base:
        return convertstring[n].zfill(q)
    else:
        return (dec2base(n // base, base) + convertstring[n % base]).zfill(q)


def ctranspose(A):
    """
    :param A: Matrix A
    :return: Conjugate transpose of A
    """
    out = A.conjugate().transpose()
    return out


def frange(start, end, step):
    tmp = start
    while tmp < end:
        yield tmp
        tmp += step


def write_to_file(name, *args):
    """
    :param name: A string that is the name of the file to which the data will be written to
    :param args: list of arguments with entries being data e.g arg[0] = time where time is a list of times
    :return:
    """

    data = array(args).T
    file = open(name, 'w+')
    savetxt(name, data, fmt=['%.5f', '%.5f'])
    file.close()


def load_data(name, marker1, xlabel='', ylabel='', val=0):
    """
    This function has two uses, it either loads the data and plots it if val=1 or it just gets the data from
     the data file and puts it into arrays which it returns
    :param name: String varible containing name of the file
    :param marker1: This sets the marker used to create points on scatter plot
    :param val: Should be 1 to load and plot data else throw error message else just loads data
    :param xlabel: xlabel for the graph
    :param ylabel: ylabel for the graph
    :return:
    """
    data_1, data_2 = loadtxt(name, dtype=float, unpack=True)

    if val == 1:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.scatter(data_1,data_2, label = marker1,color ='k')
        plt.legend()
        plt.show()
    else:
        return data_1, data_2


def generateDictKeys(string, n,step=1):
    """
    This was written as a quick and easy way of producing keys for a dictionary.
    :param string: Base String on which to attach numerals
    :param n: Number of strings to generate with '{}{}'.format(string,n) being the last string generated
    :return: yields strings like "a1" or "a2" if string= 'a' and n>2
    """
    if type(string) != str or type(n) != int:
        raise ValueError('Please input string and integer for first and second argument')
    elif step == 1:
        keylist = [string+str(i) for i in range(n)]
        return keylist
    else:
        keylist = [string+str(i) for i in range(0, n*step, step)]
        return keylist


def generatehamiltoniantring(n, s, onestring=None, pos=None, pad=None):
    """
    This generates a string that is used to construct tensor products terms in Hamiltonian with a certain kind of
    interaction term. For example suppose we have an ising Hamiltonian with 4 qubits. We generate strings in a
    list 1100, 0110,0011. The zero will be used to tensor an identity and the 1 will be used to tensor the pauli
    matrix.
    :param n: The number of qubits we have in our system
    :param s: This is a string of 1's that determines how local our hamiltonian is. So a three body interaction
    means we need '111' or a two body interaction will be '11'
    :param onestring: If you would rather generate one string than a list of strings
    :param pos: The position where the string s should start  (only if onestring is not none)
    :param pad: Rather than padding with zeros you could pad with some other number (only if onestring
    is not none)
    :return: Returns a list of strings that will be used to create the Hamiltonian
    """
    label = []
    if onestring is None:
        if isinstance(s, str):
            for i in range(0, n):
                strs = s
                strs = strs.ljust(n-i, '0')
                strs = strs.rjust(n, '0')
                label.append(strs)
        else:
            print('Please enter string for second variable and integer for first')
        return label
    else:
        strs = s
        strs = strs.ljust(n - pos, pad)
        strs = strs.rjust(n, pad)
        return strs


def controlgatestring(n, control_pos1, target_pos2, additionaloptstring = ''):
    """
    :param n: The length of the string
    :param control_pos1: This must be a tuple that specifies which qubit is control and its position
     (string,number)
    :param target_pos2:  This must be a tuple that specifies which qubit is target and its position
    :param additionaloptstring: Gives the user freedom to add string unique string after second position
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
            elif additionaloptstring != '' and i > pos2-1:
                out += additionaloptstring
            else:
                out += '0'
    else:
        print("Please make sure that you have provided a tuple with the 1st arg being a string and 2nd being a number")

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
    :param k: label for the kth basis for the traced out qubit, label begins from 1
    :return: Returns the matrix that helps trace out the mth qubit
    """
    out = 1
    tensor_label = generatetensorstring(n, m)
    terms = {"0": g.id(), "1": g.e_ij((2, 1), k, 1)}

    out = superkron(terms, string=tensor_label, val=1)
    return out


def trace_qubits(n, rho, qubit_list):
    """
    :param n: Number of qubits in the system
    :param rho: The system from which trace out qubits
    :param qubit_list: List of qubits to trace out. Labels should begin from 1
    :return: Returns a reduced density matrix
    """
    basis_labels = [1, 2]
    ops ={}
    red_dim = n - len(qubit_list)
    reduced_matrix = zeros((pow(2, red_dim), pow(2, red_dim)))

    # Initialize the tracing operator dictionary
    for q in qubit_list:
        ops[q] = []

    # Put tracing out operators in dictionary
    for q in qubit_list:
        for b in basis_labels:
            ops[q].append(partial_trace(n, q, b))

    # Perform the tracing out operation
    for q in ops:
        for b in range(0, len(ops[q])):
            reduced_matrix += dot(ops[q][b].T, dot(rho, ops[q][b]))
    return reduced_matrix


def subblock(u, p1, p2):
    """
    :param u: In put matrix
    :param p1: this is a tuple that determines the top left element of sub-block
    :param p2: this is a tuple that determines the bottom right element of sub-block
    :return: Returns the sub-block
    """
    if isinstance(p1, tuple) and isinstance(p2, tuple):
        r, c = p1
        r1, c1 = p2
        out = u[r:r1, c:c1]
    else:
        print("Please enter tuple for second and third argument")

    return out


def generateUnitary():
    """
    :return: Returns a random 2 by 2  unitary matrix
    """
    u = zeros((2,2), dtype=complex)
    zeta = random.random()  # result after calculating sine and cosine
    theta = arcsin(zeta)
    phi = random.uniform(0, 2*pi)  # angle for first phase
    chi = random.uniform(0, 2*pi)  # angle for second phase
    rho = random.uniform(0, 2*pi)

    a = exp(1j*phi)*cos(theta)
    b = exp(1j*chi)*sin(theta)
    u[0, 0] = a
    u[0, 1] = b
    u[1, 0] = -conjugate(b)
    u[1, 1] = conjugate(a)

    return dot(exp(1j*rho), u)


def gram_schmidt(A):
    """
    :param A: Matrix that does not have orthornormal column
    :param row_vecs:
    :param norm:
    :return:
    """
    for i in range(0, len(A)):
        for j in range(0, i):
            A[:, i] = A[:, i] - (dot(A[:, i].T, A[:, j])) / (dot(A[:, j].T, A[:, j]))*A[:, j]
        A[:, i] = A[:, i] / sqrt(dot(A[:, i].T, A[:, i]))

    return A


def superkron(*args, val=0,  string=''):
    """
    This generalization of kron can be used in 2 ways. It can straight forwardly take the tensor product of
    operators given the arguments i.e superkron(I,Z,X) will return the tensor product of I, Z and X.
    It can also do something more general. It can accept a dictionary of operators and a string variable
    that specifies in which order the operators in the dictionary should be tensored. E.G
    operdict = {'0': I, '1': X} and  string = '010. Then superkron(operdict, val=1,string,)
    produces tensor product superkron(I,X,I)

    :param args: List of operators to calculate tensor product of
    :param val: A val=0 makes function calculate tensor product of operators given, val=1 adds extra
    bells and whistles described in doc string above
    :param string: The order in which operators given in the operdict dictionary will be tensored
    :return: The tensor product of operators
    """

    out = 1
    if val == 0:
        for i in range(len(args)):
            out = kron(out, args[i])
    else:
        for digit in string:
            out = kron(out, args[0][digit])
    return out


def computational_basis(q, n):
    """
    :param n: type of qudit
    :param q: Number of qubits
    :return:
    """
    zero = np.array([1, 0])
    one = np.array([0, 1])
    basis_labels = createlabel(q, n)
    operators = {'0': zero, '1': one}
    basis_list = [np.array([superkron(operators, val=1, string=i)])for i in basis_labels]

    return basis_list


def realify(vec):
    """
    Gets a complex vector in C^d and produces a real vector in R^2d
    :param vec: This should be a row vector
    :return:
    """
    realified = []
    a = vec.tolist()
    for i in a[0]:
        real_part = i.real
        realified.append(real_part)
    for i in a[0]:
        imag_part = i.imag
        realified.append(imag_part)
    realified = np.array(realified)
    return realified


def construct_preunitary(arr, array_list):
    """
    Useful when constructing unitaries to make quantum states. It needs to be
    input into gram schmidt orthogonalization method
    :param arr:
    :param array_list:
    :return:
    """
    s = array_list[1].shape
    preunitary = np.array([])
    preunitary = np.append(preunitary, arr)
    preunitary = preunitary.reshape(1, s[1])
    for i in range(1, len(array_list)):
        preunitary = np.hstack((preunitary, array_list[i]))
    preunitary = preunitary.reshape(s[1], s[1]).T
    return preunitary


def makeQobj(*args):
    """
    :param args: This is a list of matrices than must be turned into Qobj in order that they can be used in the
    qutip package
    :return: returns list of Qobj in the order in which they were created. E.g makeQobj(A,B) returns a list
    l = [Qobj(A),Qobj(B)]
    """
    try:
        if isinstance(*args, list):
            l = [qt.Qobj(args[0][i]) for i in range(len(args[0]))]
        elif isinstance(*args, dict):
            l = {name: qt.Qobj(args[0][name])for name in args[0]}
        else:
            l = qt.Qobj(args[0])
        return l
    except TypeError:
        print("The input must be a list or dictionary of operators or a single operator")


def direct_sum(matrices):
    """
    :param matrices: List of matrices
    :return: The direct sum of the matrices
    """
    temp = []
    for m in matrices:
        temp.append(m.shape)
    M = zeros(tuple(map(sum, zip(*temp))), dtype=complex)
    M[:matrices[0].shape[0], :matrices[0].shape[1]] = matrices[0]
    for l in range(0, len(matrices), 2):
        if l != len(matrices)-1:
            M[matrices[l].shape[0]:matrices[l].shape[0] + matrices[l+1].shape[0],
            matrices[l].shape[1]:matrices[l].shape[1] + matrices[l+1].shape[1]] = matrices[l+1]
        else:
            M[M.shape[0]-matrices[l].shape[0]:, M.shape[1]-matrices[l].shape[1]:] = matrices[l]
    return M


def fidelity(rho, rho_1, error=False):
    """
    :param rho: First density matrix
    :param rho_1: Second density matrix
    :param error: If true returns 1 - fidelity otherwise simply returns fidelity
    :return:  Return the fidelity of two matrices in variable out
    """
    out = 0
    if error is False:
        out = trace(dot(rho, rho_1)).real
    else:
        out = 1 - trace(dot(rho, rho_1)).real

    return out


def gatecompiler(file_prefix, unitary, num_bits):
    """
    :param file_prefix:
    :param unitary
    :param num_bits:
    :return:
    """

    emb = CktEmbedder(num_bits, num_bits)
    t = Tree(True, file_prefix, emb, unitary, verbose=False)
    t.close_files()
    style = 'exact'
    MultiplexorExpander(file_prefix, num_bits, style)
    DiagUnitaryExpander(file_prefix + '_X1', num_bits, style)
    SEO_reader(file_prefix + '_X2', num_bits)


def eigen(A):
    """
    :param A:
    :return:
    """
    v, w = eig(A)
    shape_size = w.shape
    eigenvector_list = [array([w[:, i]]) for i in range(0, shape_size[1])]
    return v, eigenvector_list


def read_file(file, line_count=1, readline=None):
    """
    :param file: file name
    :param line_count: The number of lines in the file
    Note first line is indexed as 0
    :param readline: The line to read
    line by line
    :return:
    """

    with open(file, 'r') as f:
        if readline is None:
            line = f.read()
            return line  # Possible error, may not want to use return statement
        else:
            for i, lines in enumerate(f):
                if i == readline-1:
                    return lines


def file_linecount(file):
    """
    :param file: file name
    :return: Number of lines in a file
    """
    linecount = sum(1 for line in open(file))
    return linecount


def remove_line(file, string=[]):
    """
    Removes specific words from files and returns the file with same name but bad lines
    are removed
    :param file:
    :param string:
    :return:
    """
    with open(file, 'r') as f, open('tmp.txt', '+a') as new_f:
        for line in f:
            clean = True
            for word in string:
                if word in line:
                    clean = False
                if clean is True:
                    new_f.write(line)
    opsys.remove(file)
    opsys.rename('tmp.txt', file)


def unitary_to_qasm(file_prefix1, u, n, unwanted_words):
    """
    :param file_prefix1:
    :param u:
    :param n:
    :param unwanted_words: remove lines which have unwanted words
    :return: name of file with qasm code
    """
    gatecompiler(file_prefix1, u, n)
    remove_line(file_prefix1 + '_X2_' + str(n) + '_eng.txt', unwanted_words)
    Qubiter_to_IBMqasm2_5q(file_prefix1 + '_X2', n, write_qubiter_files=True)
    return file_prefix1 + '_X2_' + 'qasm2.txt'


if __name__ == '__main__':
    k = [g.z(), g.y(), g.x()]
    m = {'b1': g.b1(), 'b4': g.b4(), 'x': g.x()}
    ghz = ss.ghz_state(2, '00', [1, 2])
    cluster = ss.clusterstate(2, '00', [1, 2])
    u = generateUnitary()
    unitary = superkron(u, u)
    e, v = eigen(g.x())

    # print('list of matrices: ', k)
    # print('Direct sum of matrices: ', direct_sum(k))
    # print('string:', generatehamiltoniantring(5, '1', onestring=True, pos=2, pad= '3'))
    # print('list of string :', generatehamiltoniantring(4, '1'))
    # print('trace: ', fidelity(g.z(), g.y()))
    # print('trace operator for first bath basis: ', partial_trace(2, 2, 1))
    # print('tracing out operators: ', trace_qubits(2, g.z(), [2]))
    # print('reduced density matrix of bell state: ', trace_qubits(2, ghz.state, [2]))

    # gatecompiler('./Gate Compiler Files/random_example', unitary, 2)
    # print('binary number 3, ', dec2base(3, 2))
    # print('binary number 1, ', padded_dec2base(1, 2, 2))
    # print('eigenvalues of sigma x: ', e)
    # print('eigenvectors of sigma x:', v)
    l = read_file('/home/amara/Python Files/qComputing/Gate Compiler Files/random_example_X2_2_eng.txt', readline=None)
    print(l)
