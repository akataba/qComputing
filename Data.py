import statistics as st
import matplotlib.pyplot as plt
from random import randint, uniform


class Data(object):

        def __init__(self):
            self.d = {}  # This is a dictionary of data sets
            self.label = {}  # This is a dictionary of labels used for data sets
            self.ylabel = ''
            self.xlabel = ''
            self.color = {}  # This is dictionary of colors for each data set
            self.legend = ''
            self.average = {}
            self.dataset_average = {}
            self.variance = {}
            self.standard_deviation = {}
            self.data_median = {}
            self.data_summary = {}

        def mean(self, key):
            """
            :param key: Calculates the mean for the data labeled by the key variable
            :return:
            """
            self.average[key] = []
            self.average[key] = st.mean(self.d[key])
            return st.mean(self.d[key])

        def dataset_mean(self, set_av):
            """
            :param set_av: This is the key value for the average of the data set
            :return:
            """
            temp = []
            self.dataset_average[set_av] = [0]
            for key in self.d:
                temp.append(self.d[key])
            m = list(map(st.mean, zip(*temp)))
            self.dataset_average[set_av] = m

        def median(self, key):
            """
            :param key: Calculates the median for the data labeled by the key variable
            :return:
            """
            self.data_median[key] = []
            self.data_median[key] = st.median(self.d[key])
            return st.median(self.d[key])

        def stddev(self, key):
            """
            :param key: Calculates the standard deviation for the data labeled by the key variable
            :return:
            """
            self.standard_deviation[key] = []
            self.standard_deviation[key] = st.stdev(self.d[key])
            return st.stdev(self.d[key])

        def dataset_stddev(self, set_stddev):
            """
            Calculates the standard deviation for all the data sets in the dictionary "d"
            :param set_stddev: key variable for standard deviation data
            :return:
            """
            temp = []
            self.standard_deviation[set_stddev] = [0]
            for key in self.d:
                temp.append(self.d[key])
            m = list(map(st.stdev, zip(*temp)))
            self.standard_deviation[set_stddev] = m

        def var(self, key):
            """
            :param key: Calculates the variance for the data labeled by the key variable
            :return:
            """
            self.variance[key] = []
            self.variance[key] = st.variance(self.d[key])
            return st.variance(self.d[key])

        def add(self, key, x):
            """
            :param key: label for a specific data set
            :param x: add a data value for a specific data set labeled by the key variable
            :return:
            """
            if key in self.d:
                self.d[key].append(x)
            else:
                self.d[key] = [x]

        def data(self, key):
            """
            :param key: label for the data set
            :return: Returns the data set labeled by the key variable
            """
            return self.d[key]

        def add_label(self, data_label):
            """
            :param data_label:
            :return:
            """
            if data_label in self.xlabel:
                print("This label is already in dictionary")
            else:
                self.label[data_label] = data_label

        def get_colors(self):
            """"
            Puts random colors in the color dictionary
            :return:
            """
            for label in self.d:
                self.color[label] = (uniform(0, 1), uniform(0, 1), uniform(0, 1))

        def initialize_d(self, label):
            """
            :param label: List of labels used to intialize values in the dictionary before it is used
            :return:
            """
            if isinstance(label, list):
                for i in label:
                    self.d[i] = []
            else:
                self.d[label] = []

        def graph(self, x_axis, *args, val=0):
            """
            :param x_axis: This is the data for the x axis
            :param args: This is a positional arguments of keys for the data dictionary
            :param val: For val= 0 Graphs only the data corresponding to the keys you input while when val=1
            graphs all the data in the self.d dictionary
            :return:
            """
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            if val == 0:
                for m in range(0, len(args)):
                    plt.plot(x_axis, self.d[args[m]], color=self.color[args[m]], label=self.label[args[m]])
                    plt.legend()
                    plt.show()
            elif val == 1:
                for l in self.label:
                    plt.plot(x_axis, self.d[self.label[l]], color=self.color[self.label[l]], label=self.label[l])
                    plt.legend()
                    plt.show()

        def graph_average(self, x_axis, key):
            dict_key = list(self.d.keys())
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            plt.plot(x_axis, self.dataset_average[key], self.color[dict_key[0]], label=key)
            plt.legend()
            plt.show()

        def graph_errorbars(self, x_axis, key, key_1):
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            plt.errorbar(x_axis, self.dataset_average[key],
                         yerr=self.standard_deviation[key_1], fmt='o', label=key_1)
            plt.legend()
            plt.show()

        def addlist(self, key, l):
            if isinstance(key, str):
                self.d[key] = l

        def delete(self, key=''):
            if key != '':
                self.d[key] = []
            else:
                self.d.clear()


if __name__ == "__main__":
    k = [randint(0, 20) for i in range(20)]
    k1 = [randint(0, 20) for i in range(20)]
    k2 = [randint(0, 20) for i in range(20)]
    data = Data()
    data.d['k'] = k
    data.d['k1'] = k1
    data.d['k2'] = k2
    print(len(data))

    data.dataset_mean('average')
    data.dataset_stddev('av')
    data.graph_errorbars(list(range(0, 20)), 'average', 'av')



