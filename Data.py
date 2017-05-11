import statistics as st
import matplotlib.pyplot as plt
from random import randint


class Data(object):

        def __init__(self):
            self.d = {}  # This is a dictionary of data sets
            self.label = {}  # This is a dictionary of labels used for data sets
            self.ylabel = ''
            self.xlabel = ''
            self.color = {}  # This is dictionary of colors for each data set
            self.legend = ''
            self.average = {}
            self.variance = {}
            self.standard_deviation = {}
            self.data_median = {}
            self.data_summary = {}

        def mean(self, key):
            """
            :param key: Calculates the mean for the data labeled by the key variable
            :return:
            """
            self.average[key] = st.mean(self.d[key])
            return st.mean(self.d[key])

        def median(self, key):
            """
            :param key: Calculates the median for the data labeled by the key variable
            :return:
            """
            self.data_median[key] = st.median(self.d[key])
            return st.median(self.d[key])

        def stddev(self, key):
            """
            :param key: Calculates the standard deviation for the data labeled by the key variable
            :return:
            """
            self.standard_deviation[key] = st.stdev(self.d[key])
            return st.stdev(self.d[key])

        def var(self, key):
            """
            :param key: Calculates the variance for the data labeled by the key variable
            :return:
            """
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
                self.color[label] = '#%06X' % randint(0, 0xFFFFFF)

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
                for label in range(0, len(args)):
                    #plt.figure()
                    plt.plot(x_axis, self.d[args[label]], self.color[args[label]], label=self.label[args[label]])
                    plt.legend()
                    plt.show()
            elif val == 1:
                for l in self.label:
                    plt.plot(x_axis, self.d[self.label[l]], self.color[self.label[l]], label=self.label[l])
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