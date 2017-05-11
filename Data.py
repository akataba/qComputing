import statistics as st
import matplotlib.pyplot as plt
from random import randint


class Data(object):

        def __init__(self):
            self.d = {}
            self.label = {}
            self.ylabel = ''
            self.xlabel = ''
            self.color = {}
            self.legend = ''
            self.average = {}
            self.variance = {}
            self.standard_deviation = {}
            self.data_median = {}
            self.data_summary = {}

        def mean(self, key):
            self.average[key] = st.mean(self.d[key])
            return st.mean(self.d[key])

        def median(self, key):
            self.data_median[key] = st.median(self.d[key])
            return st.median(self.d[key])

        def stddev(self, key):
            self.standard_deviation[key] = st.stdev(self.d[key])
            return st.stdev(self.d[key])

        def var(self, key):
            self.variance[key] = st.variance(self.d[key])
            return st.variance(self.d[key])

        def add(self, key, x):
            if key in self.d:
                self.d[key].append(x)
            else:
                self.d[key] = [x]

        def data(self, key):
            return self.d[key]

        def add_label(self, data_key, data_label):
            self.label[data_key] = data_label

        def get_colors(self):
            for label in self.d:
                self.color[label] = '#%06X' % randint(0, 0xFFFFFF)

        def graph(self, x_axis, *args, val=0):
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