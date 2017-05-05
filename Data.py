import statistics as st
import matplotlib.pyplot as plt


class Data(object):

        def __init__(self):
            self.d = []
            self.label = ''
            self.ylabel = ''
            self.color = ''
            self.legend = ''
            self.average = 0
            self.variance = 0
            self.standard_deviation = 0
            self.data_median = 0
            self.data_summary = {}

        @property
        def mean(self):
            self.average = st.mean(self.d)
            return st.mean(self.d)

        @property
        def median(self):
            self.data_median = st.median(self.d)
            return st.median(self.d)

        @property
        def stddev(self):
            self.standard_deviation = st.stdev(self.d)
            return st.stdev(self.d)

        @property
        def var(self):
            self.variance = st.variance(self.d)
            return st.variance(self.d)

        def color(self, c):
            if isinstance(c, str):
                self.color = c

        def label(self, x):
            if isinstance(x, str):
                self.label = x

        def add(self, x):
            self.d.append(x)

        @property
        def data(self):
            return self.d

        def graph(self, data_2, xlabel=''):
            plt.xlabel(xlabel)
            plt.ylabel(self.ylabel)
            plt.plot(data_2, self.d, self.color, label=self.label)
            plt.show()

        def addlist(self, l):
            self.d.extend(l)

