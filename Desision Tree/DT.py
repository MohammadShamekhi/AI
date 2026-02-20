import numpy as np
import pandas as pd
import math
from treelib import Node, Tree

class Node:
    def __init__(self, examples, parent, attributes, full_data):
        self.examples = examples
        self.parent = parent
        self.attributes = attributes
        self.prediction = None
        self.children = {}
        self.best_attribute = None # attribute with highest IG
        self.full_data = full_data
    
    @staticmethod
    def entropy(data_set, y): # y is a random variable
        filter = data_set[y] == 1
        if len(data_set[filter]) == 0 or len(data_set[filter]) == len(data_set):
            return 0
        p = data_set[y].value_counts()[0] / len(data_set)
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    
    @staticmethod
    def information_gain(data_set, y, x):
        entropy_y = Node.entropy(data_set, y)
        domain = data_set[x].value_counts().index.to_list()
        remainder = 0
        for d in domain:
            filter = data_set[x] == d 
            a = data_set[x].value_counts()[d] / len(data_set)
            remainder += a * Node.entropy(data_set[filter], y)
        IG = entropy_y - remainder
        return IG
    
    @staticmethod
    def gini_index(data_set, x):
        domain = data_set[x].value_counts().index.to_list()
        GI = 1
        for d in domain:
            p = data_set[x].value_counts()[d] / len(data_set)
            GI -= p * p
        return GI
    
    @staticmethod
    def plurality_value(data_set, y):
        filter = data_set[y] == 1
        if len(data_set[filter]) == 0 or len(data_set[filter]) == len(data_set):
            if len(data_set[filter]) == 0:
                return 0
            else:
                return 1
        else:
            return data_set[y].value_counts()[1] / len(data_set)

    def generate_decision_tree(self, y, tree):
        tag = ""
        identifier = self
        par = self.parent
        data_set = self.examples
        filter_y = data_set[y] == 1
        if len(self.examples) == 0:
            p = Node.plurality_value(self.parent.examples, y)
            if(p > 0.5):
                self.prediction = 1
            else:
                self.prediction = 0
            tag = "0=0+0\npredict={}".format(self.prediction)
            tree.create_node(tag, identifier, parent=par)
            return
        elif len(self.attributes) == 0:
            p = Node.plurality_value(self.examples, y)
            if(p > 0.5):
                self.prediction = 1
            else:
                self.prediction = 0
            tag = "{}={}+{}\npredict={}".format(len(data_set), len(data_set[filter_y]), len(data_set[data_set[y] == 0]), self.prediction)
            tree.create_node(tag, identifier, parent=par)
            return
        elif len(data_set[filter_y]) == 0 or len(data_set[filter_y]) == len(data_set):
            self.prediction = Node.plurality_value(self.examples, y)
            tag = "{}={}+{}\npredict={}".format(len(data_set), len(data_set[filter_y]), len(data_set[data_set[y] == 0]), self.prediction)
            tree.create_node(tag, identifier, parent=par)
            return
        else:
            #test with IG
            max = 0
            for attribute in self.attributes:
                IG = Node.information_gain(data_set, y, attribute)
                if IG >= max:
                    self.best_attribute = attribute
                    max = IG
            #test with GI
            #min = math.inf
            #for attribute in self.attributes:
            #    GI = Node.gini_index(data_set, attribute)
            #    if GI <= min:
            #        self.best_attribute = attribute
            #        min = GI
            domain = self.full_data[self.best_attribute].value_counts().index.to_list()
            e = Node.entropy(data_set, y) # entropy
            tag = "{}={}+{}\nentropy={}\nattr={}\n{}".format(len(data_set), len(data_set[filter_y]), len(data_set[data_set[y] == 0]), format(e,".2f"), self.best_attribute, ",".join(map(str, domain[::-1])))
            tree.create_node(tag, identifier, parent=par)
            for d in domain:
                child = Node(data_set[data_set[self.best_attribute] == d], self, self.attributes - {self.best_attribute}, self.full_data)
                self.children[d] = child
                child.generate_decision_tree(y, tree)
            return

    def test(self, data_test, y):
        correct_test = 0
        if self.prediction == None:
            for d in self.children:
                filter = data_test[self.best_attribute] == d
                correct_test += self.children[d].test(data_test[filter], y)
        else:
            filter = data_test[y] == self.prediction
            correct_test = len(data_test[filter])
        return correct_test