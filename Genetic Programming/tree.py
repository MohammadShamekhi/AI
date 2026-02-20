import random
from sklearn.metrics import mean_squared_error
import math

def operator():
    op = ['+', '-', '*', '/', '^', 'sin', 'cos']
    return(random.choice(op))

def leaf_one_dimension():
    number = random.randint(1, 9)
    variable = 'x1'
    p = random.random()
    if p <= 0.5:
        return number
    else:
        return 'x1'

def leaf_two_dimension():
    number = random.randint(1, 9)
    variable = ['x1', 'x2']
    x = random.random()
    if x <= 0.5:
        return number
    else:
        return random.choice(variable)
    
def single_operation(operation):
    return operation in ['sin', 'cos']

class Node:
    def __init__(self, value, parent, children, is_leaf):
        self.parent = parent
        self.children = children
        self.value = value
        self.is_leaf = is_leaf

class Tree:
    def __init__(self, max_depth, is_two_dimension):
        self.max_depth = max_depth
        self.is_two_dimension = is_two_dimension
        self.root = None
    
    def create_tree(self, max_depth, parent=None):
        value = None
        children = []
        is_leaf = False
        random_depth = random.randint(0, max_depth)
        if random_depth == 0:
            if self.is_two_dimension:
                value = leaf_two_dimension()
            else:
                value = leaf_one_dimension()
            is_leaf = True
        else:
            value = operator()
        node = Node(value, parent, children, is_leaf)
        if parent != None:
            parent.children.append(node)
        if self.root == None:
            self.root = node
        if random_depth == 0:
            return
        random_depth -= 1
        child_number = 2
        if single_operation(value):
            child_number = 1
        for i in range(child_number):
            self.create_tree(random_depth, node)
    
    def __str__(self, node: Node):
        if node.is_leaf:
            return node.value
        else:
            if single_operation(node.value):
                value = self.__str__(node.children[0])
                if isinstance(value, str):
                    return f"{node.value}({value})"
                else:
                    if node.value == 'sin':
                        return math.sin(value)
                    else:
                        return math.cos(value)
            else:
                left_value = self.__str__(node.children[0])
                right_value = self.__str__(node.children[1])
                if isinstance(left_value, str) or isinstance(right_value, str):
                    return f"({left_value}) {node.value} ({right_value})"
                else:
                    if node.value == '+':
                        return left_value + right_value
                    elif node.value == '-':
                        return left_value - right_value
                    elif node.value == '*':
                        return left_value * right_value
                    elif node.value == '/':
                        if right_value == 0:
                            return 
                        return left_value / right_value
                    elif node.value == '^':
                        try:
                            return math.pow(left_value, right_value)
                        except:
                            return

    def calculate(self, input, node: Node):
        if node.is_leaf:
            if not self.is_two_dimension:
                if node.value == 'x1':
                    return input
                else:
                    return node.value
            else:
                if node.value == 'x1':
                    return input[0]
                elif node.value == 'x2':
                    return input[1]
                else:
                    return node.value
        else:
            value = None
            left_value = None
            right_value = None
            if len(node.children) == 1:
                value = self.calculate(input, node.children[0])
            else:
                left_value = self.calculate(input, node.children[0])
                right_value = self.calculate(input, node.children[1])
            if not single_operation(node.value):
                if left_value == None or right_value == None:
                    return
            else:
                if value == None:
                    return
            if node.value == 'sin':
                return math.sin(value)
            elif node.value == 'cos':
                return math.cos(value)
            elif node.value == '+':
                return left_value + right_value
            elif node.value == '-':
                return left_value - right_value
            elif node.value == '*':
                return left_value * right_value
            elif node.value == '/':
                if right_value == 0:
                    return
                return left_value / right_value
            elif node.value == '^':
                try:
                    return math.pow(left_value, right_value)
                except:
                    return
                
    def mse(self, x_data, y_data):
        y_prediction = []
        for i, x in enumerate(x_data):
            y = self.calculate(x, self.root)
            if y == None:
                y = y_data[i] * 100
            y_prediction.append(y)
        return mean_squared_error(y_data, y_prediction)
    
    def generate_random_node(self, node: Node):
        nodes = []
        nodes.append(node)
        i = 0
        while i != len(nodes):
            n = nodes[i]
            for child in n.children:
                nodes.append(child)
            i += 1
        return random.choice(nodes)