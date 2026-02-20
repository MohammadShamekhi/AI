import pandas as pd
import numpy as np
from DT import Node
from treelib import Node as N, Tree

train = pd.read_csv("Restaurant.csv")

train["Alt"] = train["Alt"].map( {"Yes": 1, "No": 0} ).astype(int)
train["Bar"] = train["Bar"].map( {"Yes": 1, "No": 0} ).astype(int)
train["Fri"] = train["Fri"].map( {"Yes": 1, "No": 0} ).astype(int)
train["Hun"] = train["Hun"].map( {"Yes": 1, "No": 0} ).astype(int)
train["Pat"] = train["Pat"].map( {"Full": 2, "Some": 1, "Non": 0} )
train["Price"] = train["Price"].map( {"$$$": 2, "$$": 1, "$": 0} )
train["Rain"] = train["Rain"].map( {"Yes": 1, "No": 0} ).astype(int)
train["Res"] = train["Res"].map( {"Yes": 1, "No": 0} ).astype(int)
train["Type"] = train["Type"].map( {"French": 3, "Thai": 2, "Burger": 1, "Italian": 0} )
train["Est"] = train["Est"].map( {"0-10": 3, "10-30": 2, "30-60": 1, ">60": 0} )
train["Goal"] = train["Goal"].map( {"Yes": 1, "No": 0} ).astype(int)

tree = Tree()
dt = Node(train, None, set(train.columns) - {"Goal"}, train)
dt.generate_decision_tree("Goal", tree)
print("Accuracy =",(dt.test(train, "Goal") / len(train)) * 100)
tree.to_graphviz("hello.dot")
tree.show()
import subprocess
subprocess.call(["dot", "-Tpng", "hello.dot", "-o", "DecisionTree.png"])