import pandas as pd
import numpy as np
from DT import Node
from treelib import Node as N, Tree

full_data = pd.read_csv("Airplane.csv")

# Mapping Gender
full_data["Gender"] = full_data["Gender"].map( {'Female': 1, 'Male': 0} ).astype(int)

# Mapping Customer Type
full_data["Customer Type"] = full_data["Customer Type"].map( {'Loyal Customer': 1, 'disloyal Customer': 0} ).astype(int)

# Mapping Age
full_data.loc[ full_data['Age'] <= 26.5, 'Age'] 					     = 0
full_data.loc[(full_data['Age'] > 26.5) & (full_data['Age'] <= 46.0), 'Age'] = 1
full_data.loc[(full_data['Age'] > 46.0) & (full_data['Age'] <= 65.5), 'Age'] = 2
full_data.loc[full_data['Age'] > 65.5, 'Age']                            = 3

# Mapping Type of Travel
full_data["Type of Travel"] = full_data["Type of Travel"].map( {'Business travel': 1, 'Personal Travel': 0} ).astype(int)

# Mapping Type of Class
full_data["Class"] = full_data["Class"].map({'Business': 3, 'Eco': 1, 'Eco Plus': 0})

# Mapping Flight Distance
full_data.loc[ full_data['Flight Distance'] <= 1269.0, 'Flight Distance'] 					                     = 0
full_data.loc[(full_data['Flight Distance'] > 1269.0) & (full_data['Flight Distance'] <= 2507.0), 'Flight Distance'] = 1
full_data.loc[(full_data['Flight Distance'] > 2507.0) & (full_data['Flight Distance'] <= 3745.0), 'Flight Distance'] = 2
full_data.loc[full_data['Flight Distance'] > 3745.0, 'Flight Distance']                                          = 3

# Mapping Departure Delay in Minutes
full_data.loc[full_data["Departure Delay in Minutes"] <= 265.333, "Departure Delay in Minutes"] = 0
full_data.loc[(full_data["Departure Delay in Minutes"] > 265.333) & (full_data["Departure Delay in Minutes"] <= 530.677), "Departure Delay in Minutes"] = 1
full_data.loc[(full_data["Departure Delay in Minutes"] > 530.677) & (full_data["Departure Delay in Minutes"] <= 796.0), "Departure Delay in Minutes"] = 2
full_data.loc[(full_data["Departure Delay in Minutes"] > 796.0) & (full_data["Departure Delay in Minutes"] <= 1061.333), "Departure Delay in Minutes"] = 3
full_data.loc[(full_data["Departure Delay in Minutes"] > 1061.333) & (full_data["Departure Delay in Minutes"] <= 1326.667), "Departure Delay in Minutes"] = 4
full_data.loc[full_data["Departure Delay in Minutes"] > 1326.667, "Departure Delay in Minutes"] = 5

# Mapping Arrival Delay in Minutes
avg = full_data["Arrival Delay in Minutes"].mean()
std = full_data["Arrival Delay in Minutes"].std()
null_count = full_data["Arrival Delay in Minutes"].isnull().sum()

age_null_random_list = np.random.randint(avg - std, avg + std, size=null_count)
full_data["Arrival Delay in Minutes"][np.isnan(full_data["Arrival Delay in Minutes"])] = age_null_random_list
full_data["Arrival Delay in Minutes"] = full_data["Arrival Delay in Minutes"].astype(int)

full_data.loc[full_data["Arrival Delay in Minutes"] <= 244.833, "Arrival Delay in Minutes"] = 0
full_data.loc[(full_data["Arrival Delay in Minutes"] > 244.833) & (full_data["Arrival Delay in Minutes"] <= 512.677), "Arrival Delay in Minutes"] = 1
full_data.loc[(full_data["Arrival Delay in Minutes"] > 512.677) & (full_data["Arrival Delay in Minutes"] <= 780.5), "Arrival Delay in Minutes"] = 2
full_data.loc[(full_data["Arrival Delay in Minutes"] > 780.5) & (full_data["Arrival Delay in Minutes"] <= 1048.333), "Arrival Delay in Minutes"] = 3
full_data.loc[(full_data["Arrival Delay in Minutes"] > 1048.333) & (full_data["Arrival Delay in Minutes"] <= 1316.167), "Arrival Delay in Minutes"] = 4
full_data.loc[full_data["Arrival Delay in Minutes"] > 1316.167, "Arrival Delay in Minutes"] = 5

# Mapping Type of satisfacation
full_data["satisfaction"] = full_data["satisfaction"].map( {'satisfied': 1, 'neutral or dissatisfied': 0} ).astype(int)

# Generating decition tree and test on test data
tree = Tree()
train_data = full_data[2000:]
train_data_1 = train_data[train_data["satisfaction"] == 1]
train_data_0 = train_data[train_data["satisfaction"] == 0]
train_data = [train_data_1[:2500], train_data_0[:2500]]
train_data = pd.concat(train_data)
test_data = full_data[:2000]
#train_data = full_data[10400:]
#test_data = full_data[:10400]
dt = Node(train_data, None, set(train_data.columns) - {"satisfaction", "id", "Unnamed: 0"}, full_data)
dt.generate_decision_tree("satisfaction", tree)
print("Accuracy =", (dt.test(test_data, "satisfaction") / len(test_data)) * 100)
tree.show()
tree.to_graphviz("hello.dot")
import subprocess
subprocess.call(["dot", "-Tpng", "hello.dot", "-o", "DecisionTree.png"])