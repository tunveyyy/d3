import sys
import time
import math
import json
import operator
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

class Node:
    
    node_count = 0
    
    def __init__(self, attribute_name, is_continuous, threshold, total_positive, total_negative, label, is_leaf, branch):
        self.attribute_name = attribute_name
        self.is_continuous = is_continuous
        self.threshold = threshold
        self.total_positive = total_positive
        self.total_negative = total_negative
        self.is_leaf = is_leaf
        self.label = label
        self.branch = branch
        Node.node_count += 1
        
    def get_total_nodes(self):
        return self.node_count

def read_data(train_data_path, dev_data_path, test_data_path):
    column_names = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'race', 'sex', 'hours', 'country', 'income']
    train_data = pd.read_csv(train_data_path, names = column_names)
    dev_data = pd.read_csv(dev_data_path, names = column_names)
    test_data = pd.read_csv(test_data_path, names = column_names)
    
    return (train_data, dev_data, test_data)

def get_initial_values(data):
    
    # Remove last column from attribute list
    attributes = data.columns.values
    index = np.argwhere(attributes == attributes[-1])
    attributes = np.delete(attributes, index)
    
    # Get positive and negative labels
    labels = np.unique(data.iloc[:, -1])
    positive_label = labels[0]
    negative_label = labels[1]
    
    # Get attribute value count
    attr_val_cnt = {}
    
    for attribute in attributes:
        attr = np.unique(data[attribute])
        attr_cnt = len(attr)
        attr_val_cnt[attribute] = attr_cnt
        
    return attributes, positive_label, negative_label, attr_val_cnt

class DecisionTree:
    
    def __init__(self, positive_label_val, negative_label_val, attr_val_count):
        self.positive_label_val = positive_label_val
        self.negative_label_val = negative_label_val
        self.attr_val_count = attr_val_count
        
    def build_tree(self, data, attributes):
        
        # Create a node
        node = Node(None, False, None, None, None, None, False, None)

        # Get positive and negative label
        values, count = np.unique(data.iloc[:, -1], return_counts = True)
        val_count = dict(zip(values, count))
        
        positive_label = 0
        negative_label = 0
        
        if(len(val_count) == 2):
            positive_label = list(val_count)[0]
            negative_label = list(val_count)[1]
            node.total_positive = val_count[positive_label]
            node.total_negative = val_count[negative_label]
        else:
            if(self.positive_label_val == list(val_count.keys())[0]):
                positive_label = list(val_count)[0]
                negative_label = 0 
                node.total_positive = val_count[positive_label]
                node.total_negative = 0
            else:
                positive_label = 0
                negative_label = list(val_count)[0]
                node.total_positive = 0
                node.total_negative = val_count[negative_label]
        
        if(node.total_negative == 0):
            node.is_leaf = True
            node.label = positive_label
            return node

        if(node.total_positive == 0):
            node.is_leaf = True
            node.label = negative_label
            return node

        if(len(attributes) == 0):
            node.is_leaf = True
            if(node.total_positive > node.total_negative):
                node.label = positive_label
            else:
                node.label = negative_label
            return node

        else:

            info_gains = {}
            thresholds = {}

            for attribute in attributes:
                if(np.issubdtype(data[attribute].dtype.name, np.integer)):
                    thresholds[attribute], info_gains[attribute] = self.calculate_threshold(data[attribute], data.iloc[:, -1])
                else:
                    info_gains[attribute] = self.information_gain(data[attribute], data.iloc[:, -1])

            # Attribute with maximum gain
            max_gain_attr = max(info_gains.items(), key=operator.itemgetter(1))[0]

            # If continuous attribute, set threshold value
            if(max_gain_attr in list(thresholds.keys())):
                node.threshold = thresholds[max_gain_attr]
                node.is_continuous = True

            node.attribute_name = max_gain_attr
            node.branch = {}
            
            # Check if the best attribute is continuous or categorical
            if(node.is_continuous):
                for value in ['True', 'False']:
                    if(value == 'True'):
                        data[max_gain_attr] = np.where(data[max_gain_attr] < node.threshold, 'True', 'False')
                    node.branch[value] = None
                    subset = data[data[max_gain_attr] == value]
                    if(subset.shape[0] == 0):
                        node.branch[value] = Node(None, True, None, None, None, None, True, None)
                        c, v = np.unique(data.iloc[:,-1], return_counts = True)
                        c_v = dict(zip(c, v))
                        key = max(c_v.items(), key=operator.itemgetter(1))[0]
                        node.branch[value].label = key
                    else:
                        index = np.argwhere(attributes == max_gain_attr)
                        attributes = np.delete(attributes, index)
                        node.branch[value] = self.build_tree(subset, attributes)
                
            else:
                    # For each unique value from attribute column
                for value in np.unique(data[max_gain_attr]):
                    node.branch[value] = None
                    subset = data[data[max_gain_attr] == value]
                    if(subset.shape[0] == 0):
                        node.branch[value] = Node(None, False, None, None, None, None, True, None)
                        c, v = np.unique(data.iloc[:,-1], return_counts = True)
                        c_v = dict(zip(c, v))
                        key = max(c_v.items(), key=operator.itemgetter(1))[0]
                        node.branch[value].label = key
                    else:
                        index = np.argwhere(attributes == max_gain_attr)
                        attributes = np.delete(attributes, index)
                        node.branch[value] = self.build_tree(subset, attributes)
                          
        return node
    
    def calculate_threshold(self, data, label):
        data = data.values
        label = label.values
        indexes = data.argsort()
        data = np.flip(data[indexes[::-1]])
        label = np.flip(label[indexes[::-1]])
        label_df = pd.DataFrame(label)
        candidate_threshold = []
        info_gains = []

        for i in range(data.size - 1):
            threshold = data[i] + (data[i + 1] - data[i]) / 2
            if threshold not in candidate_threshold:
                candidate_threshold.append(threshold)
            else:
                continue

        for threshold in candidate_threshold:
            values = data < threshold
            values_df = pd.DataFrame(values)
            gain = self.information_gain(values_df, label_df)
            info_gains.append(gain)

        m = max(info_gains)
        max_posn = [i for i, j in enumerate(info_gains) if j == m]
        candidate_threshold = np.array(candidate_threshold)
        threshold = candidate_threshold[max_posn][0]
        
        return threshold, m
    
    def information_gain(self, data, label):
        attributes, count = np.unique(data, return_counts = True)
        attribute_count = dict(zip(attributes, count))
        label = label.values
        data = data.values
        entropies = []
        for attribute in attribute_count:
            index = np.where(data == attribute)
            op_cls = np.take(label, index)[0]
            ops, total = np.unique(op_cls, return_counts = True)
            ops_total = dict(zip(ops, total))
            entropy = 0
            try:
                entropy = self.calculate_entropy(ops_total[list(ops_total)[0]], ops_total[list(ops_total)[1]])
            except:
                entropy = self.calculate_entropy(ops_total[list(ops_total)[0]])
            entropy = entropy * (attribute_count[attribute] / len(label))
            entropies.append(entropy)
        ops, total = np.unique(label, return_counts = True)
        ops_total = dict(zip(ops, total))
        
        entropy = self.calculate_entropy(ops_total[list(ops_total)[0]], ops_total[list(ops_total)[1]])
        information = entropy - np.sum(entropies)
        return information
    
    def calculate_entropy(self, positive, negative = 0):
        total = positive + negative
        if negative == 0:
            return - (positive/total) * math.log((positive/total), 2)
        else:
            return - (positive/total) * math.log((positive/total), 2) - (negative/total) * math.log((negative/total), 2)
        
    
    def prune(self, data, node):

        if(node.is_leaf == False):

            # Accuracy for without prune
            full_accuracy = self.score(data, node)

            # Accuracy for prune
            node.is_leaf = True
            node.label = self.get_label(node)
            prune_accuracy = self.score(data, node)

            if(full_accuracy > prune_accuracy):
                node.is_leaf = False
                node.label = None
                branches = list(node.branch.keys())
                for branch_name in branches:
                    temp_node = node.branch[branch_name]
                    if(node.is_continuous == True):
                        subset = None
                        if(branch_name == 'True'):
                            subset = data[data[node.attribute_name] < node.threshold]
                        else:
                            subset = data[data[node.attribute_name] >= node.threshold]
                        subset = subset.drop([node.attribute_name], axis = 1)
                        return prune(subset, temp_node)
                    else:
                        subset = data[data[node.attribute_name] == branch_name]
                        subset = subset.drop([node.attribute_name], axis = 1)
                        return prune(subset, temp_node)

            else:
                return node
        else:
            return node

    def get_label(self, node):
        if(node.total_positive > node.total_negative):
            return self.positive_label_val
        else:
            return self.negative_label_val
    
    def score(self, data, node):
        root = node
        y_hat = []
        y = data.iloc[:, -1].values
        x = data.drop(data.columns.values[-1], axis = 1)
        stat = False
        solution = None

        for key, value in x.iterrows():
            while(node.is_leaf != True):
                val = x[node.attribute_name][key]
                if(np.issubdtype(x[node.attribute_name].dtype.name, np.integer)):
                    if(val < node.threshold):
                        node = node.branch['True']
                    else:
                        node = node.branch['False']
                else:
                    if(val not in list(node.branch.keys())):
                        stat = True
                        positive = node.total_positive
                        negative = node.total_negative
                        if(positive > negative):
                               solution = ' <=50K'
                        else:
                            solution = ' >50K'
                        break
                    else:    
                        node = node.branch[val]  
            if(stat):
                y_hat.append(solution)
            else:
                y_hat.append(node.label)
            node = root

        y_hat = np.asarray(y_hat)
        accuracy = self.calculate_accuracy(y_hat, y)
        return accuracy
    
    def calculate_accuracy(self, y_hat, y):
        count = np.equal(y_hat, y)
        value, count = np.unique(count, return_counts = True)
        val_count = dict(zip(value, count))

        accuracy = (val_count[True] / y_hat.shape[0])

        return accuracy

def create_output(train_accuracy, dev_accuracy, test_accuracy, train_accuracy_prune, dev_accuracy_prune, test_accuracy_prune, count):
    
    file = open('output.txt', 'a+')
    file.write("\n\n##### ID3 Decision Tree Implementation Output #####")
    file.write("\n\nNumber of nodes in the tree: " + str(count)+ '\n\n')
    file.write("Training accuracy without pruning: " + str(train_accuracy)+ ' %\n')
    file.write("Validation accuracy without pruning: " + str(dev_accuracy)+ ' %\n')
    file.write("Test accuracy without pruning: " + str(test_accuracy) + ' %\n')
    file.write("\nTraining accuracy with pruning: " + str(train_accuracy_prune)+ ' %\n')
    file.write("Validation accuracy with pruning: " + str(dev_accuracy_prune)+ ' %\n')
    file.write("Test accuracy with pruning: " + str(test_accuracy_prune)+ ' %\n')
    file.close()    


def main():
    train_data_path = 'income-data/income.train.txt'
    dev_data_path = 'income-data/income.dev.txt'
    test_data_path = 'income-data/income.test.txt'
    
    # Read data
    (train_data, dev_data, test_data) = read_data(train_data_path, dev_data_path, test_data_path)
    
    # Get initial values
    (attributes, positive_lab, negative_lab, attr_val_cnt) = get_initial_values(train_data)
    
    # Create an instance of Decision Tree
    dt = DecisionTree(positive_lab, negative_lab, attr_val_cnt)
    
    # Build the tree
    root = dt.build_tree(train_data, attributes)
    
    # Print total number of nodes built in the tree
    count = root.get_total_nodes()
    
    # Get all the accuracies
    train_accuracy = round(dt.score(train_data, root) * 100, 2)
    dev_accuracy = round(dt.score(dev_data, root) * 100, 2)
    test_accuracy = round(dt.score(test_data, root) * 100, 2)

    # Begin Pruning
    new_root = dt.prune(dev_data, root)
    
    # Get accuracy after pruning
    train_accuracy_1 = round(dt.score(train_data, new_root) * 100, 2)
    dev_accuracy_1 = round(dt.score(dev_data, new_root) * 100, 2)
    test_accuracy_1 = round(dt.score(test_data, new_root) * 100, 2)

    create_output(train_accuracy, dev_accuracy, test_accuracy, train_accuracy_1, dev_accuracy_1, test_accuracy_1, count)

if __name__ == '__main__':
    main()