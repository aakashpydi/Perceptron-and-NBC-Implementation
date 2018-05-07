import sys
import pandas as pd
import numpy as np

class Perceptron:

    def __init__(self, max_iter, train_data_entries, test_data_entries):
        self.max_iterations = max_iter
        self.train_data_entries = train_data_entries
        self.test_data_entries = test_data_entries
        self.weights = np.zeros(len(train_data_entries[0]) - 1) #number of attributes
        self.average_weights = np.zeros(len(train_data_entries[0]) - 1) #number of attributes
        self.bias = 0.0
        self.average_bias = 0.0
        self.zero_one_loss = 0.0

    def train(self):
        step = float(self.max_iterations * len(self.train_data_entries))
        #print step
        for i in range(self.max_iterations):
            for data_tuple in self.train_data_entries:
                class_label = data_tuple[0]
                attributes = np.array(data_tuple[1:len(data_tuple)])

                f_x_value = self.predict(data_tuple)
                error = class_label - f_x_value
                if error != 0.0:
                    self.bias += error
                    self.weights += (error * attributes)

                    self.average_bias += (step/(self.max_iterations * len(self.train_data_entries)))*error
                    self.average_weights += (step/(self.max_iterations * len(self.train_data_entries)))*error*attributes
                step -= 1
        # print "ave bias: \t" + str(self.average_bias)
        # print "ave weights: \t" + str(self.average_weights)
        # print "\n\nbias: \t" + str(self.bias)
        # print "weights: \t" + str(self.weights)
        self.bias = self.average_bias
        self.weights = self.average_weights

    def predict(self, data_tuple):
        attributes = np.array(data_tuple[1:len(data_tuple)])
        f_x_value = np.dot(attributes, self.weights) + self.bias
        if f_x_value >= 0:
            f_x_value = 1.0
        else:
            f_x_value = 0.0
        return f_x_value

    def test(self):
        accuracy = 0.0
        for data_tuple in self.test_data_entries:
            class_label = data_tuple[0]

            attributes = np.array(data_tuple[1:len(data_tuple)])
            f_x_value = self.predict(data_tuple)

            if class_label == f_x_value:
                self.zero_one_loss += 0.0
                accuracy += 1.0
            else:
                self.zero_one_loss += 1.0
        self.zero_one_loss = self.zero_one_loss/len(self.test_data_entries)
        print "ZERO-ONE LOSS=" + str(self.zero_one_loss)
        #print "Accuracy: \t" + str(accuracy/len(self.train_data_entries))

######################################################################################################
train_filename = sys.argv[1]
test_filename = sys.argv[2]
max_iter = 2
if len(sys.argv) > 3:
    max_iter=int(sys.argv[3])

data_train = pd.read_csv(train_filename, sep=',', quotechar='"', header=0, engine='python')
data_test = pd.read_csv(test_filename, sep=',', quotechar='"', header=0, engine='python')

merged_data = data_train.append(data_test)

#print merged_data.columns
#print list(data_train.columns)

columns_to_encode = merged_data.columns.values.tolist()
columns_to_encode = columns_to_encode[0:len(columns_to_encode) -1] #last column class_label
#print columns_to_encode

merged_data = pd.get_dummies(merged_data, columns=columns_to_encode)

#print data

X = merged_data.as_matrix()
X_train = X[0:len(data_train), :]
X_test = X[len(data_train):len(data_train)+len(data_test), :]

p = Perceptron(max_iter, X_train, X_test)
p.train()
p.test()
######################################################################################################



# p.predict()
#print error
# print pred_label
# print class_label
# print f_x_value
# print "\n"
# if pred_label != class_label:
#     self.bias += class_label
#     self.weights += (class_label * attributes)
# # error = pred_label - class_label
# if pred_label != class_label:
#     error = class_label * 2
# #print error
# if error != 0.0:
#     #print "entered"
#     self.bias = self.bias + error
#     self.weights = self.weights + (error * attributes)
