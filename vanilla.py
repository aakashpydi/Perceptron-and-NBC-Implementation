import sys
import pandas as pd
import numpy as np

class Perceptron:

    def __init__(self, max_iter, train_data_entries, test_data_entries):
        self.max_iterations = max_iter
        self.train_data_entries = train_data_entries
        self.test_data_entries = test_data_entries
        self.weights = np.zeros(len(train_data_entries[0]) - 1) #number of attributes
        self.bias = 0.0

    def train(self):
        for i in range(self.max_iterations):
            for data_tuple in self.train_data_entries:
                class_label = data_tuple[0]
                attributes = np.array(data_tuple[1:len(data_tuple)])

                f_x_value = self.predict(data_tuple)
                error = class_label - f_x_value
                if error != 0.0:
                    self.bias += error
                    self.weights += (error * attributes)
        #print "bias: \t" + str(self.bias)
        #print "weights: \t" + str(self.weights)

    def predict(self, data_tuple):
        attributes = np.array(data_tuple[1:len(data_tuple)])
        f_x_value = np.dot(attributes, self.weights) + self.bias
        if f_x_value >= 0:
            f_x_value = 1.0
        else:
            f_x_value = 0.0
        return f_x_value

    def test(self):
        zero_one_loss = 0.0
        accuracy = 0.0
        for data_tuple in self.test_data_entries:
            class_label = data_tuple[0]

            attributes = np.array(data_tuple[1:len(data_tuple)])
            f_x_value = self.predict(data_tuple)
            #print 'PREDICTED: ' +str(f_x_value)
            #print 'ACTUAL: ' + str(class_label) +"\n"
            if class_label == f_x_value:
                zero_one_loss += 0.0
                accuracy += 1.0
            else:
                zero_one_loss += 1.0
        print "ZERO-ONE LOSS=" + str(zero_one_loss/len(self.test_data_entries))
        #print "Accuracy: \t" + str(accuracy/len(self.test_data_entries))


train_filename = sys.argv[1]
test_filename = sys.argv[2]
max_iter = 2
if len(sys.argv) > 3:
    max_iter = int(sys.argv[3])

data_train = pd.read_csv(train_filename, sep=',', quotechar='"', header=0, engine='python')
data_test = pd.read_csv(test_filename, sep=',', quotechar='"', header=0, engine='python')

merged_data = data_train.append(data_test)

#print merged_data.columns
#print list(data_train.columns)

columns_to_encode = merged_data.columns.values.tolist()
columns_to_encode = columns_to_encode[0:len(columns_to_encode) -1] #last column class_label
#print columns_to_encode

merged_data = pd.get_dummies(merged_data, columns=columns_to_encode)

X = merged_data.as_matrix()
X_train = X[0:len(data_train), :]
X_test = X[len(data_train):len(data_train)+len(data_test), :]


p = Perceptron(max_iter, X_train, X_test)
p.train()
p.test()
