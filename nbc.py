import sys
import pandas as pd
import numpy as np


class NBC:

    def __init__(self, data_train, data_test):
        self.data_train = data_train
        self.data_test = data_test

        self.class_1_value_dict = dict()
        self.class_0_value_dict = dict()

        self.zero_one_loss = 0.0
        self.squared_loss = 0.0

        self.value_set = set()
        for data_tuple in self.data_train.values:
            columns = list(self.data_train.columns)
            for i, element in enumerate(data_tuple):
                value = str(columns[i]) +"="
                value += str(element)
                self.value_set.add(value)
        # print self.value_set
        # print len(self.value_set)
        for data_tuple in self.data_test.values:
            columns = list(self.data_test.columns)
            for i, element in enumerate(data_tuple):
                value = str(columns[i]) +"="
                value += str(element)
                self.value_set.add(value)
        self.value_list = list(self.value_set)

        columns = self.data_train.columns
        self.possible_value_count_dict = dict()
        for column in columns:
            self.possible_value_count_dict[column] = 0

        for value in self.value_list:
            prefix = value[:value.find('=')]
            self.possible_value_count_dict[prefix] += 1

        #print self.possible_value_count_dict


        for value in self.value_list:
            self.class_0_value_dict[value] = 0
            self.class_1_value_dict[value] = 0
        #print self.class_0_value_dict
        #print self.value_list
        #print len(self.value_list)
        # print self.value_set
        # print len(self.value_set)

    def train(self):
        for data_tuple in self.data_train.values:
            columns = list(self.data_train.columns)
            class_label = data_tuple[len(columns) - 1]

            if class_label == 1.0:
                for i, element in enumerate(data_tuple):
                    value = str(columns[i]) +"="
                    value += str(element)
                    self.class_1_value_dict[value] += 1
            else:
                for i, element in enumerate(data_tuple):
                    value = str(columns[i]) +"="
                    value += str(element)
                    self.class_0_value_dict[value] += 1

        ### USED FOR ANSWERING QUESTIONS ###
        # TO_FIND = 'attire'
        # for key in self.class_1_value_dict:
        #     prefix = key[:key.find('=')]
        #     #print prefix
        #     if prefix == TO_FIND:
        #         sm_conditional_prob_temp = float(self.class_1_value_dict[key] + 1)/float(self.class_1_value_dict['goodForGroups=1'] + self.possible_value_count_dict[prefix])
        #         conditional_prob_temp = float(self.class_1_value_dict[key])/float(self.class_1_value_dict['goodForGroups=1'])
        #         #print self.possible_value_count_dict[prefix]
        #         print "Without Smoothing:\tP( "+str(key) + " | goodForGroups=1) = "  + str(conditional_prob_temp)
        #         print "With Smoothing:\t\tP( "+str(key) + " | goodForGroups=1) = "  + str(sm_conditional_prob_temp)
        #         #print self.class_1_value_dict[key]
        #         print "\n"
        #
        # for key in self.class_0_value_dict:
        #     prefix = key[:key.find('=')]
        #     #print prefix
        #     if prefix == TO_FIND:
        #         sm_conditional_prob_temp = float(self.class_0_value_dict[key] + 1)/float(self.class_0_value_dict['goodForGroups=0'] + self.possible_value_count_dict[prefix])
        #         conditional_prob_temp = float(self.class_0_value_dict[key])/float(self.class_0_value_dict['goodForGroups=0'])
        #         #print self.possible_value_count_dict[prefix]
        #         print "Without Smoothing:\tP( "+str(key) + " | goodForGroups=0) = "  + str(conditional_prob_temp)
        #         print "With Smoothing:\t\tP( "+str(key) + " | goodForGroups=0) = "  + str(sm_conditional_prob_temp)
        #         #print self.class_1_value_dict[key]
        #         print "\n"

        ####################################
        #print self.class_0_value_dict
        #print self.class_1_value_dict

    def predict(self, column_list, data_tuple):
        class_0_count = float(self.class_0_value_dict['goodForGroups=0'])
        class_1_count = float(self.class_1_value_dict['goodForGroups=1'])

        #print class_0_count
        #print class_1_count

        ## evaluate probability tuple belongs to class 0
        class_0_prob = class_0_count/(class_0_count + class_1_count)
        #print class_0_prob

        for i, element in enumerate(data_tuple):
            if i == len(data_tuple) - 1:
                continue
            value = str(column_list[i]) + "="
            value += str(element)
            conditional_prob_temp = float(self.class_0_value_dict[value] + 1)/float(class_0_count + self.possible_value_count_dict[column_list[i]])

            class_0_prob *= conditional_prob_temp


        ## evaluate probability tuple belongs to class 1
        class_1_prob = class_1_count/(class_0_count + class_1_count)

        for i, element in enumerate(data_tuple):
            if i == len(data_tuple) - 1:
                continue
            value = str(column_list[i]) + "="
            value += str(element)
            conditional_prob_temp = float(self.class_1_value_dict[value] + 1)/float(class_1_count + self.possible_value_count_dict[column_list[i]])
            class_1_prob *= conditional_prob_temp
        # print class_0_prob
        # print class_1_prob
        temp = class_0_prob + class_1_prob
        class_0_prob = class_0_prob / temp
        class_1_prob = class_1_prob / temp
        # class_0_prob = class_0_prob/(class_0_prob + class_1_prob)
        # class_1_prob = class_1_prob/(class_0_prob + class_1_prob)

        # print class_0_prob/(class_0_prob + class_1_prob)
        # print class_1_prob/(class_0_prob + class_1_prob)
        #print "\n"
        actual_class = int(data_tuple[len(data_tuple) - 1])
        if actual_class == 0:
            self.squared_loss += (1-class_0_prob)*(1-class_0_prob)
        else:
            self.squared_loss += (1-class_1_prob)*(1-class_1_prob)

        #print  +str(actual_class)
        #print data_tuple
        if class_0_prob > class_1_prob:
            #print "returning 0\n"
            return 0
        else:
            #print "returning 1\n"
            return 1

    def test(self):
        accuracy = 0.0

        for data_tuple in self.data_test.values:
            #print data_tuple
            columns = list(self.data_train.columns)
            actual_class_label = int(data_tuple[len(columns) - 1])
            predicted_class_label = self.predict(self.data_test.columns, data_tuple)

            if actual_class_label == predicted_class_label:
                self.zero_one_loss += 0.0
                accuracy += 1.0
            else:
                self.zero_one_loss += 1.0
            #print actual_class_label
            #print predicted_class_label
            #print "\n"
        self.zero_one_loss = self.zero_one_loss/len(self.data_test.values)
        self.squared_loss = self.squared_loss/len(self.data_test.values)

        print "ZERO-ONE LOSS=" + str(self.zero_one_loss)
        #print "Accuracy: \t" + str(accuracy/len(self.data_test.values))


######################################################################################################
train_filename = sys.argv[1]
test_filename = sys.argv[2]

data_train = pd.read_csv(train_filename, sep=',', quotechar='"', header=0, engine='python')
#print data_train
data_test = pd.read_csv(test_filename, sep=',', quotechar='"', header=0, engine='python')

nbc = NBC(data_train, data_test)
nbc.train()
nbc.test()
######################################################################################################



# merged_data = data_train.append(data_test)
#
# #print merged_data.columns
# print list(data_train.columns)
#
# columns_to_encode = merged_data.columns.values.tolist()
# columns_to_encode = columns_to_encode[0:len(columns_to_encode) -1] #last column class_label
# #print columns_to_encode
#
# merged_data = pd.get_dummies(merged_data, columns=columns_to_encode)
#
# #print data
#
# X = merged_data.as_matrix()
# X_train = X[0:len(data_train), :]
# X_test = X[len(data_train):len(data_train)+len(data_test), :]
