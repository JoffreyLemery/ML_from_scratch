import numpy as np 
import pandas as pd
from functools import reduce
import operator
import sys

dataframe = pd.DataFrame([[140000, 50, 'Yes', 2, 1,'No'],
                         [150000, 55, 'No', 3, 1,'Yes'],
                         [100000, 38, 'Yes', 2, 1,'No'],
                         [200000, 72, 'No', 3, 2,'No'],
                         [220000, 70, 'Yes', 4, 2,'Yes'],
                         [120000, 40, 'No', 2, 1,'Yes'],
                         [198000, 68, 'No', 3, 2,'Yes'],
                         [130000, 54, 'No', 2, 1,'No'],
                         [140000, 62, 'No', 3, 1,'No'],
                         [190000, 79, 'Yes', 2, 1,'Yes'],
                         [170000, 67, 'Yes', 4, 3,'No'],
                         [90000, 40, 'No', 2, 1,'Yes']],
                         columns=['prix', 'surface', 'garage', 'nb_piece','nb_chambre', 'balcony'])
dataframe



class decision_tree_regressor :
    """The purpose of this class is to create an automatic learning 
    algorithm decision tree regressor"""
    
    def __init__(self, target, dataframe, max_depth):
        self.target = target
        self.dataframe = dataframe
        self.max_depth = max_depth
        
        
    def __quanti_split__(self, feature, value, dataset) :
        """Thus function aims to split the dataset according to the value of the feature
        INPUT
        - feature : integer which represente the variable to splitthe variable
        - dataset : dataframe to split
        OUTPUT
        -left = dataset with values =< value
        -right = dataset with values > value
        """
        left_split = dataset[dataset.loc[:,feature]<=value]
        right_split = dataset[dataset.loc[:,feature]>value]
        
        return left_split, right_split
        
        
    def __quali_split__(self, feature, value, dataset) :
        """Thus function aims to split the dataset accordinf to the value of the feature
        INPUT
        - feature : integer which represente the variable to splitthe variable
        - dataset : dataframe to split
        OUTPUT
        -left = dataset with values =< value
        -right = dataset with values > value
        """
        left_dataset = dataset[dataset.loc[:,feature]==value]
        right_dataset = dataset[dataset.loc[:,feature]!=value]
        
        return left_dataset, right_dataset

    
    def __mse__(self, dataset) :
        """Mean Square Error : MSE
        Calcul the MSE of the dataset
        INPUT 
           - dataset : dataset which contains 'self.target'
        OUTPUT
           - impurity : calculated impurity
        """
        rows = dataset[self.target]
        prediction_mean = np.ones(rows.shape[0])*np.mean(rows)
        if len(rows) == 0 :
            mse = 0
        else :
            mse = (1/len(rows))*np.sum((rows-prediction_mean)**2)
        return mse
    
    
    def __split_evaluator__(self, dataset, left_dataset, right_dataset) : 
        """ Calcul the cost of separation - Used for the optimization problem : Lowest cost
        INPUT 
           - left_dataset : left dataframe which come from a quali/quanti split
           - right_dataset : right dataframe which come from a quali/quanti split
        OUTPUT
           - cost : coût de la séparation
        """
        left_mse = self.__mse__(left_dataset)
        nb_left = left_dataset.shape[0]
        right_mse = self.__mse__(right_dataset)
        nb_right = right_dataset.shape[0]
        nb_tot = nb_left+ nb_right
        cost = left_mse* nb_left/nb_tot + right_mse* nb_right/nb_tot
        return cost
    
    
    def __value_type__(self, value,vect_type_value) :
        """Function to check the consistancy of data and to choose between quanti or quali split
        INPUT : 
        - Value : Value to evaluate
        OUTPUT :
        - vect_type_value : Dataset which containt the type of values"""
        
        if isinstance(value,str) :
            vect_type_value.append('str')
            return vect_type_value
        
        elif isinstance(value,int) :
            vect_type_value.append('int')
            return vect_type_value
            
        else :
            print("Check your data")
            sys.exit()
    
    def __test_quali__(self, dataset, feature) :
        """ Test all possible split for a feature
        INPUT 
        - dataset : dataset to evaluate
        - feature : variable of the dataset to evaluate
        OUTPUT 
        - df_eval : dataframe which cost of every separation
        """
        df_eval = pd.DataFrame([], columns = ('feature', 'value', 'nature', 'cost'))
        for value in dataset.loc[:,feature].unique() :
            left_dataset, right_dataset = self.__quali_split__(feature,value,dataset)
            cost = self.__split_evaluator__(dataset,left_dataset, right_dataset)
            df_eval=df_eval.append(pd.DataFrame([[feature, value, 'quali', cost]], columns = ('feature', 'value', 'nature', 'cost')))
        return df_eval    
    
    
    def __test_quanti__(self, dataset, feature) :
        """ Test all possible split for a feature
        INPUT 
        - dataset : dataset to evaluate
        - feature : variable of the dataset to evaluate
        OUTPUT 
        - df_eval : dataframe which cost of every separation
        """
        df_eval = pd.DataFrame([], columns = ('feature', 'value', 'nature', 'cost'))
        value_to_test = np.unique((dataset.loc[:, feature].sort_values()[1:].values + dataset.loc[:, feature].sort_values()[:-1].values)/2)
        for value in dataset.loc[:,feature].unique() :
            left_dataset, right_dataset = self.__quanti_split__(feature,value,dataset)
            cost = self.__split_evaluator__(dataset,left_dataset, right_dataset)
            df_eval=df_eval.append(pd.DataFrame([[feature, value, 'quanti', cost]], columns = ('feature', 'value', 'nature', 'cost')))
        return df_eval
    
    def __find_best_split__(self, dataset) : 
        """ Find best split for a given dataset, through features and values
        INPUT :
        - dataset : dataset to evaluate
        OUTPUT : def_eval : dataset which contains 'feature','value',
        'nature','cost'
        """
        df_eval = pd.DataFrame([], columns = ('feature', 'value', 'nature', 'cost'))
        columns_to_feature = dataset.columns[np.logical_not(dataset.columns == self.target)]
        columns_to_feature
 
        for column in columns_to_feature :
            vect_type_value = []
        
            for value in dataset[column] :
                vect_type_value = self.__value_type__(value, vect_type_value)
                
            if len(np.unique(vect_type_value)) > 1 :
                print("Check your data, differents types are found on a same column")
                sys.exit()
                
            if np.unique(vect_type_value)[0] == 'str' and len(np.unique(vect_type_value)) == 1 : 
                df_eval = df_eval.append(self.__test_quali__(dataset, column))
 
            if np.unique(vect_type_value)[0] == 'int' and len(np.unique(vect_type_value)) == 1 :
                df_eval = df_eval.append(self.__test_quanti__(dataset, column))
                
        df_eval = df_eval.reset_index(drop=True)
        id_cost_min = df_eval['cost'].idxmin(axis=0, skipna=True)
        
        return df_eval.iloc[id_cost_min, :]
    
    def __create_leaf__(self, dataset):
        """ Leaf creation 
        INPUT 
        - dataset : dataset of the leaf to construct
        OUTPUT 
        - leaf : created leaf with all required information :  labels of the leaf, population and prediction"""
        
        labels = dataset[self.target]
        pop = labels.shape[0]
        prediction = np.mean(labels)
        
        return leaf(labels, pop, prediction) # This function will call the leaf class to create a leaf

    
    def __training__(self, dataset, depth=0):
        """ This functions aims to construct the tree by he use of recursive fonction for node creation
        INPUT : 
        - dataset : dataset that is used to create the tree
        OUTPUT : 
        - node of the tree
        """
        
        #We check that we can still divise our dataset
        no_more_split = True
        columns = dataset.columns[np.logical_not(dataset.columns == 'self.target')]
        for column in columns :
            if len(np.unique(dataset[column])) > 1 :
                no_more_split = False

                   
        # Si le dataset est pure, ou que la profondeur maximum est atteinte ou
        # que le dataset ne peut plus être séparé nous créons une feuille
        if no_more_split ==True or len(np.unique(dataset[self.target]))==1 or depth == self.max_depth :
            return self.__create_leaf__(dataset)
        
        #Find the best feature and value for the data split
        split_eval = self.__find_best_split__(dataset)
                   
        #Chech that the new cost is lower thant the previous one, otherwise it is useless
        if split_eval['cost'] >= self.__mse__(dataset):
            return self.__create_leaf__(dataset)
        
        #Dataset split according to the selected feature, the value and the nature of the value
        if split_eval['nature'] == 'quali' :
            left_branch, right_branch = self.__quali_split__(split_eval['feature'], split_eval['value'], dataset)
        elif split_eval['nature'] == 'quanti' :
            left_branch, right_branch = self.__quanti_split__(split_eval['feature'], split_eval['value'], dataset)
        
        #Recursive training
        left_node = self.__training__(left_branch, depth +1)
        right_node = self.__training__(right_branch, depth +1)
        
        #Return of the the tree's nodes
        return node(split_eval['feature'], 
                    split_eval['value'], 
                    split_eval['cost'], 
                    split_eval['nature'],
                    left_node,
                    right_node, 
                    depth, 
                    dataset.shape[0])
    def fit(self) :
        """ Traininf of the model
        OUTPUT 
        - node : roots of the tree"""
        return self.__training__(self.dataframe)



class node:
    """ This class aims to represent the node of the tree. Nodes are the body of the tree
    """
    def __init__(self, feature, value, cost, nature, left_branch, right_branch, depth, pop) : 
        self.feature = feature
        self.value = value
        self.cost = cost
        self.nature = nature
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.depth = depth
        self.pop = pop
        
    def __split__(self):
        if self.nature == 'quali' : 
            return self.feature + "==" + str(self.value)
        if self.nature == 'quanti' : 
            return self.feature + "<=" + str(self.value)
    
    def make_pred(self, datatest):
        if self.nature  == 'quali':
            if datatest[self.feature][0] == self.value :
                    self.left_branch.make_pred(datatest)
            else : self.right_branch.make_pred(datatest)

        elif self.nature  == 'quanti':
            if datatest[self.feature][0] <= self.value :
                    self.left_branch.make_pred(datatest)
            else : self.right_branch.make_pred(datatest)


class leaf : 
    """This class aims to represent the leaf of the tree. It will be called by the __create_leaf__ 
    when the recursive function of node split with reach a exit condition (leaf creation)
    """
    def __init__(self, label, pop, prediction):
        self.label = label
        self.pop = pop
        self.prediction = prediction

    def make_pred(self, datatest):
        print('The prediction is : {}'.format(self.prediction))

#En raison de changement à venir dans les fonctionnalités, des warning sont levés. Nous ne voulons pas les voir

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def print_tree(node, spacing=""):
    """Recursive print of the tree
    INPUT 
    - node : Branch to print
    - spacings : Space which represent the depth of nodes """

    # Différents affichages si c'est une feuille 
    if isinstance(node, leaf):
        print (spacing + "Predict", node.prediction)
        return

    # Affichage de la condition de la séparation
    print (spacing + node.__split__())

    # Dans le cas où la condition est vérifiée
    print (spacing + '--> True:')
    print_tree(node.left_branch, spacing + "  ")

    # Dans le cas où la condition n'est pas vérifiée
    print (spacing + '--> False:')
    print_tree(node.right_branch, spacing + "  ")


tree = decision_tree_regressor('prix', dataframe, 4)
tree_trained = tree.fit()
print_tree(tree_trained)


datatest = pd.DataFrame([[130, 'No', 3, 1,'Yes']], columns=['surface', 'garage', 'nb_piece','nb_chambre', 'balcony'])
datatest
tree_trained.make_pred(datatest)