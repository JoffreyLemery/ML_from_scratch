# %% [markdown]
# This code is intended to illustrate the methodology of establishing a CLASSIFICATION tree.
# - The data are taken from the internet to contextualize the exercise
# - The algorithm is based on the work of AIforYou and includes modifications
# - The algorithm is not optimized and is not intended to be put into production

# %% [markdown]
# Creation of a homemade DataFrame concerning the species belonging to the antrhopod group
# This data set is far from perfect but will serve as a first approximation


# %%
import numpy as np 
import pandas as pd
from functools import reduce
import operator
import sys

# %%
df=pd.DataFrame([['crustacé', 'crabe', 4, 0, 'Yes', 'No', 'No'],
                        ['crustacé', 'homard', 4, 0, 'Yes', 'No', 'No'],
                        ['crustacé', 'crevette', 4, 0, 'Yes', 'No', 'No'],
                        ['insectes', 'papillons', 2, 6, 'No', 'No', 'Yes'],
                        ['insectes', 'mante religieuse', 2, 6, 'No', 'No', 'Yes'],
                        ['insectes', 'libellule', 2, 6, 'No', 'No', 'Yes'],
                        ['insectes', 'coccinelle', 2, 6, 'No', 'No', 'Yes'],
                        ['arachnides', 'araignée', 0, 8, 'No', 'No', 'No'],
                        ['arachnides', 'mygales', 0, 8, 'No', 'No', 'No'],
                        ['arachnides', 'scorpion', 0, 8, 'No', 'No', 'No'],
                        ['arachnides', 'acarien', 0, 8, 'No', 'No', 'No'],
                        ['myriapode', 'mille-pattes', 2, 0, 'No', 'Yes', 'No'],
                        ['myriapode', 'scolopendre', 2, 0, 'No', 'Yes', 'No'],
                        ['myriapode', 'lithobie', 2, 0, 'No', 'Yes', 'No']],
                       columns=['Classification','Espèce','Nb antennes','Nb pattes régulier','céphalo-thorax',
                                'paires pattes', 'ailes'])

df_classification=df.drop(['Classification'],axis=1)
df

# %% [markdown]
# The objective of this code is to provide an algorithm to classify a species according to its characteristics. Also, our dataset allowing the creation of the tree does not require the species column

# %% [markdown]
# Creation of classes :
# 
# In order to make our tree functional we need to create three classes:
# - A class carrying all the functions that allow the logic of separation and cost calculation
# - A class of branches
# - A class of leaves

# %% [markdown]
# Classe de DECISION:
# 
# The objective is to create a class containing all the methods used by the algorithm:
# Let's start with the initialization function. This function will contain :
# 
# - The target of the species to classify
# - The dataframe
# - The maximum depth of the tree
# 
# 
# Separation
# The purpose of a classification tree is to start from the dataframe and impose rules to separate the dataframe.
# This one allows to discriminate the animals until we can classify them. 
# In order to be able to separate our data sets we must be able to separate according to quantitative and qualitative criteria

# %%
class decision_tree_classifier:
    #Class of the tree : Allowing the creation of the tree and related methods
    
    def __init__(self, target, dataframe, max_depth):
        self.target = target #target feature of the tree
        self.dataframe =  dataframe #dataframe to train the tree
        self.max_depth = max_depth #The maximum size (in depth) of the tree
    
    def __quanti_split__(self, feature, value, dataset):
        """The purpose of this function is to separate the dataframe according to the "value" value carried in the quantitative variable "fearture".
        INPUT
        - feature : integer corresponding to the variable to separate
        - value : integer corresponding to the quantitative data to separate our dataset
        - dataset : dataframe to separate
        OUTPUT
        -left = dataframe with the data of feature =< value
        -right = dataframe with feature data > value
        """

        left = dataset[dataset.loc[:,feature]<=value]
        right = dataset[dataset.loc[:,feature]>value]
        return left, right
    
    def __quali_split__(self, feature, value, dataset):
        """The purpose of this function is to separate the dataframe according to the "value" value carried in the quantitative variable "fearture".
        INPUT
        - feature : integer corresponding to the variable to separate
        - value : integer corresponding to the quantitative data to separate our dataset
        - dataset : dataframe to separate
        OUTPUT
        -left = dataframe with the data of feature == value
        -right = dataframe with feature data != value
        """
    
        left = dataset[dataset.loc[:,feature]==value]
        right = dataset[dataset.loc[:,feature]!=value]
    
        return left, right

    
    def __add_dict__(self, prec_dict, new_dict):
        """ This function merges dictionaries 
        by summing the values of similar keys: the objective is to make a count of the similar population
        INPUT 
           - prec_dict : dictionary to merge - see reduce function
           - new_dict : dictionary to merge - see reduce function
        OUTPUT 
           - prec_dict : dictionary with 'class' as key
           and in 'value' its occurrence in the dataset
        """

        if list(new_dict.keys())[0] in prec_dict:
            prec_dict[list(new_dict.keys())[0]] += 1 #is called by a following reduce fonction on a unitary dictionnary for every element of the dataframe
        else :
            prec_dict[list(new_dict.keys())[0]] = 1
        return prec_dict


    def __gini__(self, dataset):
        """Classification And Regression Tress : CART
        Compute the Gini indices of the dataset passed in parameter
        INPUT 
           - dataset : dataframe containing the variable 'self.target'
        OUTPUT
           - impurity : the impurity computed from the dataset
        """
        rows=dataset[self.target] #dataset extraction according to a feature target
        class_dict = list(map(lambda x: {x: 1}, rows))
        class_dict_sum = reduce(self.__add_dict__, class_dict,{'':0}) # To prevent the case of separation creating an empty set during the optimization calculation, it is important to pass an initializer
        occu_class = np.fromiter(class_dict_sum.values(), dtype=float) # Get the occurence of each class of the dataset and put them in a array

        pop=np.sum(occu_class)
        if pop == 0 : #Prevent a split try on a empty population
            Gini = 1
        else :
            Gini = 1-np.sum((occu_class/pop)**2)

        return Gini
    
    
    
    def __split_evaluator__(self,left_dataset,right_dataset):    
        """Calculate separation costs - This function will be used to find the cost related to a split
        INPUT 
           - left_dataset : left dataframe from a split quali/quanti
           - right_dataset : right dataframe from a split quali/quanti
        OUTPUT
           - cost : cost of the split
        """
        left_eval = self.__gini__(left_dataset)
        nb_left = left_dataset.shape[0]
        right_eval = self.__gini__(right_dataset)
        nb_right = right_dataset.shape[0]
        nb_total = nb_left + nb_right
        cost = nb_left/nb_total*left_eval + nb_right/nb_total*right_eval
        return cost
    
    
    def __test_quali__(self, dataset, feature):
        """ Test all possible separations of a qualitative variable
        INPUT 
        - dataset : dataset to evaluate
        - feature : variable of the dataset to evaluate
        OUTPUT 
        - df_eval : dataframe containing the cost of each separation"""

        df_eval = pd.DataFrame([], columns =('feature','value','nature','cost'))
        for value in dataset.loc[:, feature].unique():
            left_dataset,right_dataset=self.__quali_split__(feature,value,dataset)
            cost_result = self.__split_evaluator__(left_dataset,right_dataset)
            df_eval = df_eval.append(pd.DataFrame([[feature,value,'quali',cost_result]],columns =('feature','value','nature','cost')))
        return df_eval
    
    
    
    def __test_quanti__(self, dataset, feature):
        """ Test all possible separations of a qualitative variable
        INPUT 
       - dataset : dataset to evaluate
       - feature : variable of the dataset to evaluate
        OUTPUT 
       - df_eval : dataframe containing the cost of each separation"""
        
        df_eval = pd.DataFrame([], columns =('feature','value','nature','cost'))
        value_to_test = (dataset.loc[:, feature].sort_values()[1:].values + dataset.loc[:, feature].sort_values()[:-1].values)/2 #Calcul of the value to test as split value by the mean of two following values (sorted)
        for value in value_to_test :
            left_dataset,right_dataset=self.__quanti_split__(feature, value, dataset)
            cost_result = self.__split_evaluator__(left_dataset, right_dataset)
            df_eval = df_eval.append(pd.DataFrame([[feature,value,'quanti',cost_result]],columns =('feature','value','nature','cost')))
        return df_eval
        
    
    
    def __value_type__(self,value,vect_type_value) :
        """Protection method that sets an image vector of the type of values in the feature 
        to choose between "Quanti" and "Quali".
        INPUT : 
        - Value : Value to evaluate
        OUTPUT :
        - vect_type_value : Dataset containing the type of the value
        """
        
        if isinstance(value,str) :
            vect_type_value.append('str')
            return vect_type_value
        
        elif isinstance(value,int) :
            vect_type_value.append('int')
            return vect_type_value
            
        else :
            print("Check your data : must be Integer or String")
            sys.exit()
    

    def __find_best_split__(self, dataset) :
        """ Find the best separation of our dataset by scanning the features and test values
        INPUT:
       - dataset: dataset to evaluate
       OUTPUT : def_eval : dataset containing 'feature' (variable to separate), 'value' (separation value)
       'nature' (nature of the separation), 'cost' (cost of the separation)
       """
        
        df_eval = pd.DataFrame([], columns =('feature','value','nature','cost'))
        columns_to_feature = dataset.columns[dataset.columns != self.target]
        
        
        for column in columns_to_feature : 
            vect_type_value = [] #Protective vector to verify the type unicity of value in a feature
            for values in dataset[column]:
                vect_type_value = self.__value_type__(values, vect_type_value)
            vect_type_value=pd.Series(vect_type_value)
            
            if len(vect_type_value.unique()) > 1 :
                print('Check your data - Unicity is not respected for the feature : {}'.format(column))
                sys.exit()
                
            elif len(vect_type_value.unique()) == 1 and vect_type_value.unique() == 'str' :
                df_eval = df_eval.append(self.__test_quali__(dataset, column))
                
            elif len(vect_type_value.unique()) == 1 and vect_type_value.unique() == 'int' :
                df_eval = df_eval.append(self.__test_quanti__(dataset, column))
                
            else :
                print('Check your data - Unaccepted type of date on a feature : {} - Must be Integer or String'.format(column))
                sys.exit()           
            
        df_eval = df_eval.reset_index(drop=True)

        idx_cost_min = df_eval['cost'].idxmin(axis=0, skipna=True)

        return df_eval.iloc[idx_cost_min, :]           
            
    
    
    def create_leaf(self, dataset):
        """ Creating a leaf 
        INPUT 
       - dataset : dataset of the leaf to build
        OUTPUT 
       - leaf : the leaf class created with the information of our dataset
       """
   
        labels = dataset[self.target]
        pop = labels.shape[0]
        class_dict = list(map(lambda x : {x:1}, labels))
        class_dict_sum = reduce(self.__add_dict__, class_dict, {'':0})
        class_dict_sum.pop('') # Remove the initial and protective value
        prediction = max( class_dict_sum.items(), key = operator.itemgetter(1))[0] #Meme si la leaf n'est pas pure, il faut la catégoriser avec une prédiction
        proba = {k : v/pop for k,v  in class_dict_sum.items()}
        
        return leaf(dataset, pop, class_dict_sum, prediction, proba)
    
    
    
    def training(self, dataset, depth=0):
        """This function will build the decision tree according to the 
        parameters provided at the initialization of this class.
        INPUT 
        - depth : current depth of the tree
        OUTPUT 
        - node : root of the tree
        """
        
        #Check if a plit is still possible according to the number of values by feature
        no_more_split = True
        columns = dataset.columns[dataset.columns != self.target]
        for column in columns :
            if len(dataset[column].unique())>1 :
                no_more_split = False
                
                   
        # If the dataset is pure, or the maximum depth is reached or that the dataset can no longer be separated we create a leaf
        if len(dataset[self.target].unique())==1 or depth ==self.max_depth or no_more_split :
            return self.create_leaf(dataset)
            
        
        #find the best split if a split is possible and we're not in a pure leaf
        split_eval = self.__find_best_split__(dataset)
               
        # If the cost of the separation obtained after the separation is worse than the current one. Then there is no point in going any further
        if split_eval['cost'] >= self.__gini__(dataset): # All the population used to calculate the cost of the new separation is present at the previous node so the Gini = Cost
            return self.create_leaf(dataset)
            
               
        #Once the separation is validated by the previous points, the separation is performed via the identified feature
        if split_eval['nature'] == 'quali' : 
            left_branch, right_branch = self.__quali_split__(split_eval['feature'],split_eval['value'],dataset)
            
        elif split_eval['nature'] == 'quanti' : 
            left_branch, right_branch = self.__quanti_split__(split_eval['feature'],split_eval['value'],dataset)
            
        #Recursive training of the left branch
        left_node = self.training(left_branch, depth+1)
        

        #Recursive training of the right branch
        right_node = self.training(right_branch, depth+1)
        

        # Return the root of the tree
        return node(split_eval['feature'], 
                    split_eval['value'], 
                    split_eval['cost'], 
                    split_eval['nature'],
                    left_node,
                    right_node,
                    depth,
                    dataset.shape[0])


    def fit(self):
        """ Model drive
        OUTPUT 
        - node: root of the tree
        """
        return self.training(self.dataframe)


# %% [markdown]
# Node Class
# 
# This class will carry the characteristics of a branch, which are the heart of our algorithm 

# %%
                          
class node :
    """This class is intended to represent the branches of our classification tree.
   classification tree.
   """  
    def __init__(self, feature, value, cost, nature, left_branch, right_branch, depth, pop):
        self.feature = feature
        self.value = value
        self.cost = cost
        self.nature = nature
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.depth = depth
        self.pop = pop
    
    def __split__(self):
        if self.nature == 'quanti' :
            return self.feature + '<=' + str(self.value)
        if self.nature == 'quali' : 
            return self.feature + '==' + str(self.value)

    def make_pred(self, datatest):
        if self.nature  == 'quali':
            if datatest[self.feature][0] == self.value :
                    self.left_branch.make_pred(datatest)
            else : self.right_branch.make_pred(datatest)

        elif self.nature  == 'quanti':
            if datatest[self.feature][0] <= self.value :
                    self.left_branch.make_pred(datatest)
            else : self.right_branch.make_pred(datatest)


# %% [markdown]
# LEAVES class
# 
# This class will carry the characteristics of a leaf

# %%
class leaf :
    """This class is intended to represent the leaves of our classification tree.
    classification tree.
    """
    def __init__(self, dataset, pop, class_dict_sum, prediction, proba) :
        self.dataset = dataset
        self.pop = pop
        self.class_dict_sum = class_dict_sum
        self.prediction = prediction
        self.proba = proba
 
    def make_pred(self, datatest):
        print('The classification is : {}'.format(self.prediction))
     

# %%
def print_tree(node, spacing=" "):
    """ Display of the decision tree
    INPUT 
    - node : branch to display
    - spacings : space to display according to the depth of the branch
    """

  # Leaf display
    if isinstance(node, leaf):
        print(spacing + "Predict", node.prediction)
        print(spacing + "Predict", node.proba)
        return

  # Splitting condition display
    print(spacing + node.__split__())

  # Condition == True
    print(spacing + '--> True:')
    print_tree(node.left_branch, spacing + "    ")

  # Condition == False
    print(spacing + '--> False:')
    print_tree(node.right_branch, spacing + "    ")




# %%
#En raison de changement à venir dans les fonctionnalités, des warning sont levés. Nous ne voulons pas les voir
import warnings

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

    
tree_classif = decision_tree_classifier('Classification', df, 4)
tree_trained = tree_classif.fit()
print_tree(tree_trained)



# %%
datatest = pd.DataFrame([['crevette', 4, 0, 'Yes', 'No', 'No']], 
                        columns=['Espèce','Nb antennes','Nb pattes régulier','céphalo-thorax','paires pattes', 'ailes'])

# %%
tree_trained.make_pred(datatest)
# %%
