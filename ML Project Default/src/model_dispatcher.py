# -*- coding: utf-8 -*-
# model_dispatcher.py
"""
Created on Mon Jul 19 16:33:08 2021

@author: rchauhan32

this is used to parameterize the model by passing it 
from outside the code to our training script

"""

from sklearn import tree
from sklearn import ensemble

models = {
    
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),
    "rf": ensemble.RandomForestClassifier()
    }

