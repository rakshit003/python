# -*- coding: utf-8 -*-
# train.py
"""
Created on Mon Jul 19 15:55:34 2021

@author: rchauhan32
This code consist of trainig the model with Train data
"""

import joblib
import pandas as pd

from sklearn import metrics

import config
import model_dispatcher

# fold represents the kfold to run the test on
def run(fold,model):
    #read data with folds
    df=pd.read_csv(config.TRAINING_FILE)
    #training data is where kfold is not equal to the provided fold
    # we reset the index to randomise
    
    df_train=df[df.kfold != fold].reset_index(drop=True)
    
    #validation is where kfold is equal to the provided fold

    df_valid= df[df.kfold==fold].reset_index(drop=True)
    
    #droping the target label in df and converting into numpy array
    #target is label column in dataframe
    
    #for training set
    x_train=df_train.drop("label",axis=1).values
    y_train=df_train.label.values
    
    # for validation set
    x_valid=df_valid.drop("label",axis=1).values
    y_valid=df_valid.label.values
    
    #initialize the model
    
    clf=model_dispatcher.models[args.model]
    
    #fit the model
    clf.fit(x_train,y_train)

    #create predictions for validation samples
    preds=clf.predict(x_valid)
    
    #calculate & print accuracy
    accuracy= metrics.accuracy_score(y_valid,preds)
    print(f"Fold={fold}, Accuracy={accuracy}")
    
    #save the model
    joblib.dump(clf, f"{config.MODEL_OUTPUT}/dt_{fold}.bin")
    
"""
somestimes its not advisable to run al the folds in the same script as the
memory consumption may keep increasing and program may crash.
To avoid it we can pass arguments to training script
"""

import argparse

if __name__== "__main__":
    #initialize ArgumentParser class of argparse
    parser= argparse.ArgumentParser()
    
    #add different argument you need and thier types
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    # read the aguments from the command line
    args= parser.parse_args()
    
    #run fold specified by command line
    run(
        fold=args.fold,
        model=args.model
        )
    
