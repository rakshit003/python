import pandas as pd
import config
from sklearn import model_selection

def Kfold(input):
    df=pd.read_csv(input)
    
    # initialize kfold column
    df["kfold"]=-1
    
    #randomaise rows of dataset
    df=df.sample(frac=1).reset_index(drop=True)
    
    #fetch targets, assuming name of he outcome variable = label
    y=df.label.values
    
    #initiate K fold from model_selection module
    kf= model_selection.StratifiedKFold(n_splits=5)
    
    for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_,'kfold']=f
    
    return df

data=Kfold(config.RAW_FILE_TRAIN)
data.to_csv(config.TRAINING_FILE,index=False)

# testing the fold distribution across output
for i in range(5):
    print(f"for kfold={i}\n{data[data['kfold']==i].label.value_counts()}")
     
