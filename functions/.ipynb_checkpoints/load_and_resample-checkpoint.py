import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_score, recall_score, classification_report, f1_score
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.utils import resample

def resample_and_save(raw_data):
    '''Loads the raw data, and resamples so that there are equal 'HeartDiseaseorAttack'==0 and ==1 rows 
    It does this by reducing 'HeartDiseaseorAttack'==0 rows
    
    Input: filepath for the raw data
    Output: Resampled df
    
    '''
    heart_df = pd.read_csv(raw_data)
    train = heart_df.copy()
    
    heart_issues = train[train["HeartDiseaseorAttack"] == 1]
    no_heart_issues = train[train["HeartDiseaseorAttack"] == 0]
    
    undersampled_heart_df =  resample(no_heart_issues,
                             replace=False, 
                             n_samples = len(heart_issues),
                             random_state=0)
    
    resampled_df = pd.concat([undersampled_heart_df, heart_issues])
    resampled_df.to_csv('../data/clean/resampled_data.csv', index=False)
    return resampled_df