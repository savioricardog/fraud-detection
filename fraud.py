#%% [markdown]
# # --- Imports ---
#%%
import kagglehub
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engine import discretisation, encoding
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.metrics import precision_score, recall_score, roc_auc_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier, RandomForestClassifier
# from sklearn.compose import 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sys.path.append(os.path.abspath(os.path.join('..')))

#%% [markdown]
# ## -- CONFIGURING JUPYTER PAGE --
#%%
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 500)

%load_ext autoreload
%reload_ext autoreload
%autoreload 2

#%% [markdown]
# ## -- DOWNLOAD DATASET LATEST VERSION --
#%%
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Path to dataset files:", path)

#%% [markdown]
# ## -- VARIABLES CONFIG --
#%%
path_dataset = r'C:\Users\savio\.cache\kagglehub\datasets\mlg-ulb\creditcardfraud\versions\3'

#%% [markdown]
# # -- READ AND SAMPLE DATASET --
#%%
df = pd.read_csv(f'{path_dataset}\creditcard.csv')
df.head(3)

#%% [markdown]
# # --- UNDERSTANDING DATASET ---
#%%
#%% [markdown]
# ## -- FIRST IMPRESSIONS --
#%%
print(df.info())
print(f'\n Shape df: {df.shape}')
