import pandas as pd
import numpy as np



def clean_data():
    data=pd.read_csv('data/archive/Algerian_forest_fires_dataset.csv')

    data.replace(['Na', '-',''], np.nan, inplace=True)
    data.dropna(inplace=True)
    data.columns = data.columns.str.strip()
    data['Classes'] = data['Classes'].str.strip()

    # i used binary assymitric encoding
    data['Classes'] = data['Classes'].map({'not fire': 0, 'fire': 1})   
    cols = data.columns.to_list()                      
    data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')

    data = data[(data['Classes'] == 0) | (data['Classes'] == 1)]

    data.drop(columns=['day'],inplace=True)
    data.dropna(inplace=True)                                            
    data.reset_index(drop=True, inplace=True)

    print(data.isnull().sum())
    print(data.isna().sum())

    return data      
                                 
data=clean_data()
print(data.head(5))

