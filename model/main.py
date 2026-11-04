import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

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
    data.drop(columns=['year'],inplace=True)
    data.dropna(inplace=True)                                            
    data.reset_index(drop=True, inplace=True)

   
    
    return data      


def main():

    data=clean_data()
     
    y=np.array(data['Classes'])
    x=np.array(data.drop(columns='Classes'))

    scaler=StandardScaler()
    x_scaled=scaler.fit_transform(x)


    x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,train_size=0.8,test_size=0.2,random_state=50,stratify=y)

    model=LogisticRegression()
    model.fit(x_train,y_train)

   
   
    with open('model/model.pkl','wb') as f:
        pickle.dump(model,f)

    with open('model/scaler.pkl','wb') as f:
            pickle.dump(scaler,f)
    
if __name__ == "__main__":
    main()