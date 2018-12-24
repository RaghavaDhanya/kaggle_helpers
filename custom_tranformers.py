from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split

class FeatureExtractor(TransformerMixin):
    def __init__(self,columns):
        self.columns=columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

class ModOneHotEncoder(TransformerMixin):
    def __init__(self):
        self.ohe = OneHotEncoder(handle_unknown='ignore',sparse=False)
    
    def fit(self, X, y=None):
        self.__column_names=X.columns
        self.ohe.fit(X,y)
        return self
    def transform(self, X):
        ndarr=self.ohe.transform(X)
        return pd.DataFrame(data=ndarr,columns=self.ohe.get_feature_names(self.__column_names))
       
        

data=pd.read_csv("train.csv")
# data=FeatureExtractor(["Sex","Pclass"]).fit_transform(data)
# print(data.head())
# data=ModOneHotEncoder().fit_transform(data)
# print(data.head())

categoricals = ["Pclass", "Sex", "Embarked"]
numerics = ["Age", "Fare"]

pipeline=make_pipeline(
    FeatureUnion([
        ("numeric", make_pipeline(
            FeatureExtractor(numerics),
            SimpleImputer(missing_values=np.nan, strategy="mean"),
            StandardScaler()
        )),
        ("categoricals",make_pipeline(
            FeatureExtractor(categoricals),
            SimpleImputer(missing_values=np.nan, strategy="most_frequent"),
            OneHotEncoder(handle_unknown='ignore', sparse=False)
        ))
    ]),
    LR()
)
X_train, X_test, y_train, y_test = train_test_split(data.drop(["Survived"], axis=1), data["Survived"], test_size=0.33, random_state=33)
pipeline.fit(X_train,y_train)
print(pipeline.score(X_test,y_test))
