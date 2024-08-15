from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from DecisionTree import DecisionTreeClassify, DecisionTreeRegression
from Utility_functions import TrainTestSplit
import numpy as np
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_squared_error,mean_absolute_error,r2_score

import sys

if(len(sys.argv)!=2):
    raise ValueError("Number of argument should be one and equal to the model you wanna test...")

choice=sys.argv[1]

if choice=="multiclass":
    iris_df = pd.read_csv('Decision_Tree/iris.csv')  # Read Iris dataset from iris.csv
    X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values

    # Encode the target variable 'species' into numerical labels
    class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    y = iris_df['species'].map(class_mapping).values

    splitter=TrainTestSplit(test_size=0.5, random_state=42)
    X_train, X_test, y_train, y_test = splitter.split(X, y)
    
    for i in [5,8,10]:
      for j in ['gini','entropy']:
        clf = DecisionTreeClassifier(max_depth=i,criterion=j)
        clf.fit(X_train, y_train)
        y_predsk = clf.predict(X_test)

        clf1 = DecisionTreeClassify(max_depth=i,loss=j)
        clf1.train(X_train, y_train)
        y_predmy = clf1.predict(X_test)
        

        accuracysk = accuracy_score(y_test, y_predsk)
        accuracymy = accuracy_score(y_test, y_predmy)
        psk = precision_score(y_test, y_predsk,average='macro')
        pmy = precision_score(y_test, y_predmy,average='macro')
        rsk = recall_score(y_test, y_predsk,average='macro')
        rmy = recall_score(y_test, y_predmy,average='macro')
        f1sk = f1_score(y_test, y_predsk,average='macro')
        f1my = f1_score(y_test, y_predmy,average='macro')

        print("SKlearn acc,pre,rec,f1 for depth ",i,'and criterion ',j,':', accuracysk,'|',psk,'|',rsk,'|',f1sk)
        print("FML LIBRARY acc,pre,rec,f1 for depth ",i,'and criterion ',j,':', accuracymy,'|',pmy,'|',rmy,'|',f1my)

elif choice=="binary":
    
    cell_df = pd.read_csv('Decision_Tree/cell_samples.csv')

    cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
    cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')

    feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
       'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]

    X = np.asarray(feature_df)
    y = np.asarray(cell_df['Class'])
    y = [-1 if t == 2 else 1 for t in y]
    y = np.array(y)

    splitter=TrainTestSplit(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = splitter.split(X, y)
    
    for i in [5,8,10]:
      for j in ['gini','entropy']:
        clf = DecisionTreeClassifier(max_depth=i,criterion=j)
        clf.fit(X_train, y_train)
        y_predsk = clf.predict(X_test)

        clf1 = DecisionTreeClassify(max_depth=i,loss=j)
        clf1.train(X_train, y_train)
        y_predmy = clf1.predict(X_test)
        

        accuracysk = accuracy_score(y_test, y_predsk)
        accuracymy = accuracy_score(y_test, y_predmy)
        psk = precision_score(y_test, y_predsk,average='macro')
        pmy = precision_score(y_test, y_predmy,average='macro')
        rsk = recall_score(y_test, y_predsk,average='macro')
        rmy = recall_score(y_test, y_predmy,average='macro')
        f1sk = f1_score(y_test, y_predsk,average='macro')
        f1my = f1_score(y_test, y_predmy,average='macro')

        print("SKlearn acc,pre,rec,f1 for depth ",i,'and criterion ',j,':', accuracysk,'|',psk,'|',rsk,'|',f1sk)
        print("FML LIBRARY acc,pre,rec,f1 for depth ",i,'and criterion ',j,':', accuracymy,'|',pmy,'|',rmy,'|',f1my)


    
elif choice=="regression":
    
    california_housing_dataset = fetch_california_housing()
    X = california_housing_dataset.data
    y = california_housing_dataset.target

    splitter=TrainTestSplit(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = splitter.split(X, y)

    # declare the regressor and train the model
    rgr = DecisionTreeRegression(max_depth=5,loss='mse')
    rgr.train(X_train,y_train)
    # make predictions
    yp = rgr.predict(X_test)
    print("rmse for FML LIBRARY regressor:",np.sqrt(mean_squared_error(y_test,yp)))
    print("mae for FML LIBRARY regressor:", mean_absolute_error(y_test,yp))
    print("r2 for FML LIBRARY regressor:" ,r2_score(y_test,yp))
    rgr = DecisionTreeRegressor(max_depth=5,criterion='squared_error')
    rgr.fit(X_train,y_train)
    yp = rgr.predict(X_test)
    print("rmse For sklearn: ", np.sqrt(mean_squared_error(y_test,yp)))
    print("mae For sklearn: ", mean_absolute_error(y_test,yp))
    print("r2 for sklearn: " ,r2_score(y_test,yp))
    


else:
    raise ValueError("Invalid choice. Please provide 'knn', 'lr', or 'regression'.")