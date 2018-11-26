# Import all the Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
empPerfDF = pd.read_excel("Employee_Perf_Analysis/Hr_data.xls")
print(type(empPerfDF[['PerformanceRating']]))
print(empPerfDF.columns)
print(empPerfDF.dtypes)
# empPerfDF_Desc = pd.DataFrame(empPerfDF.describe())
objTypeCols = empPerfDF[[i for i in empPerfDF.columns if empPerfDF[i].dtype == 'object']]; del(i)

# Adding missing values intentionally
# empPerfDF.loc[np.random.choice(len(empPerfDF), size=200, replace=False), 'EmpLastSalaryHikePercent'] = np.nan # Inserting NaN values for analysis
# empPerfDF['EmpLastSalaryHikePercent'].isnull().value_counts()
# empPerfDF['EmpLastSalaryHikePercent'].isnull().sum()

# Imputing these missing values
# from sklearn.preprocessing import Imputer
# imputeMissing = Imputer() # When we do "fit", it calculates mean of each column. 
# imputeMissing.fit_transform(empPerfDF[['EmpLastSalaryHikePercent']]) # When we do "transform", it imputes
# empPerfDF['EmpLastSalaryHikePercent'].mean()
# When we use the same transform on test data, it will use training info to impute the test data

# Another Approach - Adding boolean col which represents original value as missing or not along with imputed column
# cols_w_missing = (col for col in empPerfDF.columns if empPerfDF[col].isnull().any()) # Generator Comprehension
# for col in cols_w_missing:
#     empPerfDF[col+"_missing"] = empPerfDF[col].isnull() # Adding new cols that contains T if missing
# empPerfDF['EmpLastSalaryHikePercent_missing']

#
corrEmpPerfDF = empPerfDF.corr()
print(corrEmpPerfDF)
corrEmpPerfDF['PerformanceRating'].sort_values(ascending=False)
corrEmpPerfDF['PerformanceRating'].sort_values(ascending=False).index[:-4:-1]
# EmpEnvironmentSatisfaction, EmpLastSalaryHikePercent is having high Corr with PerformanceRating
empPerfDF[corrEmpPerfDF['PerformanceRating'].sort_values(ascending=False).index[0:4]].head()
empPerfDF[corrEmpPerfDF['PerformanceRating'].sort_values(ascending=False).index[[0,-1,-2,-3]]].head()

# Dropping un-necessary information

# Group by Department and find the which Department has the Highest Performance rating
empPerfDF.groupby("EmpDepartment")['PerformanceRating'].mean()
"""
Obeservations from above line:
1)Performance Rating for employees in the Development department is Highest.
2)Performance rating for employess in the Sales department is lowest
3)Performance Rating for employees in the Data Science department is next to Development team
4)SalaryHike for the employee in the Data Science department is highest.
"""

# Check for Unique Departments in the Dataset
empPerfDF['EmpDepartment'].value_counts()

# Find the Number of Employees who have rating of 2,3,4
empPerfDF["PerformanceRating"].value_counts()

# Converting data type to 'category' - encoding

# Label Encoding
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# le = LabelEncoder() # LabelEncoder - Code categories into 0,1,2.....
# for i in objTypeCols.columns.values.tolist()[1:]:
#     empPerfDF[i+"_coded"] = empPerfDF[i].astype('category')
#     empPerfDF[i+"_coded"] = le.fit_transform(empPerfDF[i+"_coded"])

# One Hot Encoding
for i in objTypeCols.columns.values.tolist()[1:]:
    dummyCols = pd.get_dummies(empPerfDF[i], prefix=i) # Convert 1/0 based on presence; C no. of columns
    empPerfDF = empPerfDF.join(dummyCols)
del dummyCols,i
# ohe = OneHotEncoder()
# tmp = OneHotEncoder(categorical_features=le.fit_transform(empPerfDF['MaritalStatus']))

# Model building
from sklearn.model_selection import train_test_split
train, test = train_test_split(empPerfDF, test_size=0.3, random_state=123, stratify=empPerfDF["PerformanceRating"])

# ['EmpNumber', u'Gender', u'EducationBackground', u'MaritalStatus',
#       u'EmpDepartment', u'EmpJobRole', u'BusinessTravelFrequency',
#       u'OverTime', u'Attrition'],

######################################### Decision Tree ###############################################
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion="entropy", max_depth=8, min_samples_split=20, random_state=99)
features = [col for col in empPerfDF.columns if col not in objTypeCols.columns if col != "PerformanceRating"]
dtc.fit(train[features], train["PerformanceRating"])
preds = dtc.predict(test[features])
dtc.score(test[features], test["PerformanceRating"]) # Accuracy
# Comparing actual and predicted values using CrossTab function
pd.crosstab(test["PerformanceRating"], preds, rownames=['Actual'], colnames=['Predictions'])
dtc.feature_importances_
tmp = pd.DataFrame(zip(features, dtc.feature_importances_), columns = ["Feature","Importance"])
tmp = tmp.sort_values(by='Importance', ascending=False)

## Applying k-fold cross validation
from sklearn.cross_validation import KFold
crossvalidation = KFold(n=train[features].shape[0], n_folds=10, shuffle=True, random_state=12)
# Finding Accuracy using K-Fold
from sklearn.cross_validation import cross_val_score
score = np.mean(cross_val_score(dtc,train[features],train['PerformanceRating'],scoring='accuracy',cv=crossvalidation,n_jobs=1))
print(score)

# Plotting the Decision Tree
from sklearn.externals.six import StringIO  
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dtc, out_file="Employee_Perf_Analysis/decisionTree.dot", feature_names=features,
                filled=True, rounded=True, special_characters=True)
# graph = pydotplus.graph_from_dot_data("dt.dot")
# graph.write_jpg("dtree2.jpg")

######################################## Random Forest ############################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
rfc = RandomForestClassifier(n_estimators=300, random_state=123)
rfc.fit(train[features], train["PerformanceRating"])
preds = rfc.predict(test[features])
# pd.crosstab(test["PerformanceRating"], preds, rownames=['Actual'], colnames=['Predictions'])
print(confusion_matrix(test["PerformanceRating"], preds))
print(accuracy_score(test["PerformanceRating"], preds))

# Feature importance
featImp = pd.DataFrame(data=rfc.feature_importances_*100.0, columns=["GiniValue"])
featImp.index = features
featImp.sort_values(['GiniValue'], axis=0, ascending=False, inplace=True)
print(featImp.head())

"""
From above lines we can infer that the Feature which has the Highest Percentage  affects the Performance Rating.
1)EmpLastSalaryHikePercent---21.24%
2)EmpEnvirnmentSatisfaction---20.21%
3)YearsSinceLastPromotion---10.06%  
are the three factors affecting the Performance Rating

Obeservations from the above line:
1) Employee who has the Highest Performance rating has more Environment Satisfaction and Employee 
who has the low arting has less Environment Satisfaction
2) Employee who has Highest performance rating has  higest Salary hike Percentage
3) Employee who has Lowest Performance rating is the Current role for more number of years,
employee who has the highest Performance rating has lowest number of Years in the Current Role. 
This implies that there is no career growth 
"""  

# Feature Selection
from sklearn.feature_selection import RFECV
clf_rf_4 = RandomForestClassifier()
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5, scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(train[features], train["PerformanceRating"])
print('Optimal number of features :', rfecv.n_features_)
print('Best features :', train[features].columns[rfecv.support_])