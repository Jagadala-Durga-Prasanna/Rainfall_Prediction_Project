# Libraries required
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
from sklearn import linear_model
from sklearn import ensemble
from sklearn import tree
from sklearn.svm import SVC

import xgboost as xgb

# Load dataset
data = pd.read_csv("rainfall_india.csv")

# Show first 5 rows
print(data.head())

print(data.describe())


# Dataset info
print(data.info())

data.shape

data.isnull().sum()

import missingno as msno
import matplotlib.pyplot as plt
msno.matrix(data,color=(0.55, 0.255, 0.225), fontsize=16)
msno.matrix(data)

data = data.dropna(thresh=len(data)*0.8, axis=1)

data['JAN'].fillna(data['JAN'].mean(), inplace=True)
data['FEB'].fillna(data['FEB'].mean(), inplace=True)
data['MAR'].fillna(data['MAR'].mean(), inplace=True)
data['APR'].fillna(data['APR'].mean(), inplace=True)
data['MAY'].fillna(data['MAY'].mean(), inplace=True)
data['JUN'].fillna(data['JUN'].mean(), inplace=True)
data['JUL'].fillna(data['JUL'].mean(), inplace=True)
data['AUG'].fillna(data['AUG'].mean(), inplace=True)
data['SEP'].fillna(data['SEP'].mean(), inplace=True)
data['OCT'].fillna(data['OCT'].mean(), inplace=True)
data['NOV'].fillna(data['NOV'].mean(), inplace=True)
data['DEC'].fillna(data['DEC'].mean(), inplace=True)
data['ANNUAL'].fillna(data['ANNUAL'].mean(), inplace=True)

state_names = ['SUBDIVISION']

from sklearn.impute import SimpleImputer
import numpy as np

imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

data_state = imp_mode.fit_transform(data[state_names])

data_state = pd.DataFrame(data_state, columns=state_names)

# Update the original column with imputed values to avoid duplicate columns
data['SUBDIVISION'] = data_state['SUBDIVISION'].values

data.corr(numeric_only=True)

cor=data.corr(numeric_only=True)

sns.heatmap(data=cor,xticklabels=cor.columns.values,yticklabels=cor.columns.values)
plt.show()

sns.pairplot(data.select_dtypes(include='number'))
plt.show()

data.boxplot()
plt.show()

sns.jointplot(
        x=data['YEAR'],
         y=data['ANNUAL'],
          kind='scatter',
           height=7
           )
plt.show()

sns.jointplot(
        x="JJAS",
         y="MAM",
          data=data,
           hue="YEAR"
           )
plt.show()

sns.histplot(data['ANNUAL'])
plt.show()

sns.scatterplot(x=data['YEAR'], y=data['ANNUAL'])
plt.show()

sns.displot(data['JJAS'], kde=True)
plt.show()

# Splitting X and y
y = data['ANNUAL']
X = data.drop(columns=['ANNUAL', 'SUBDIVISION'])

from sklearn.preprocessing import StandardScaler

# Target
y = data['ANNUAL']

X = data.drop(columns=['ANNUAL', 'SUBDIVISION'])

# Keep only numeric columns
X = X.select_dtypes(include=['int64', 'float64'])

y = data['ANNUAL']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)


# Fix missing values safely
data.fillna(data.mean(numeric_only=True))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("rainfall_india.csv")

# Fill missing values
data = data.fillna(data.mean(numeric_only=True))

# Features & target
X = data.drop(columns=['ANNUAL', 'SUBDIVISION'])
y = data['ANNUAL']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize models
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)

# Train models
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predict
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# Accuracy (R2 score)
print("Linear Regression R2:", r2_score(y_test, lr_pred))
print("Random Forest R2:", r2_score(y_test, rf_pred))


y_pred = rf.predict(X_test)
threshold = y_test.mean()

y_test_bin = (y_test >= threshold).astype(int)
y_pred_bin = (y_pred >= threshold).astype(int)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test_bin, y_pred_bin)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


y_pred = rf.predict(X_test)
import numpy as np

threshold = y_test.mean()

y_test_bin = (y_test >= threshold).astype(int)
y_pred_bin = (y_pred >= threshold).astype(int)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_test_bin, y_pred_bin)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve')
plt.legend(loc="lower right")
plt.show()

import pickle

model = rf   # choose best model (Random Forest)

# model saving
pickle.dump(model, open('rainfall.pkl', 'wb'))

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# example: encoding target column
data['ANNUAL'] = le.fit_transform(data['ANNUAL'])

import pickle

# encoder saving
pickle.dump(le, open('encoder.pkl', 'wb'))

# imputer saving
pickle.dump(imp_mode, open('imputer.pkl', 'wb'))

# scaling the data
pickle.dump(sc, open('scale.pkl', 'wb'))


