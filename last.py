import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #to plot charts
import seaborn as sns #used for data visualization
import warnings #avoid warning flash
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# # 2. Loading the dataset 

df=pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:07:41.877172Z","iopub.execute_input":"2021-11-19T07:07:41.877707Z","iopub.status.idle":"2021-11-19T07:07:41.910231Z","shell.execute_reply.started":"2021-11-19T07:07:41.877657Z","shell.execute_reply":"2021-11-19T07:07:41.908842Z"}}
df.head() #get familier with dataset, display the top 5 data records

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:07:41.912233Z","iopub.execute_input":"2021-11-19T07:07:41.912809Z","iopub.status.idle":"2021-11-19T07:07:41.920446Z","shell.execute_reply.started":"2021-11-19T07:07:41.912732Z","shell.execute_reply":"2021-11-19T07:07:41.919404Z"}}
df.shape #getting to know about rows and columns we're dealing with - 768 rows , 9 columns

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:07:41.922808Z","iopub.execute_input":"2021-11-19T07:07:41.924214Z","iopub.status.idle":"2021-11-19T07:07:41.939147Z","shell.execute_reply.started":"2021-11-19T07:07:41.924148Z","shell.execute_reply":"2021-11-19T07:07:41.938434Z"}}
df.columns #learning about the columns

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:07:41.940215Z","iopub.execute_input":"2021-11-19T07:07:41.94089Z","iopub.status.idle":"2021-11-19T07:07:41.956405Z","shell.execute_reply.started":"2021-11-19T07:07:41.940854Z","shell.execute_reply":"2021-11-19T07:07:41.955645Z"}}
df.dtypes #knowledge of data type helps for computation

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:07:41.957473Z","iopub.execute_input":"2021-11-19T07:07:41.957925Z","iopub.status.idle":"2021-11-19T07:07:41.998315Z","shell.execute_reply.started":"2021-11-19T07:07:41.957888Z","shell.execute_reply":"2021-11-19T07:07:41.997393Z"}}
df.info() #Print a concise summary of a DataFrame. This method prints information about a DataFrame including the index dtype and columns, non-null values and memory usage.

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:07:41.999977Z","iopub.execute_input":"2021-11-19T07:07:42.000739Z","iopub.status.idle":"2021-11-19T07:07:42.046717Z","shell.execute_reply.started":"2021-11-19T07:07:42.00068Z","shell.execute_reply":"2021-11-19T07:07:42.045851Z"}}
df.describe() #helps us to understand how data has been spread across the table.
df=df.drop_duplicates()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:07:42.062859Z","iopub.execute_input":"2021-11-19T07:07:42.063186Z","iopub.status.idle":"2021-11-19T07:07:42.084798Z","shell.execute_reply.started":"2021-11-19T07:07:42.063142Z","shell.execute_reply":"2021-11-19T07:07:42.084116Z"}}
#check for missing values, count them and print the sum for every column
df.isnull().sum() #conclusion :- there are no null values in this dataset

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:07:42.08607Z","iopub.execute_input":"2021-11-19T07:07:42.086465Z","iopub.status.idle":"2021-11-19T07:07:42.10278Z","shell.execute_reply.started":"2021-11-19T07:07:42.086419Z","shell.execute_reply":"2021-11-19T07:07:42.101846Z"}}
#checking for 0 values in 5 columns , Age & DiabetesPedigreeFunction do not have have minimum 0 value so no need to replace , also no. of pregnancies as 0 is possible as observed in df.describe
print(df[df['BloodPressure']==0].shape[0])
print(df[df['Glucose']==0].shape[0])
print(df[df['SkinThickness']==0].shape[0])
print(df[df['Insulin']==0].shape[0])
print(df[df['BMI']==0].shape[0])

df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())#normal distribution
df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())#normal distribution
df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].median())#skewed distribution
df['Insulin']=df['Insulin'].replace(0,df['Insulin'].median())#skewed distribution
df['BMI']=df['BMI'].replace(0,df['BMI'].median())#skewed distribution

sns.countplot('Outcome',data=df)

df.hist(bins=10,figsize=(10,10))
plt.show()

plt.figure(figsize=(16,12))
sns.set_style(style='whitegrid')
plt.subplot(3,3,1)
sns.boxplot(x='Glucose',data=df)
plt.subplot(3,3,2)
sns.boxplot(x='BloodPressure',data=df)
plt.subplot(3,3,3)
sns.boxplot(x='Insulin',data=df)
plt.subplot(3,3,4)
sns.boxplot(x='BMI',data=df)
plt.subplot(3,3,5)
sns.boxplot(x='Age',data=df)
plt.subplot(3,3,6)
sns.boxplot(x='SkinThickness',data=df)
plt.subplot(3,3,7)
sns.boxplot(x='Pregnancies',data=df)
plt.subplot(3,3,8)
sns.boxplot(x='DiabetesPedigreeFunction',data=df)

from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(20,20));

corrmat=df.corr()
sns.heatmap(corrmat, annot=True)

df_selected=df.drop(['BloodPressure','Insulin','DiabetesPedigreeFunction'],axis='columns')

from sklearn.preprocessing import QuantileTransformer
x=df_selected
quantile  = QuantileTransformer()
X = quantile.fit_transform(x)
df_new=quantile.transform(X)
df_new=pd.DataFrame(X)
df_new.columns =['Pregnancies', 'Glucose','SkinThickness','BMI','Age','Outcome']
df_new.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:07:53.388167Z","iopub.execute_input":"2021-11-19T07:07:53.389122Z","iopub.status.idle":"2021-11-19T07:07:54.35077Z","shell.execute_reply.started":"2021-11-19T07:07:53.389077Z","shell.execute_reply":"2021-11-19T07:07:54.349739Z"}}
plt.figure(figsize=(16,12))
sns.set_style(style='whitegrid')
plt.subplot(3,3,1)
sns.boxplot(x=df_new['Glucose'],data=df_new)
plt.subplot(3,3,2)
sns.boxplot(x=df_new['BMI'],data=df_new)
plt.subplot(3,3,3)
sns.boxplot(x=df_new['Pregnancies'],data=df_new)
plt.subplot(3,3,4)
sns.boxplot(x=df_new['Age'],data=df_new)
plt.subplot(3,3,5)
sns.boxplot(x=df_new['SkinThickness'],data=df_new)

# %% [markdown]
# # 5. Split the Data Frame into X and y

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:07:54.351998Z","iopub.execute_input":"2021-11-19T07:07:54.352238Z","iopub.status.idle":"2021-11-19T07:07:54.358646Z","shell.execute_reply.started":"2021-11-19T07:07:54.352209Z","shell.execute_reply":"2021-11-19T07:07:54.357611Z"}}
target_name='Outcome'
y= df_new[target_name]#given predictions - training data 
X=df_new.drop(target_name,axis=1)#dropping the Outcome column and keeping all other columns as X

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:07:54.360003Z","iopub.execute_input":"2021-11-19T07:07:54.360326Z","iopub.status.idle":"2021-11-19T07:07:54.37886Z","shell.execute_reply.started":"2021-11-19T07:07:54.360282Z","shell.execute_reply":"2021-11-19T07:07:54.377992Z"}}
X.head() # contains only independent features 

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:07:54.380478Z","iopub.execute_input":"2021-11-19T07:07:54.381102Z","iopub.status.idle":"2021-11-19T07:07:54.389789Z","shell.execute_reply.started":"2021-11-19T07:07:54.381067Z","shell.execute_reply":"2021-11-19T07:07:54.389125Z"}}
y.head() #contains dependent feature

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:07:54.391054Z","iopub.execute_input":"2021-11-19T07:07:54.39192Z","iopub.status.idle":"2021-11-19T07:07:54.460616Z","shell.execute_reply.started":"2021-11-19T07:07:54.391875Z","shell.execute_reply":"2021-11-19T07:07:54.459865Z"}}
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=0)#splitting data in 80% train, 20%test

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:07:54.461997Z","iopub.execute_input":"2021-11-19T07:07:54.462911Z","iopub.status.idle":"2021-11-19T07:07:54.469857Z","shell.execute_reply.started":"2021-11-19T07:07:54.462862Z","shell.execute_reply":"2021-11-19T07:07:54.469075Z"}}
X_train.shape,y_train.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:07:54.471138Z","iopub.execute_input":"2021-11-19T07:07:54.47181Z","iopub.status.idle":"2021-11-19T07:07:54.48454Z","shell.execute_reply.started":"2021-11-19T07:07:54.471746Z","shell.execute_reply":"2021-11-19T07:07:54.483635Z"}}
X_test.shape,y_test.shape


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:14:53.265694Z","iopub.execute_input":"2021-11-19T07:14:53.266289Z","iopub.status.idle":"2021-11-19T07:14:53.276062Z","shell.execute_reply.started":"2021-11-19T07:14:53.266249Z","shell.execute_reply":"2021-11-19T07:14:53.275107Z"}}
#List Hyperparameters to tune
knn= KNeighborsClassifier()
n_neighbors = list(range(15,25))
p=[1,2]
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']

#convert to dictionary
hyperparameters = dict(n_neighbors=n_neighbors, p=p,weights=weights,metric=metric)

#Making model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=knn, param_grid=hyperparameters, n_jobs=-1, cv=cv, scoring='f1',error_score=0)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:14:57.718579Z","iopub.execute_input":"2021-11-19T07:14:57.719065Z","iopub.status.idle":"2021-11-19T07:15:12.201904Z","shell.execute_reply.started":"2021-11-19T07:14:57.719022Z","shell.execute_reply":"2021-11-19T07:15:12.201038Z"}}
best_model = grid_search.fit(X_train,y_train)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:16:43.91972Z","iopub.execute_input":"2021-11-19T07:16:43.920855Z","iopub.status.idle":"2021-11-19T07:16:43.92984Z","shell.execute_reply.started":"2021-11-19T07:16:43.9208Z","shell.execute_reply":"2021-11-19T07:16:43.928849Z"}}
#Best Hyperparameters Value
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:17:12.200627Z","iopub.execute_input":"2021-11-19T07:17:12.20129Z","iopub.status.idle":"2021-11-19T07:17:12.210142Z","shell.execute_reply.started":"2021-11-19T07:17:12.201236Z","shell.execute_reply":"2021-11-19T07:17:12.209164Z"}}
#Predict testing set
knn_pred = best_model.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:17:15.014073Z","iopub.execute_input":"2021-11-19T07:17:15.015028Z","iopub.status.idle":"2021-11-19T07:17:15.320389Z","shell.execute_reply.started":"2021-11-19T07:17:15.01497Z","shell.execute_reply":"2021-11-19T07:17:15.319681Z"}}
print("Classification Report is:\n",classification_report(y_test,knn_pred))
print("\n F1:\n",f1_score(y_test,knn_pred))
print("\n Precision score is:\n",precision_score(y_test,knn_pred))
print("\n Recall score is:\n",recall_score(y_test,knn_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test,knn_pred))

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

param_grid_nb = {
    'var_smoothing': np.logspace(0,-2, num=100)
}
nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:18:38.084462Z","iopub.execute_input":"2021-11-19T07:18:38.085055Z","iopub.status.idle":"2021-11-19T07:18:40.127789Z","shell.execute_reply.started":"2021-11-19T07:18:38.085008Z","shell.execute_reply":"2021-11-19T07:18:40.126873Z"}}
best_model= nbModel_grid.fit(X_train, y_train)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:19:08.296504Z","iopub.execute_input":"2021-11-19T07:19:08.297142Z","iopub.status.idle":"2021-11-19T07:19:08.305101Z","shell.execute_reply.started":"2021-11-19T07:19:08.297094Z","shell.execute_reply":"2021-11-19T07:19:08.303598Z"}}
nb_pred=best_model.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:19:16.771322Z","iopub.execute_input":"2021-11-19T07:19:16.772402Z","iopub.status.idle":"2021-11-19T07:19:17.089663Z","shell.execute_reply.started":"2021-11-19T07:19:16.772337Z","shell.execute_reply":"2021-11-19T07:19:17.088636Z"}}
print("Classification Report is:\n",classification_report(y_test,nb_pred))
print("\n F1:\n",f1_score(y_test,nb_pred))
print("\n Precision score is:\n",precision_score(y_test,nb_pred))
print("\n Recall score is:\n",recall_score(y_test,nb_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test,nb_pred))

# %% [markdown]
# # 9.3 Support Vector Machine :- 
# 
# It is typically leveraged for classification problems, constructing a hyperplane where the distance between two classes of data points is at its maximum. This hyperplane is known as the decision boundary, separating the classes of data points (e.g., has diabetes vs doesn't have diabetes ) on either side of the plane.

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:19:33.268543Z","iopub.execute_input":"2021-11-19T07:19:33.268918Z","iopub.status.idle":"2021-11-19T07:19:33.27492Z","shell.execute_reply.started":"2021-11-19T07:19:33.268876Z","shell.execute_reply":"2021-11-19T07:19:33.273827Z"}}
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:19:35.700692Z","iopub.execute_input":"2021-11-19T07:19:35.701225Z","iopub.status.idle":"2021-11-19T07:19:35.707651Z","shell.execute_reply.started":"2021-11-19T07:19:35.701177Z","shell.execute_reply":"2021-11-19T07:19:35.706638Z"}}
model = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:21:15.9747Z","iopub.execute_input":"2021-11-19T07:21:15.975723Z","iopub.status.idle":"2021-11-19T07:21:15.982101Z","shell.execute_reply.started":"2021-11-19T07:21:15.97567Z","shell.execute_reply":"2021-11-19T07:21:15.980877Z"}}
# define grid search
grid = dict(kernel=kernel,C=C,gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='f1',error_score=0)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:21:18.799672Z","iopub.execute_input":"2021-11-19T07:21:18.800821Z","iopub.status.idle":"2021-11-19T07:22:08.442378Z","shell.execute_reply.started":"2021-11-19T07:21:18.800735Z","shell.execute_reply":"2021-11-19T07:22:08.441642Z"}}
grid_result = grid_search.fit(X, y)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:23:39.401694Z","iopub.execute_input":"2021-11-19T07:23:39.402787Z","iopub.status.idle":"2021-11-19T07:23:39.411654Z","shell.execute_reply.started":"2021-11-19T07:23:39.402726Z","shell.execute_reply":"2021-11-19T07:23:39.410649Z"}}
svm_pred=grid_result.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:23:41.455686Z","iopub.execute_input":"2021-11-19T07:23:41.456108Z","iopub.status.idle":"2021-11-19T07:23:41.778684Z","shell.execute_reply.started":"2021-11-19T07:23:41.456066Z","shell.execute_reply":"2021-11-19T07:23:41.776982Z"}}
print("Classification Report is:\n",classification_report(y_test,svm_pred))
print("\n F1:\n",f1_score(y_test,knn_pred))
print("\n Precision score is:\n",precision_score(y_test,knn_pred))
print("\n Recall score is:\n",recall_score(y_test,knn_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test,svm_pred))

# %% [markdown]
# ## 9.4 Decision Tree 

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:24:01.875405Z","iopub.execute_input":"2021-11-19T07:24:01.875746Z","iopub.status.idle":"2021-11-19T07:24:01.915828Z","shell.execute_reply.started":"2021-11-19T07:24:01.875709Z","shell.execute_reply":"2021-11-19T07:24:01.915028Z"}}
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
dt = DecisionTreeClassifier(random_state=42)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:24:04.459827Z","iopub.execute_input":"2021-11-19T07:24:04.46042Z","iopub.status.idle":"2021-11-19T07:24:04.465211Z","shell.execute_reply.started":"2021-11-19T07:24:04.460377Z","shell.execute_reply":"2021-11-19T07:24:04.464295Z"}}
# Create the parameter grid based on the results of random search 
params = {
    'max_depth': [5, 10, 20,25],
    'min_samples_leaf': [10, 20, 50, 100,120],
    'criterion': ["gini", "entropy"]
}

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:24:12.415304Z","iopub.execute_input":"2021-11-19T07:24:12.415673Z","iopub.status.idle":"2021-11-19T07:24:12.420732Z","shell.execute_reply.started":"2021-11-19T07:24:12.415614Z","shell.execute_reply":"2021-11-19T07:24:12.419861Z"}}
grid_search = GridSearchCV(estimator=dt, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:24:19.05102Z","iopub.execute_input":"2021-11-19T07:24:19.051517Z","iopub.status.idle":"2021-11-19T07:24:19.594366Z","shell.execute_reply.started":"2021-11-19T07:24:19.051478Z","shell.execute_reply":"2021-11-19T07:24:19.593738Z"}}
best_model=grid_search.fit(X_train, y_train)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:24:22.616388Z","iopub.execute_input":"2021-11-19T07:24:22.616785Z","iopub.status.idle":"2021-11-19T07:24:22.623391Z","shell.execute_reply.started":"2021-11-19T07:24:22.616714Z","shell.execute_reply":"2021-11-19T07:24:22.622319Z"}}
dt_pred=best_model.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:24:25.131731Z","iopub.execute_input":"2021-11-19T07:24:25.133007Z","iopub.status.idle":"2021-11-19T07:24:25.412586Z","shell.execute_reply.started":"2021-11-19T07:24:25.132951Z","shell.execute_reply":"2021-11-19T07:24:25.41162Z"}}
print("Classification Report is:\n",classification_report(y_test,dt_pred))
print("\n F1:\n",f1_score(y_test,dt_pred))
print("\n Precision score is:\n",precision_score(y_test,dt_pred))
print("\n Recall score is:\n",recall_score(y_test,dt_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test,dt_pred))

# %% [markdown]
# ## 9.5 Random Forest :- 
# The "forest" references a collection of uncorrelated decision trees, which are then merged together to reduce variance and create more accurate data predictions.

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:24:36.503507Z","iopub.execute_input":"2021-11-19T07:24:36.504038Z","iopub.status.idle":"2021-11-19T07:24:36.538711Z","shell.execute_reply.started":"2021-11-19T07:24:36.504004Z","shell.execute_reply":"2021-11-19T07:24:36.537983Z"}}
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:24:40.051992Z","iopub.execute_input":"2021-11-19T07:24:40.052377Z","iopub.status.idle":"2021-11-19T07:24:40.057792Z","shell.execute_reply.started":"2021-11-19T07:24:40.052266Z","shell.execute_reply":"2021-11-19T07:24:40.056865Z"}}
# define models and parameters
model = RandomForestClassifier()
n_estimators = [1800]
max_features = ['sqrt', 'log2']

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:24:42.504541Z","iopub.execute_input":"2021-11-19T07:24:42.5052Z","iopub.status.idle":"2021-11-19T07:24:42.512733Z","shell.execute_reply.started":"2021-11-19T07:24:42.505144Z","shell.execute_reply":"2021-11-19T07:24:42.511379Z"}}
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:24:50.386407Z","iopub.execute_input":"2021-11-19T07:24:50.386823Z","iopub.status.idle":"2021-11-19T07:26:34.968869Z","shell.execute_reply.started":"2021-11-19T07:24:50.386747Z","shell.execute_reply":"2021-11-19T07:26:34.968025Z"}}
best_model = grid_search.fit(X_train, y_train)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:26:44.124126Z","iopub.execute_input":"2021-11-19T07:26:44.124479Z","iopub.status.idle":"2021-11-19T07:26:44.328571Z","shell.execute_reply.started":"2021-11-19T07:26:44.124443Z","shell.execute_reply":"2021-11-19T07:26:44.327849Z"}}
rf_pred=best_model.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:26:53.691827Z","iopub.execute_input":"2021-11-19T07:26:53.692484Z","iopub.status.idle":"2021-11-19T07:26:53.981441Z","shell.execute_reply.started":"2021-11-19T07:26:53.692413Z","shell.execute_reply":"2021-11-19T07:26:53.980574Z"}}
print("Classification Report is:\n",classification_report(y_test,rf_pred))
print("\n F1:\n",f1_score(y_test,knn_pred))
print("\n Precision score is:\n",precision_score(y_test,knn_pred))
print("\n Recall score is:\n",recall_score(y_test,knn_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test,rf_pred))


# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:27:04.691096Z","iopub.execute_input":"2021-11-19T07:27:04.691458Z","iopub.status.idle":"2021-11-19T07:27:04.696784Z","shell.execute_reply.started":"2021-11-19T07:27:04.691412Z","shell.execute_reply":"2021-11-19T07:27:04.695808Z"}}
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:27:20.769555Z","iopub.execute_input":"2021-11-19T07:27:20.769942Z","iopub.status.idle":"2021-11-19T07:27:20.793095Z","shell.execute_reply.started":"2021-11-19T07:27:20.769898Z","shell.execute_reply":"2021-11-19T07:27:20.792209Z"}}
reg = LogisticRegression()
reg.fit(X_train,y_train)                         

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:27:23.676794Z","iopub.execute_input":"2021-11-19T07:27:23.677849Z","iopub.status.idle":"2021-11-19T07:27:23.683791Z","shell.execute_reply.started":"2021-11-19T07:27:23.6778Z","shell.execute_reply":"2021-11-19T07:27:23.683005Z"}}
lr_pred=reg.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-19T07:27:26.810511Z","iopub.execute_input":"2021-11-19T07:27:26.811192Z","iopub.status.idle":"2021-11-19T07:27:27.109528Z","shell.execute_reply.started":"2021-11-19T07:27:26.811147Z","shell.execute_reply":"2021-11-19T07:27:27.108669Z"}}
print("Classification Report is:\n",classification_report(y_test,lr_pred))
print("\n F1:\n",f1_score(y_test,lr_pred))
print("\n Precision score is:\n",precision_score(y_test,lr_pred))
print("\n Recall score is:\n",recall_score(y_test,lr_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test,lr_pred))