"importing libraries"
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("Train_data.csv")
test = pd.read_csv("Test_data.csv")

print(train.head(4))
print(test.head(4))
print("---------------------")
print("Training data has {} rows & {} columns".format(train.shape[0],train.shape[1]))
print()
print("Testing data has {} rows & {} columns".format(test.shape[0],test.shape[1]))
print("---------------------")
train.describe()

#Visualization
#Heatmap for train data
plt.figure(figsize=(15, 15))
sns.heatmap(train.corr(), linewidths=.8)

"removing num_outbound_cmds column from datasets"
train.drop(['num_outbound_cmds'], axis=1, inplace=True)
test.drop(['num_outbound_cmds'], axis=1, inplace=True)
print()

"Attack Class Distribution count"
train['class'].value_counts()

"Standard scaler function"
scaler = StandardScaler()

"extract numerical attributes and scale it to have zero mean and unit variance  "
cols = train.select_dtypes(include=['float64','int64']).columns
sc_train = scaler.fit_transform(train.select_dtypes(include=['float64','int64']))
sc_test = scaler.fit_transform(test.select_dtypes(include=['float64','int64']))

"turn the result back to a dataframe"
sc_traindf = pd.DataFrame(sc_train, columns = cols)
sc_testdf = pd.DataFrame(sc_test, columns = cols)

"Label encoding"
encoder = LabelEncoder()

"extract categorical attributes from both training and test sets "
cattrain = train.select_dtypes(include=['object']).copy()
cattest = test.select_dtypes(include=['object']).copy()

"encode the categorical attributes"
traincat = cattrain.apply(encoder.fit_transform)
testcat = cattest.apply(encoder.fit_transform)

"separate target column from encoded data "
enctrain = traincat.drop(['class'], axis=1)
cat_Ytrain = traincat[['class']].copy()

train_x = pd.concat([sc_traindf,enctrain],axis=1)
train_y = train['class']
train_x.shape

test_df = pd.concat([sc_testdf,testcat],axis=1)
test_df.shape

"Training and testing sets"
X_train,X_test,Y_train,Y_test = train_test_split(train_x,train_y,train_size=0.80, random_state=2)

" KNeighborsClassifier Model"
KNN_Classifier = KNeighborsClassifier(n_jobs=-1)
KNN_Classifier.fit(X_train, Y_train); 

"Random Forest Model"
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,Y_train)


models = []

models.append(('Random Forest', clf))
models.append(('KNeighborsClassifier', KNN_Classifier))

for i, v in models:
    scores = cross_val_score(v, X_train, Y_train, cv=5)
    accuracy = metrics.accuracy_score(Y_train, v.predict(X_train))
    confusion_matrix = metrics.confusion_matrix(Y_train, v.predict(X_train))
    classification = metrics.classification_report(Y_train, v.predict(X_train))
    print()
    print('============ {} Model Evaluation ============'.format(i))
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print()
    print (" Accuracy:" "\n", accuracy)
    print()
    


