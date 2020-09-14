import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import xgboost as xgb
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler

# reading in CSV's from a file path
trainingData = pd.read_csv('mentalDataSet.csv')

#dealing with missing data
#Let’s get rid of the variables "Timestamp",“comments”, “state” just to make our lives easier.
trainingData = trainingData.drop(['comments'], axis= 1)
trainingData = trainingData.drop(['state'], axis= 1)
trainingData = trainingData.drop(['Timestamp'], axis= 1)

trainingData.isnull().sum().max() #just checking that there's no missing data missing...
trainingData.head(5)

# Assign default values for each data type
defaultInt = 0
defaultString = 'NaN'
defaultFloat = 0.0

# Create lists by data tpe
intFeatures = ['Age']
stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                 'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                 'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                 'seek_help']
floatFeatures = []

# Clean the NaN's
for feature in trainingData:
    if feature in intFeatures:
        trainingData[feature] = trainingData[feature].fillna(defaultInt)
    elif feature in stringFeatures:
        trainingData[feature] = trainingData[feature].fillna(defaultString)
    elif feature in floatFeatures:
        trainingData[feature] = trainingData[feature].fillna(defaultFloat)
    else:
        print('Error: Feature %s not recognized.' % feature)
trainingData.head(5)


gender = trainingData['Gender'].str.lower()


#Select unique elements
gender = trainingData['Gender'].unique()

#Made gender groups
male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]
female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

for (row, col) in trainingData.iterrows():

    if str.lower(col.Gender) in male_str:
        trainingData['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

    if str.lower(col.Gender) in female_str:
        trainingData['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

    if str.lower(col.Gender) in trans_str:
        trainingData['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

#Get rid of bullshit
stk_list = ['A little about you', 'p']
trainingData = trainingData[~trainingData['Gender'].isin(stk_list)]

#complete missing age with mean
trainingData['Age'].fillna(trainingData['Age'].median(), inplace = True)

# Fill with median() values < 18 and > 120
s = pd.Series(trainingData['Age'])
s[s<18] = trainingData['Age'].median()
trainingData['Age'] = s
s = pd.Series(trainingData['Age'])
s[s>120] = trainingData['Age'].median()
trainingData['Age'] = s

#Ranges of Age
trainingData['age_range'] = pd.cut(trainingData['Age'], [0, 20, 30, 65, 100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)

#There are only 0.014% of self employed so let's change NaN to NOT self_employed
#Replace "NaN" string from defaultString
trainingData['self_employed'] = trainingData['self_employed'].replace([defaultString], 'No')
print(trainingData['self_employed'].unique())

#Replace "NaN" string from defaultString

trainingData['work_interfere'] = trainingData['work_interfere'].replace([defaultString], 'Don\'t know')
print(trainingData['work_interfere'].unique())

labelDict = {}
for feature in trainingData:
    le = preprocessing.LabelEncoder()
    le.fit(trainingData[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    trainingData[feature] = le.transform(trainingData[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] = labelValue

for key, value in labelDict.items():
    print(key, value)

# Get rid of 'Country'
trainingData = trainingData.drop(['Country'], axis=1)
trainingData.head()

# Scaling Age
scaler = MinMaxScaler()
trainingData['Age'] = scaler.fit_transform(trainingData[['Age']])
trainingData.head()

feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere', 'coworkers', 'supervisor']
X = trainingData[feature_cols]
y = trainingData.treatment

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

model = xgb.XGBClassifier(max_depth=5, n_estimators=400, learning_rate=0.01)

model.fit(X_train,y_train)

testPred = model.predict(X_test)
acc = sk.metrics.accuracy_score(y_test, testPred)
print("Accuracy :"+str(acc*100))
