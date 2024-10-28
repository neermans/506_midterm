# ADDING METRICS TO CSV

import csv
import re



with open('external_resources/positive-words.txt', 'r') as f:
    positive_words = set(f.read().lower().strip('\n').split('\n'))

with open('external_resources/negative-words.txt', 'r') as f:
    negative_words = set(f.read().lower().strip('\n').split('\n'))


def get_helpfulness(entry):
  try:
    value = int(entry['HelpfulnessNumerator']) / int(entry['HelpfulnessDenominator'])
  except:
    value = 0
  return value


def get_positive(entry):
  split_entry = re.sub(r'[^a-zA-Z]', ' ', entry['Text']).lower().split()
  num_neg = sum([split_entry.count(word) for word in negative_words])
  num_pos = sum([split_entry.count(word) for word in positive_words])
  ratio = num_pos / (num_neg + num_pos) if num_neg + num_pos else 0.5
  return {
    'NumNegative': num_neg,
    'NumPositive': num_pos,
    'PercPositive': ratio
  }



with open('train.csv', 'r') as f, open('train.modded.csv', 'w') as out_f:
  reader = csv.reader(f)
  out = csv.writer(out_f)
  headers = next(reader)
  modded_headers = headers + ['Helpfulness', 'NumNegative', 'NumPositive', 'PercPositive']
  out.writerows([modded_headers])
  for i, line in enumerate(reader):
    if i % 1_000 == 0:
       print(f'PROCESSED {i} LINES')
    entry = dict(zip(headers, line))
    entry['Helpfulness'] = get_helpfulness(entry)
    entry.update(get_positive(entry))
    _ = out.writerows([[entry[k] for k in modded_headers]])




'''
possible interesting metrics
total number of reviews for movie
time from earliest review???
score for summary?
number of words in review

'''


#EFFECTIVELY THE SAME AS WHAT THE TEACHER GAVE YOU


import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
from os.path import exists
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


trainingSet = pd.read_csv("train.csv")
testingSet = pd.read_csv("test.csv")

print("train.csv shape is ", trainingSet.shape)
print("test.csv shape is ", testingSet.shape)

print()

print(trainingSet.head())
print()
print(testingSet.head())

print()

print(trainingSet.describe())

trainingSet['Score'].value_counts().plot(kind='bar', legend=True, alpha=.5)
plt.show()

print()
print("EVERYTHING IS PROPERLY SET UP! YOU ARE READY TO START")


with open('external_resources/positive-words.txt', 'r') as f:
    positive_words = set(f.read().lower().strip('\n').split('\n'))

with open('external_resources/negative-words.txt', 'r') as f:
    negative_words = set(f.read().lower().strip('\n').split('\n'))



def add_features_to(df):
    # This is where you can do all your feature extraction
    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Helpfulness'] = df['Helpfulness'].fillna(0)
    # df['Text'] = df['Text'].fillna('').astype(str)
    # df['NumNegative'] = df['Text'].apply(lambda x: sum(x.lower().count(rf'\b{word}\b') for word in negative_words))
    # df['NumPositive'] = df['Text'].apply(lambda x: sum(x.lower().count(rf'\b{word}\b') for word in positive_words))
    # df['PercPositive'] = df['NumPositive'] / (df['NumPositive'] + df['NumNegative'])
    # df['PercPositive'].fillna(0.5)
    return df


# Load the feature extracted files if they've already been generated
if exists('X_train.csv'):
    X_train = pd.read_csv("X_train.csv")


if exists('X_submission.csv'):
    X_submission = pd.read_csv("X_submission.csv")
else:
    # Process the DataFrame
    train = add_features_to(trainingSet)
    # Merge on Id so that the submission set can have feature columns as well
    X_submission = pd.merge(train, testingSet, left_on='Id', right_on='Id')
    X_submission = X_submission.drop(columns=['Score_x'])
    X_submission = X_submission.rename(columns={'Score_y': 'Score'})
    # The training set is where the score is not null
    X_train =  train[train['Score'].notnull()]
    X_submission.to_csv("X_submission.csv", index=False)
    X_train.to_csv("X_train.csv", index=False)


x, y = [], []

# Split training set into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(
    X_train.drop(columns=['Score']),
    X_train['Score'],
    test_size=1/4.0,
    random_state=0
)

features = ['HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time', 'Helpfulness']#, 'NumPositive', 'NumNegative', 'PercPositive']

X_train_select = X_train[features]
X_test_select = X_test[features]
X_submission_select = X_submission[features]

# Learn the model
model = KNeighborsClassifier(n_neighbors=3).fit(X_train_select, Y_train)

# Predict the score using the model
Y_test_predictions = model.predict(X_test_select)

# Evaluate your model on the testing set
print("Accuracy on testing set = ", accuracy_score(Y_test, Y_test_predictions))

# Plot a confusion matrix
cm = confusion_matrix(Y_test, Y_test_predictions, normalize='true')
sns.heatmap(cm, annot=True)
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Create the submission file
X_submission['Score'] = model.predict(X_submission_select)
submission = X_submission[['Id', 'Score']]
submission.to_csv("submission.csv", index=False)

#Fixed!Success!Done.
