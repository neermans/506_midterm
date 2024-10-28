import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
from os.path import exists
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


trainingSet = pd.read_csv("train.modded.csv")
testingSet = pd.read_csv("test.csv")

print("train.modded.csv shape is ", trainingSet.shape)
print("test.csv shape is ", testingSet.shape)

print()

print(trainingSet.head())
print()
print(testingSet.head())

print()

print(trainingSet.describe())

# trainingSet['Score'].value_counts().plot(kind='bar', legend=True, alpha=.5)
# plt.show()

print()
print("EVERYTHING IS PROPERLY SET UP! YOU ARE READY TO START")



def add_features_to(df):
    return df


# Load the feature extracted files if they've already been generated
if exists('X_train.csv'):
    X_train = pd.read_csv("X_train.csv")


if exists('X_submission.csv'):
    X_submission = pd.read_csv("X_submission.csv")
else:
    # Process the DataFrame
    print('Adding features')
    train = add_features_to(trainingSet)
    # Merge on Id so that the submission set can have feature columns as well
    X_submission = pd.merge(train, testingSet, left_on='Id', right_on='Id')
    X_submission = X_submission.drop(columns=['Score_x'])
    X_submission = X_submission.rename(columns={'Score_y': 'Score'})
    # The training set is where the score is not null
    X_train =  train[train['Score'].notnull()]
    X_submission.to_csv("X_submission.csv", index=False)
    X_train.to_csv("X_train.csv", index=False)



y = X_train['Score']  # Target variable (5 possible values)

# Encode target variable if it's categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split training set into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(
    X_train.drop(columns=['Score']),
    y_encoded,
    test_size=1/4.0,
    random_state=0
)

features = ['HelpfulnessNumerator',
            'HelpfulnessDenominator',
            'Helpfulness',
            'NumNegativeText',
            'NumPositiveText',
            'PercPositiveText',
            'NumNegativeSummary',
            'NumPositiveSummary',
            'PercPositiveSummary',
            'NumProductReviews',
            'NumUserReviews',
            ]#'TimeFromFirst']

X_train_select = X_train[features]
X_test_select = X_test[features]
X_submission_select = X_submission[features]

# Learn the model
model = xgb.XGBClassifier(
    objective='multi:softmax',  # For classification tasks with finite values
    num_class=5,  # Number of possible classes (as per your description)
    eval_metric='mlogloss'  # Evaluation metric for multiclass classification
)

model.fit(X_train_select, Y_train)
Y_test_predictions = model.predict(X_test_select)


# Evaluate your model on the testing set
print("Accuracy on testing set = ", accuracy_score(Y_test, Y_test_predictions))


y_pred_labels = label_encoder.inverse_transform(Y_test_predictions)


# Plot a confusion matrix
cm = confusion_matrix(Y_test, y_pred_labels, normalize='true')
sns.heatmap(cm, annot=True)
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# # Create the submission file
# X_submission['Score'] = model.predict(X_submission_select)
# submission = X_submission[['Id', 'Score']]
# submission.to_csv("submission.csv", index=False)
