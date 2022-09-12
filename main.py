# Importing dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import random

# Reading .csv file
mail_dataframe = pd.read_csv('mail.csv')

# Converting labels to integer format
mail_dataframe.loc[mail_dataframe['label'] == 'spam', 'label', ] = 0
mail_dataframe.loc[mail_dataframe['label'] == 'ham', 'label', ] = 1
mail_dataframe['label'] = mail_dataframe['label'].astype('int32')

# Train_test splitting
X = mail_dataframe['text']
y = mail_dataframe['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Selecting a random mail from test set for testing
print(X_test.shape)
a = random.randint(0, 1034)
print("The selected example mail: ", X_test.iloc[a])

# Feature extraction
extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train = extraction.fit_transform(X_train)
X_test = extraction.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction and accuracy testing
prediction = model.predict(X_test)
if prediction[a] == 0:
    print("Spam")
else:
    print("Ham")
accuracy = accuracy_score(y_test, prediction)
print("Accuracy of the model:", accuracy)
