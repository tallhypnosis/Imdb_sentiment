#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer


# In[47]:


df = pd.read_csv('/content/Train.csv')


# In[48]:


df


# In[49]:


df.head()


# In[50]:


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[51]:


def prerpocess_text(text):
  text = text.lower()
  tokens = nltk.word_tokenize(text)
  filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
  return ' '.join(filtered_tokens)


# In[52]:


def lemmatize_text(text):
  return ' '.join([WordNetLemmatizer().lemmatize(word) for word in text.split()])


# In[53]:


df.head()


# In[54]:


df['cleaned_text'] = df['text'].apply(prerpocess_text)


# In[55]:


df['cleaned_text_lemmatized'] = df['cleaned_text'].apply(lemmatize_text)


# In[56]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[57]:


tfidf = TfidfVectorizer(max_features=50000, max_df=0.90, min_df=5).fit(df['cleaned_text_lemmatized'])


# In[58]:


tfidf.vocabulary_.__len__()


# In[59]:


X = tfidf.fit_transform(df['cleaned_text_lemmatized'])
y = df['label']


# In[60]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)


# In[61]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

lr_predictions = lr_model.predict(X_test)
nb_predictions = nb_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions))

print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions ))
print("Classification Report:\n", classification_report(y_test, nb_predictions))

cm = confusion_matrix(y_test, lr_predictions)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

cm = confusion_matrix(y_test, nb_predictions)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Naive Bayes Confusion Matrix')
plt.show()


# In[62]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Print the classification report (precision, recall, F1-score)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f"Confusion Matrix for {type(model).__name__}")
    plt.show()

    return accuracy

# Example for Logistic Regression
logistic_accuracy = evaluate_model(lr_model, X_test, y_test)

# Example for Naive Bayes
nb_accuracy = evaluate_model(nb_model, X_test, y_test)


# In[63]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Define parameter grid for Logistic Regression
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

# GridSearchCV for Logistic Regression
log_grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
log_grid.fit(X_train, y_train)

# Print the best parameter and accuracy
print(f"Best C for Logistic Regression: {log_grid.best_params_}")
log_best_model = log_grid.best_estimator_
log_best_accuracy = evaluate_model(log_best_model, X_test, y_test)


# In[64]:


from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

# Define parameter grid for Naive Bayes
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}

# GridSearchCV for Naive Bayes
nb_grid = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
nb_grid.fit(X_train, y_train)

# Print the best parameter and accuracy
print(f"Best alpha for Naive Bayes: {nb_grid.best_params_}")
nb_best_model = nb_grid.best_estimator_
nb_best_accuracy = evaluate_model(nb_best_model, X_test, y_test)


# In[65]:


df_val = pd.read_csv('/content/Valid.csv')


# In[66]:


df_test = pd.read_csv('/content/Test.csv')


# In[67]:


df_val


# In[25]:


df_val['cleaned_text'] = df_val['text'].apply(prerpocess_text)


# In[32]:


df_val['cleaned__lemmatized_text'] = df_val['cleaned_text'].apply(lemmatize_text)


# In[33]:


df_test['cleaned_text'] = df_test['text'].apply(prerpocess_text)


# In[34]:


df_test['cleaned_lemmatized_text'] = df_test['cleaned_text'].apply(lemmatize_text)


# In[35]:


X_val_tfidf = tfidf.transform(df_val['cleaned_lemmatized_text'])


# In[36]:


X_test_tfidf = tfidf.transform(df_test['cleaned_lemmatized_text'])


# In[69]:


y_val = df_val['label']
y_test_2 = df_test['label']


# In[70]:


# Predict the labels for the validation set
y_val_pred = log_best_model.predict(X_val_tfidf)

# Evaluate the performance on the validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Set Accuracy: {val_accuracy:.4f}")

# Print classification report for the validation set
print("Validation Set Classification Report:\n", classification_report(y_val, y_val_pred))

# Plot confusion matrix for validation set
conf_matrix_val = confusion_matrix(y_val, y_val_pred)
sns.heatmap(conf_matrix_val, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Validation Set')
plt.show()


# In[71]:


# Predict the labels for the test set
y_test_pred = log_best_model.predict(X_test_tfidf)

# Evaluate the performance on the test set
test_accuracy = accuracy_score(y_test_2, y_test_pred)
print(f"Test Set Accuracy: {test_accuracy:.4f}")

# Print classification report for the test set
print("Test Set Classification Report:\n", classification_report(y_test_2, y_test_pred))

# Plot confusion matrix for the test set
conf_matrix_test = confusion_matrix(y_test_2, y_test_pred)
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Test Set')
plt.show()


# In[43]:


# Predict the labels for the validation set
y_val_pred = log_best_model.predict(X_val_tfidf)

# Evaluate the performance on the validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Set Accuracy: {val_accuracy:.4f}")

# Print classification report for the validation set
print("Validation Set Classification Report:\n", classification_report(y_val, y_val_pred))

# Plot confusion matrix for validation set
conf_matrix_val = confusion_matrix(y_val, y_val_pred)
sns.heatmap(conf_matrix_val, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Validation Set')
plt.show()


# In[72]:


# Predict the labels for the validation set
y_val_pred = nb_best_model.predict(X_val_tfidf)

# Evaluate the performance on the validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Set Accuracy: {val_accuracy:.4f}")

# Print classification report for the validation set
print("Validation Set Classification Report:\n", classification_report(y_val, y_val_pred))

# Plot confusion matrix for validation set
conf_matrix_val = confusion_matrix(y_val, y_val_pred)
sns.heatmap(conf_matrix_val, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Validation Set')
plt.show()


# In[73]:


# Predict the labels for the test set
y_test_pred = nb_best_model.predict(X_test_tfidf)

# Evaluate the performance on the test set
test_accuracy = accuracy_score(y_test_2, y_test_pred)
print(f"Test Set Accuracy: {test_accuracy:.4f}")

# Print classification report for the test set
print("Test Set Classification Report:\n", classification_report(y_test_2, y_test_pred))

# Plot confusion matrix for the test set
conf_matrix_test = confusion_matrix(y_test_2, y_test_pred)
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Test Set')
plt.show()


# In[78]:


import nbformat
from nbconvert import PythonExporter

# Load the notebook
with open("https://colab.research.google.com/drive/18KVuQWWpMkfM_UC81B8BLN7e-4Vzbv2J#scrollTo=NotcHYHO-RCy") as f:
    notebook_content = nbformat.read(f, as_version=4)

# Convert to Python script
python_exporter = PythonExporter()
(script, resources) = python_exporter.from_notebook_node(notebook_content)

# Save the script to a .py file
with open("ImdbSenti.py", "w") as f:
    f.write(script)


# In[75]:


get_ipython().system('pwd')


# In[76]:


ls


# In[79]:


from google.colab import drive
drive.mount('/content/drive')


# In[80]:


get_ipython().system('ls /content/drive/My\\ Drive')


# In[ ]:




