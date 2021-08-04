"""
@author: Yi Yang
@contact: yyang2@bowdoin.edu
@date: 8-4-2021
@desc:

This script utilizes the process_clauses_from_mastercsv.py script file.
Uses the two CSV files created from the above script file to run several traditional machine learning algorithms.
Stores the accuracy score and other relevant results of the algorithm into a text file.

This script requires that that all the imported modules and packages below to be installed within the Python environment
you are running the script in.

Also, file paths have been hard-coded, and requires user to manually change them in the code.

The file can also be imported as a module, and contains the following functions:
    - tf_idf_representation
    - model_fitting
"""
import lightgbm
import nltk
from process_clauses_from_mastercsv import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import xgboost as xgb
import lightgbm as lgb
from sklearn import tree
import pandas as pd
import numpy as np

# Traditional ML Classifiers
nb_classifier = MultinomialNB()
knn_classifier = KNeighborsClassifier()
cart_classifier = tree.DecisionTreeClassifier()
svc_classifer = SVC()   # C-Support Vector Classification. Implements the "One-Versus-One" approach
linearsvc_classifier = LinearSVC() # Implements the "One-Versus-Rest" approach
rf_classifier = RandomForestClassifier(n_estimators=3000, random_state=0)
# c45_model()
xgboost_classifier = xgb.XGBClassifier(objective='multi:softmax')
lightgbm_classifier = lgb.LGBMClassifier()

# Check for overfitting using this guide: https://www.kaggle.com/prashant111/lightgbm-classifier-in-python


def tf_idf_representation(list_of_sentences, labels):
    # Bag of Words
    vectorizer = CountVectorizer(max_features=20000, ngram_range=(1, 3), min_df=0.01, max_df=0.8,
                                 stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(list_of_sentences).toarray()
    Y = labels
    # TF-IDF Conversion
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()

    print("tf-idf completed")

    return X, Y


def modelfitting(classifier):
    cv_rf = cross_validate(classifier, X_train, y_train, cv=10)
    # print("Cross validation accuracy mean score: ", cv_rf['test_score'].mean())
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    confusion_m = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix: ", confusion_m, "\n")
    classification_r = classification_report(y_test, y_pred)
    # print("Classification Report: ", classification_r, "\n")
    accuracy_s = accuracy_score(y_test, y_pred, "\n")
    # print("Accuracy Score: ", accuracy_s)

    with open(r'C:\Users\yiyan\Desktop\Legal Text Classification\tf_idf\classifiers_results4.txt', 'a') as text_file:
        text_file.write('\n')
        text_file.write("Classification Report: " + '\n' + str(classification_r) + '\n')
        text_file.write("Accuracy Score: " + str(accuracy_s) + '\n')
        text_file.write("Cross validation accuracy mean score: " + str(cv_rf['test_score'].mean()) + "\n")
        text_file.write("=========================================================" + '\n')

    print("Model Fitting completed")


# # C4.5
# def c45_model():
#     # In second key, 'Decision' is equal to 'Sentence ID'
#     dict_sentences_to_label = {'Sentence Text': list_of_sentences, 'Decision': labels}
#     df = pd.DataFrame(dict_sentences_to_label)
#     config = {'algorithm': 'C4.5', 'enableParallelism': True, 'num_cores': 2}
#     model = chef.fit(df, config=config)
#     return

if __name__ == '__main__':
    doc_result = extract_and_create_documentlevel_csv()
    # sent_result = extract_and_create_sentencelevel_csv()
    # for item in result[0]:
    #     print("Item: ", item)

    # list_of_sentences = sent_result[0]
    # sent_labels = sent_result[3]

    list_of_text = doc_result[0]
    doc_labels = doc_result[2]

    X, Y = tf_idf_representation(list_of_sentences=list_of_text, labels=doc_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    modelfitting(classifier=lightgbm_classifier)

