import nltk
from process_clauses_from_mastercsv import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# No Pre-Processing on Sentences - Accuracy: 0.7395377888819488
# Pre-Processing on Sentences - Accuracy: 0.7282948157401624
def tf_idf_randomforest(list_of_sentences, labels):
    vectorizer = CountVectorizer(max_features=20000, ngram_range=(1, 3), min_df=0.01, max_df=0.8,
                                 stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(list_of_sentences).toarray()
    Y = labels
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()
    print(type(X))
    print(X)
    print(type(Y))
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    classifier = RandomForestClassifier(n_estimators=3000, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


# four model: RandomForestClassifier,LinearSVC,MultinomialNB,LogisticRegression
def multimodels(document_list, labels):
    unique_labels = list(set(labels))
    vectorizer = CountVectorizer(max_features=20000, ngram_range=(1, 3), min_df=0.01, max_df=0.8,
                                 stop_words=stopwords.words('english'))
    features = vectorizer.fit_transform(document_list).toarray()
    X = vectorizer.fit_transform(document_list).toarray()
    Y = labels
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    clf = MultinomialNB().fit(X_train, y_train)

    models = [
        RandomForestClassifier(n_estimators=5000, random_state=0),
        LinearSVC(max_iter=5000),
        MultinomialNB(),
        LogisticRegression(solver='lbfgs', multi_class='auto',random_state=0),
    ]
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df,
                  size=8, jitter=True, edgecolor="gray", linewidth=3)
    plt.show()
    performance = cv_df.groupby('model_name').accuracy.mean()
    print(performance)


if __name__ == '__main__':
    result = extract_and_create_new_csv()
    # for item in result[0]:
    #     print("Item: ", item)
    tf_idf_randomforest(result[0], result[1])
