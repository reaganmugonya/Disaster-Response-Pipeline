# import libraries
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import pickle


def load_data(database_filepath):
    '''
    Args:
    database_filepath: path to the database
    
    return:
    X: Dataframe containing features
    Y: Dataframe containing labels
    category_names: names of labels
    '''
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('InsertTableName', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    '''
    Args:
    text: string message 
    
    return:
    Stemmed, lemmatized and not stop words
    '''
    #remove punctuation characters and convert to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #stop words
    stop_words = stopwords.words("english")
    
    # tokenize text
    tokens = word_tokenize(text)
    
    #Reduce tokenz to their stems
    stemmed = [PorterStemmer().stem(w) for w in tokens]
    
    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]
   
    return lemmed


def build_model():
    '''
    Args:
    none
    
    return:
    Model with a GridSearchCV
    '''
    
    #build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])
    
    #create GridSearch  parameters
    '''  
    There are more parameter that could be use.
    More parameters lead to better results 
    but require more training time
    
    '''
    
    parameters = {'tfidf__use_idf':[True, False],
              #'tfidf__norm': ('l1', 'l2', None),
              'vect__max_df': (0.5, 0.75, 1.0),
              'clf__estimator__n_estimators':[10, 25], 
              'clf__estimator__min_samples_split':[2, 5]}

    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 5)
    
    
    
    #return model
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    Args:
    Model: fitted model
    X_test: test features 
    Y_test: test labels
    category_name: category names
    
    return:
    none
    '''
    # predict on test data
    Y_pred = model.predict(X_test)

    #f1 score, precision and recall for each output category
    for i, category_names in enumerate(Y_test):
        print(category_names)
        print(classification_report(Y_test[category_names], Y_pred[:, i]))


def save_model(model, model_filepath):
    '''
    model: fitted model
    model_filepath: path to save the fitted model
    
    returns:
    none
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()