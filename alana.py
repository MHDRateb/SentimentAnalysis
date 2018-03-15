#!/usr/bin/env python

#  Author: Angela Chapman
#  Date: 8/6/2014
#
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Part 1 of the tutorial on Natural Language Processing.
#
# *************************************** #

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score 

if __name__ == '__main__':
    pa=os.path.dirname(__file__), '','trainSet.csv'
    train = pd.read_csv(os.path.join(*pa), header=0, \
                    delimiter=",", quoting=3)
    pa2=os.path.dirname(__file__), '','testSet.csv'
    test = pd.read_csv(os.path.join(*pa2), header=0, delimiter=",", \
                   quoting=3 )




    print ('Download text data sets. If you already have NLTK datasets downloaded, just close the Python download window...')
    #nltk.download()  # Download text data sets, including stop words

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list

    print ("Cleaning and parsing the training set ...\n")
    for i in range( 0, len(train["utterance"])):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["utterance"][i], True)))


    # ****** Create a bag of words from the training set
    #
    print ("Creating the bag of words...\n")


    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    np.asarray(train_data_features)

    # ******* Train a random forest using the bag of words
    #
    print ("Training the random forest (this may take a while)...")


    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 25)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit( train_data_features, train["sentiment"] )



    # Create an empty list and append the clean reviews one by one
    clean_test_reviews = []

    print ("Cleaning and parsing the test set...\n")
    for i in range(0,len(test["utterance"])):
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["utterance"][i], True)))

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    np.asarray(test_data_features)

    # Use the random forest to make sentiment label predictions
    print ("Predicting test labels...\n")
    result = forest.predict(test_data_features)

    print ("Test Accuracy  :: ", accuracy_score(test["sentiment"], result)) 

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

    # Use pandas to write the comma-separated output file
    pa3=os.path.dirname(__file__), '','dataPredication.csv'
    output.to_csv(os.path.join(*pa3), index=False, quoting=3)
    print ("Wrote results to dataPredication.csv")


