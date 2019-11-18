#!/usr/bin/env python3
# for the sentiment contest

import pandas
import argparse
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


# Where the magic happens
def process(cmd_line_args):
    # Ensure we have this for lemmatization
    nltk.download("wordnet")
    # Open our file(s)
    print("Reading Data")
    train_set, test_set = open_files(cmd_line_args)

    # Preprocess
    print("Preprocessing")
    train_set.text = train_set.text.apply(preprocess_tweet)
    test_set.text = test_set.text.apply(preprocess_tweet)

    # Lemmatize
    # print("Lemmatizing")
    # train_set = lemmatize(train_set)
    # test_set = lemmatize(test_set)

    # TFIDF-ize
    print("Running TFIDF")
    training_tfidf, testing_tfidf = tfidf(train_set, test_set)

    # PREDICT!
    print("Predicting?")
    test_set = predict(training_tfidf, testing_tfidf, train_set, test_set)

    # If we're running a test, we'll do a score
    if not cmd_line_args.test_file:
        score(test_set)

    # Print out the file
    output_file(test_set, cmd_line_args.test_file)


def score(test_set):
    score = accuracy_score(
        test_set.sentiment.to_numpy(dtype=str),
        test_set.predicted_sentiment.to_numpy(dtype=str),
    )
    print(f"Score: {score}")


def preprocess_tweet(tweet):
    # Preprocess the text in a single tweet
    # convert the tweet to lower case
    tweet = str(tweet)
    tweet.lower()
    # convert all urls to sting "URL"
    tweet = re.sub("((www\.[^\s]+)|(https?://[^\s]+))", "URL", tweet)
    # convert all @username to "AT_USER"
    tweet = re.sub("@[^\s]+", "AT_USER", tweet)
    # correct all multiple white spaces to a single white space
    tweet = re.sub("[\s]+", " ", tweet)
    # convert "#topic" to just "topic"
    tweet = re.sub(r"#([^\s]+)", r"\1", tweet)
    return tweet


def lemmatize(data_frame):
    # Still gotta figure this one out
    return data_frame


def open_files(cmd_line_args):
    data_frame = pandas.read_csv(
        "Training.txt",
        sep="\t",
        error_bad_lines=False,
        warn_bad_lines=False,
        engine="python",
    )
    if cmd_line_args.test_file:
        train_set = data_frame
        test_set = pandas.read_csv(
            cmd_line_args.test_file,
            sep="\t",
            error_bad_lines=False,
            warn_bad_lines=False,
            engine="python",
        )
    else:
        train_set, test_set = train_test_split(data_frame, test_size=0.2)
    train_set = train_set.rename(
        columns={"Sentiment1": "sentiment", "SentimentText": "text"}
    )
    train_set = train_set.drop("TonyID", axis=1)
    test_set = test_set.rename(
        columns={"Sentiment1": "sentiment", "SentimentText": "text"}
    )
    test_set = test_set.drop(["TonyID"], axis=1)
    return train_set, test_set


def predict(training_tfidf, testing_tfidf, train_set, test_set):
    classifier = MultinomialNB()
    classifier = classifier.fit(training_tfidf, train_set.sentiment.astype("U"))
    test_set["predicted_sentiment"] = classifier.predict(testing_tfidf)
    return test_set


def tfidf(training_data_frame, testing_data_frame):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        # ngram_range=(1, 1),
        # analyzer="word",
        # max_df=0.57,
        # binary=False,
        # token_pattern=r"\w+",
        sublinear_tf=True,
    )
    training_tfidf = vectorizer.fit_transform(training_data_frame.text)
    testing_tfidf = vectorizer.transform(testing_data_frame.text)
    return training_tfidf, testing_tfidf


def output_file(test_set, test_file):
    filename = f"coltri_{test_file}"
    print(f"Writing File: {filename}")
    test_set = test_set.sort_index()
    columns = ["predicted_sentiment"]
    test_set.to_csv(filename, index=True, columns=columns, header=False, sep=" ")


def _with_cmd_line_args(f):
    def wrapper(*args, **kwargs):
        p = argparse.ArgumentParser()
        p.add_argument("-f", "--test_file", help="Path to the test file")
        return f(p.parse_args(), *args, **kwargs)

    return wrapper


@_with_cmd_line_args
def main(cmd_line_args):
    print("Who is feeling positive??!?!?!")
    if not cmd_line_args.test_file:
        print("*** TRIAL RUN ***")
    process(cmd_line_args)


if __name__ == "__main__":
    main()
