from html.parser import HTMLParser
import re
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
import numpy as np
from data import vocab


def processEmail(emailContents):
    """PROCESSEMAIL preprocesses a the body of an email and
    returns a list of word_indices
    word_indices = PROCESSEMAIL(email_contents) preprocesses
    the body of an email and returns a list of indices of the
    words contained in the email.
    """

    # Load Vocabulary
    vocabList = vocab.dictionary

    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================
    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers

    # MATLAB CODE, NOT PYTHON:
    # hdrstart = strfind(email_contents, ([char(10) char(10)]));
    # email_contents = email_contents(hdrstart(1):end);

    # Lower case
    emailContents = emailContents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    soup = BeautifulSoup(emailContents, "lxml")
    emailContents = soup.get_text()

    # Handle Numbers
    # Look for one or more characters between 0-9
    numRegex = re.compile('[0-9]+')
    emailContents = numRegex.sub('number', emailContents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    linkRegex = re.compile('(http|https)://[^\s]*')
    emailContents = linkRegex.sub('httpaddr', emailContents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    emailRegex = re.compile('[^\s]+@[^\s]+')
    emailContents = emailRegex.sub('emailaddr', emailContents)

    # Handle $ sign
    dollarRegex = re.compile('[$]+')
    emailContents = dollarRegex.sub('dollar', emailContents)

    # remove all punctuation
    punctuationRegex = re.compile('[^\w\s]|_')
    emailContents = punctuationRegex.sub('', emailContents)

    # split into word vector
    emailContents = emailContents.split()

    # stem and look up the word
    wordVector = []
    ps = PorterStemmer()
    for word in emailContents:
        word = ps.stem(word)
        if word in vocabList:
            wordVector.append(vocabList[word])

    return wordVector


def getEmailFeatures(wordIndices):
    """takes in a word_indices vector and produces a feature vector from the word indices
    x = EMAILFEATURES(word_indices) takes in a word_indices vector and
    produces a feature vector from the word indices."""

    # Total number of words in the dictionary
    n = 1899

    # You need to return the following variables correctly.
    x = np.zeros(n)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return a feature vector for the
    #               given email (word_indices). To help make it easier to
    #               process the emails, we have have already pre-processed each
    #               email and converted each word in the email into an index in
    #               a fixed dictionary (of 1899 words). The variable
    #               word_indices contains the list of indices of the words
    #               which occur in one email.
    #               Concretely, if an email has the text:
    #
    #                  The quick brown fox jumped over the lazy dog.
    #
    #               Then, the word_indices vector for this text might look
    #               like:
    #                   60  100   33   44   10     53  60  58   5
    #
    #               where, we have mapped each word onto a number, for example:
    #
    #                   the   -- 60
    #                   quick -- 100
    #                   ...
    #
    #              (note: the above numbers are just an example and are not the
    #               actual mappings).
    #
    #              Your task is take one such word_indices vector and construct
    #              a binary feature vector that indicates whether a particular
    #              word occurs in the email. That is, x(i) = 1 when word i
    #              is present in the email. Concretely, if the word 'the' (say,
    #              index 60) appears in the email, then x(60) = 1. The feature
    #              vector should look like:
    #              x = [ 0 0 0 0 1 0 0 0 ... 0 0 0 0 1 ... 0 0 0 1 0 ..];

    wordIndices = set(wordIndices)
    wordIndices = list(wordIndices)
    x[wordIndices] = 1
    return x
