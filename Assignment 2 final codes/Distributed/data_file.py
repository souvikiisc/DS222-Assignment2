from __future__ import division
import string
import math
import re
import numpy as np
from collections import defaultdict
from itertools import dropwhile
# from nltk.corpus import stopwords
import pickle
from scipy import sparse
stop_words = {"some", "don", "she", "wasn't", "didn't", "once", "it", "between", "which", "s", "these",
			  "such", "me", "few", "from", "will", "same", "whom", "needn", "very", "through", "are", "she's", "we", "down", "off", "against",
			  "here", "o", "above", "when", "him", "he", "if", "am", "who", "haven't", "do", "has", "should", "you", "mustn't", "a", "again", "had", "up",
			  "ve", "weren", "yours", "their", "only", "her", "them", "aren", "not", "shouldn", "ourselves", "your", "too", "hasn", "most", "until", "d", "did", "all", "ma", "won't",
			  "no", "after", "wouldn", "didn", "of", "with", "t", "you'd", "couldn", "so", "doesn", "wouldn't", "ours", "there", "don't", "hadn", "needn't", "aren't", "its", "now",
			  "mightn", "yourselves", "you'll", "doing", "can", "but", "you've", "in", "other", "wasn", "and", "further", "won", "own", "they", "an", "how", "this", "because", "than",
			  "hadn't", "before", "were", "just", "as", "having", "isn", "himself", "during", "what", "couldn't", "ain", "into", "shouldn't", "weren't", "does", "was", "is", "that'll",
			  "about", "nor", "themselves", "while", "y", "mustn", "you're", "myself", "where", "at", "yourself", "doesn't", "itself", "i", "re", "each", "why", "those", "theirs", "to",
			  "both", "that", "his", "below", "ll", "mightn't", "should've", "haven", "it's", "any", "out", "being",
			  "shan", "then", "isn't", "herself", "hasn't", "under", "have", "on", "hers", "m", "over", "our", "shan't", "for", "more", "be", "or", "the", "my", "by", "been", }


def tokenize1(document):
    values = document.split("\t")
    label = values[0].rstrip().split(",")
    words = re.sub("\d+", "", values[1].rsplit('"', 1)[0].split('"', 1)[1])
    regex = r'(\w*)'
    list1 = re.findall(regex, words)
    while '' in list1:
        list1.remove('')
    list1 = map(str.lower, list1)
    # stop_words = set(stopwords.words('english'))
    list1 = [w for w in list1 if not w in stop_words]
    return list1


def labelize(document):
    values = document.split("\t")
    label = values[0].rstrip().split(",")
    return label


dy = defaultdict(int)
dz = defaultdict(int)
train_dir = "/scratch/ds222-2017/assignment-1/DBPedia.full/full_train.txt"
# train_dir="/scratch/ds222-2017/assignment-1/DBPedia.verysmall/verysmall_train.txt"
filename_train = open(train_dir, "r")
lines = filename_train.readlines()[3:]
# lines=[lines[165]]
for line in lines:
    values = line.split("\t")
    labels = values[0].rstrip().split(",")
    words = re.sub("\d+", "", values[1].rsplit('"', 1)[0].split('"', 1)[1])
    regex = r'(\w*)'
    list1 = re.findall(regex, words)
    while '' in list1:
        list1.remove('')
    list1 = map(str.lower, list1)
    # stop_words = set(stopwords.words('english'))
    list1 = [w for w in list1 if not w in stop_words]
    for word in list1:
        dy[word] += 1
    for label in labels:
        dz[label] += 1

# dy=sorted(dy.iteritems(), key=lambda (k,v): (v,k))
# dz=sorted(dz.iteritems(), key=lambda (k,v): (v,k))


threshold_value = 100

i = 0
dy_ = {}
for k, v in dy.items():
    if v > threshold_value:
        dy_[k] = dy[k]
    i = i + 1

i = 0
dy = dy_.copy()
for la in dy.keys():
    dy[la] = i
    i = i + 1

i = 0
for lz in dz.keys():
    dz[lz] = i
    i = i + 1

print(len(dy))
print(len(dz))
# print(dy)
# print(dz)

train_dir = "/scratch/ds222-2017/assignment-1/DBPedia.full/full_train.txt"
# train_dir="/scratch/ds222-2017/assignment-1/DBPedia.verysmall/verysmall_train.txt"
filename_train = open(train_dir, "r")
lines_train = filename_train.readlines()

test_dir = "/scratch/ds222-2017/assignment-1/DBPedia.full/full_test.txt"
# test_dir="/scratch/ds222-2017/assignment-1/DBPedia.verysmall/verysmall_test.txt"
filename_test = open(test_dir, "r")
lines_test = filename_test.readlines()

train = np.zeros((len(lines_train), len(dy)), dtype=np.float32)
test = np.zeros((len(lines_test), len(dy)), dtype=np.float32)
train_l = np.zeros((len(lines_train), len(dz)), dtype=np.float32)
test_l = np.zeros((len(lines_test), len(dz)), dtype=np.float32)

i = 0
for line in lines_test:
    word = tokenize1(line)
    label = labelize(line)

    for l in label:
        test_l[i, dz[l]] = 1
    for w in word:
        if dy.__contains__(w):
            test[i, dy[w]] = 1
    test_l[i, :] = test_l[i, :] / np.sum(test_l[i, :])
    i = i + 1

i = 0
for line in lines_train:
    word = tokenize1(line)
    label = labelize(line)

    for l in label:
        train_l[i, dz[l]] = 1
    for w in word:
        if dy.__contains__(w):
            train[i, dy[w]] = 1
    train_l[i, :] = train_l[i, :] / np.sum(train_l[i, :])
    i = i + 1

print(np.sum(train[0, :]))
print(train_l[0, :])
print(np.sum(test[0, :]))
print(test_l[0, :])

np.save("train_labels", train_l)
# np.save("train",train)
# np.save("test",test)
np.save("test_labels", test_l)
train = sparse.csr_matrix(train)
test = sparse.csr_matrix(test)

sparse.save_npz('train_set.npz', train)

sparse.save_npz('test_set.npz', test)