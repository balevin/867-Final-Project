import pandas as pd
from collections import defaultdict
import numpy as np


def captionList():
    data = pd.read_csv('../parsedData.csv')
    return list(data['Description'])

captions = captionList()
# print(captions)
def build_dictionary(train_captions):
    word_counts = defaultdict(float)
    captions = train_captions
    for sent in captions:
        print(sent)
        for word in sent.replace('.', '').replace(',', '').split(' '):
            word_counts[word] += 1

    vocab = [w for w in word_counts if word_counts[w] >= 0]

    ixtoword = {}
    ixtoword[0] = '<end>'
    wordtoix = {}
    wordtoix['<end>'] = 0
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    train_captions_new = []
    for t in train_captions:
        rev = []
        for w in t:
            if w in wordtoix:
                rev.append(wordtoix[w])
        # rev.append(0)  # do not need '<end>' token
        train_captions_new.append(rev)


    return vocab, train_captions_new, ixtoword, wordtoix, len(ixtoword)
    

def get_caption(sent_ix):
    # a list of indices for a sentence
    sent_caption = np.asarray(captions[sent_ix]).astype('int64')
    if (sent_caption == 0).sum() > 0:
        print('ERROR: do not need END (0) token', sent_caption)
    num_words = len(sent_caption)
    # pad with 0s (i.e., '<end>')
    x = np.zeros((18, 1), dtype='int64')
    x_len = num_words
    if num_words <= 18:
        x[:num_words, 0] = sent_caption
    else:
        ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
        np.random.shuffle(ix)
        ix = ix[:18]
        ix = np.sort(ix)
        x[:, 0] = sent_caption[ix]
        x_len = 18
    return x, x_len

print(build_dictionary(captions))
vocab, train_captions_new, ixtoword, wordtoix, length = build_dictionary(captions)
sentence = "A short haired asian girl with a soft smile"
indexList = []
for word in sentence.split(' '):
    indexList.append(vocab.index(word))
print(indexList)
# print('________________')
# print(vocab)
# print(len(vocab))

# print("________________")
# print(train_captions_new)
# get_caption(0)