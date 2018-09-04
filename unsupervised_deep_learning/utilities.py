# https://deeplearningcourses.com/c/deep-learning-recurrent-neural-networks-in-python
# https://udemy.com/deep-learning-recurrent-neural-networks-in-python
import numpy as np
import string
import os

def all_parity_pairs(nbit, dtype=np.float32):
    # total number of samples (Ntotal) will be a multiple of 100
    # why did I make it this way? I don't remember.
    N = 2**nbit
    remainder = 100 - (N % 100)
    Ntotal = N + remainder
    X = np.zeros((Ntotal, nbit))
    Y = np.zeros(Ntotal)
    for ii in range(Ntotal):
        i = ii % N
        # now generate the ith sample
        for j in range(nbit):
            if i % (2**(j+1)) != 0:
                i -= 2**j
                X[ii,j] = 1
        Y[ii] = X[ii].sum() % 2
    return X.astype(dtype=dtype), Y.astype(dtype=dtype)


def all_parity_pairs_with_sequence_labels(nbit, dtype=np.float32):
    X, Y = all_parity_pairs(nbit)
    N, t = X.shape

    # we want every time step to have a label
    Y_t = np.zeros(X.shape, dtype=np.int32)
    for n in range(N):
        ones_count = 0
        for i in range(t):
            if X[n,i] == 1:
                ones_count += 1
            if ones_count % 2 == 1:
                Y_t[n,i] = 1

    X = X.reshape(N, t, 1).astype(np.float32)
    return X.astype(dtype=dtype), Y_t


def remove_punctuation(s):
    translator = str.maketrans(dict.fromkeys(string.punctuation))
    return str.translate(s, translator)

def get_robert_frost():
    word2idx = {'START': 0, 'END': 1}
    current_idx = 2
    sentences = []
    for line in open('small_files/robert_frost.txt'):
        line = line.strip()
        if line:
            tokens = remove_punctuation(line.lower()).split()
            sentence = []
            for t in tokens:
                if t not in word2idx:
                    word2idx[t] = current_idx
                    current_idx += 1
                idx = word2idx[t]
                sentence.append(idx)
            sentences.append(sentence)
    return sentences, word2idx


def get_wikipedia_data(n_files=None, vocab_size_limit=None, mandatory_words=[]):
    # get the names of all the files we want to read
    # DEVNOTE: first iteration - read only the files from folder AA
    prefix = "large_files/AA/"
    input_files = [file for file in os.listdir(prefix) if file.startswith("wiki")]
    sample_vocabulary = (vocab_size_limit is not None)

    # get only the specified number of files (if a number is specified)
    if n_files is not None:
        input_files = input_files[:n_files]

    # create the list to hold the sentences
    sentences = []

    # create the word2idx and idx2word mappings and the variable that will hold the index of the next word to add
    word2idx = {"START": 0, "END": 1}
    # idx2word = {0: "START", 1: "END"}
    current_idx = 2

    if sample_vocabulary:
        word_count = {"START": np.inf, "END": np.inf}

    # create a flag that tells us whether or not the previous line was HTML, meaning the next is a header and should
    # not be added to the list of sentences
    skip_next_line = False

    for file_name in input_files:
        print("Loading file: " + 'large_files/AA/' + file_name)
        for line in open('large_files/AA/' + file_name, encoding="utf8"):
            # skip HTML tag and set the flag that skips the next line, which is a header
            if line[0] == "<":
                skip_next_line = True

            # skip the end of the HTML tag
            elif line[:2] == "</":
                pass

            # skip header
            elif skip_next_line:
                skip_next_line = False

            # only add line if it's not empty, i.e. if it has length larger than 1
            elif len(line) > 1:
                # create list to hold the words in the current line
                sentence = []

                # strip of punctuation, convert to lowercase and split the words
                tokens = remove_punctuation(line.lower()).split()

                # now go through the words and add their index to the sentence
                for word in tokens:
                    if word in word2idx:
                        sentence.append(word2idx[word])
                        if sample_vocabulary:
                            word_count[word] += 1
                    else:
                        word2idx[word] = current_idx
                        sentence.append(current_idx)
                        if sample_vocabulary:
                            word_count[word] = 1
                        current_idx += 1

                sentences.append(sentence)

    # return the full result if a vocabulary sample size was not specified
    if not sample_vocabulary:
        return sentences, word2idx

    # create idx to word dictionary
    idx2word = {v: k for k, v in word2idx.items()}

    # if there was a limit specified, do extra processing
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    # create the truncated word2idx dictionary
    word2idx_truncated = {t[0]: idx for idx, t in enumerate(sorted_word_count[:vocab_size_limit])}

    # add the mandatory words, i.e. the words we don't want to fall under OTHER, to the word2idx dictionary
    idx = max(word2idx_truncated.values()) + 1
    for word in mandatory_words:
        if word not in word2idx_truncated.keys():
            word2idx_truncated[word] = idx
            idx += 1

    # finally, put all the other words under OTHER
    word2idx_truncated["OTHER"] = idx
    sentences_truncated = []

    # create the truncated version of the sentence list
    for sentence in sentences:
        sentence_truncated = []
        for word_idx in sentence:
            if idx2word[word_idx] in word2idx_truncated:
                sentence_truncated.append(word2idx_truncated[idx2word[word_idx]])
            else:
                sentence_truncated.append(word2idx_truncated["OTHER"])

        sentences_truncated.append(sentence_truncated)

    # return truncated values
    return sentences_truncated, word2idx_truncated
