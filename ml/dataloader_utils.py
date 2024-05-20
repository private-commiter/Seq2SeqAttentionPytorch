import os
import re
import random
import unicodedata
from io import open


# SOS_token = 0
# EOS_token = 1
# MAX_LENGTH = 10

# eng_prefixes = (
#     "i am ", "i m ",
#     "he is", "he s ",
#     "she is", "she s ",
#     "you are", "you re ",
#     "we are", "we re ",
#     "they are", "they re "
# )


class Lang:
    def __init__(self, name, sos_token, eos_token):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {sos_token: "SOS", eos_token: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


class TextPreprocessor:
    def __init__(self, data_dir, max_length, sos_token, eos_token):
        self._data_dir = data_dir
        self._max_length = max_length
        self._eng_prefixes = (
            "i am ", "i m ",
            "he is", "he s ",
            "she is", "she s ",
            "you are", "you re ",
            "we are", "we re ",
            "they are", "they re "
        )
        self._sos_token = sos_token
        self._eos_token = eos_token

    def _readLangs(self, lang1, lang2, reverse=False):
        print("Reading lines...")

        # Read the file and split into lines
        file_path = os.path.join(self._data_dir, f'{lang1}-{lang2}.txt')
        lines = open(file_path, encoding='utf-8').\
            read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [[normalizeString(s) for s in line.split('\t')] for line in lines]

        # Reverse pairs, make Lang instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(lang2, self._sos_token, self._eos_token)
            output_lang = Lang(lang1, self._sos_token, self._eos_token)
        else:
            input_lang = Lang(lang1, self._sos_token, self._eos_token)
            output_lang = Lang(lang2, self._sos_token, self._eos_token)

        return input_lang, output_lang, pairs

    def _filterPair(self, p):
        return len(p[0].split(' ')) < self._max_length and \
            len(p[1].split(' ')) < self._max_length and \
            p[1].startswith(self._eng_prefixes)

    def _filterPairs(self, pairs):
        return [pair for pair in pairs if self._filterPair(pair)]

    def prepareData(self, lang1, lang2, reverse=False):
        input_lang, output_lang, pairs = self._readLangs(lang1, lang2, reverse)
        print("Read %s sentence pairs" % len(pairs))
        pairs = self._filterPairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            input_lang.add_sentence(pair[0])
            output_lang.add_sentence(pair[1])
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs


# if __name__ == '__main__':
#     input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
#     print(random.choice(pairs))
