import csv
import json
from collections import Counter

max_len = 95

pairs = []
with open('nasbench201_corpus.csv', 'r') as f:
    lines = csv.reader(f)
    for line in lines:
        pairs.append(line)

word_freq = Counter()
for pair in pairs:
    word_freq.update(pair[0])
    word_freq.update(pair[1])

min_word_freq = 5
words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
word_map = {k: v + 1 for v, k in enumerate(words)}
word_map['<unk>'] = len(word_map) + 1
word_map['<start>'] = len(word_map) + 1
word_map['<end>'] = len(word_map) + 1
word_map['<pad>'] = 0

print('Total words are: {}'.format(len(word_map)))

with open('WORDMAP_NB201_corpus.json', 'w') as j:
    json.dump(word_map, j)


def encode_question(words, word_map):
    enc_c = [word_map.get(word, word_map['<unk>'])
             for word in words] + [word_map['<pad>']] * (
                 max_len - len(words))
    return enc_c


def encode_reply(words, word_map):
    enc_c = [word_map['<start>']] + [
        word_map.get(word, word_map['<unk>']) for word in words
    ] + [word_map['<end>']] + [word_map['<pad>']] * (
        max_len - len(words))
    return enc_c


pairs_encoded = []
for pair in pairs:
    qus = encode_question(pair[0], word_map)
    ans = encode_reply(pair[1], word_map)
    pairs_encoded.append([qus, ans])

with open('pairs_encoded.json', 'w') as p:
    json.dump(pairs_encoded, p)
