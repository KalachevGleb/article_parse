import abc

# import tools for natural language processing
import json
import math
import os
import pickle
import re
from collections import defaultdict
from copy import copy
from typing import List, Tuple, Optional, Union, Dict

import yaml
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import *

from parse_text import tokenize_text_nltk, parse_grammar, grammar_str, math_env_names

# statistics class for ngrams
# contains following stats:
# - ngrams by parts of speech obtained from nltk (number of occurences of each ngram)
# - for each i in 1...n each part of ngram(w[1], w[2], w[i-1], w[i+1],..., w[n]) stats of w[i] by parts of speech
# - for each word x stats of (w[1], w[2], w[i-1], x, w[i+1],..., w[n])
#        for different i for different parts of speech w[1], w[2],..., w[n]
from texdocument import TexDocument, TextFragment


class NgramStats:
    def __init__(self, n: int):
        self.words_stats = None
        self.n = n
        self.ngrams = defaultdict(lambda: 0)
        self.ngrams_by_pos = defaultdict(lambda: defaultdict(lambda: 0))
        self.ngrams_by_word = defaultdict(lambda: defaultdict(lambda: 0))
        self.words = defaultdict(lambda: 0)

    # add ngram to stats
    def add_ngram(self, ngram: List[Tuple[str, str]]):  # ngram is a list of tuples (word, part of speech)
        parts_of_speech = [ngram[i][1] for i in range(self.n)]  # type: List[Optional[str]]
        self.ngrams[tuple(parts_of_speech)] += 1
        for i in range(self.n):
            parts_of_speech[i] = None
            self.ngrams_by_pos[tuple(parts_of_speech)][ngram[i][1]] += 1
            self.ngrams_by_word[ngram[i][0]][tuple(parts_of_speech)] += 1
            parts_of_speech[i] = ngram[i][1]

    def add_tokenized_text(self, text: List[List[Tuple[str, str]]]):
        for sentence in text:
            for i in range(len(sentence) - self.n):
                self.add_ngram(sentence[i:i + self.n])
            for i in range(len(sentence)):
                self.words[sentence[i][0]] += 1

    def collect_stats_for_words(self, excluded_ps=('EQN', 'EQNP', 'SOL', 'EOL')):
        self.words_stats = defaultdict(lambda: [])
        for word in self.words:
            stats = defaultdict(lambda: 0)
            for ngram, num in self.ngrams_by_word[word].items():
                ngram_stats = defaultdict(lambda: 0)
                total = 0
                for ps, num_ps in self.ngrams_by_pos[ngram].items():
                    if ps not in excluded_ps and ps[0].isalpha():
                        ngram_stats[ps] += num_ps
                        total += num_ps
                for ps, num_ps in ngram_stats.items():
                    if num_ps > 0:
                        stats[ps] += math.log(num_ps)  # ?? do we need to divide by total?
            max_stat = max(stats.values())
            sub = sum(math.exp(stat - max_stat) for stat in stats.values())
            probs = sorted([(key, math.exp(stat - max_stat) / sub) for key, stat in stats.items()],
                           reverse=True, key=lambda x: x[1])
            self.words_stats[word] = probs

        return self.words_stats

    def save_stats(self, path=None):
        path = path or f'word_stats_{self.n}.pickle'
        with open(path, 'wb') as f:
            pickle.dump(self.words_stats, f)

    def load_stats(self, path=None):
        path = path or f'word_stats_{self.n}.pickle'
        with open(path, 'rb') as f:
            self.words_stats = pickle.load(f)
        return self

    def print_stats(self, word):
        print(f'{word}:')
        ps_width = max(len(ps) for ps in self.words_stats[word])
        for ps, prob in self.words_stats[word]:
            print(f'{ps:<{ps_width}}: {prob:.3f}')


def prepare_tex_file_contents(filename):
    document = TexDocument(filename=filename)
    # collect text and math environments
    text_segments = []
    eqn_counter = 0
    for env in document.items_and_envs([TextFragment], math_env_names):
        if isinstance(env, TextFragment):
            text_segments.append(env.remove_formatting_macros())
        else:
            eqn_counter += 1
            text_segments.append(f"equation_{eqn_counter}")
            last_frag = env.items[-1]
            if isinstance(last_frag, TextFragment) and last_frag.text.count('.'):
                text_segments.append('.')

    return " ".join(text_segments)


# collect ngram stats from file
def collect_ngram_stats(n: int, path: str):
    stats = NgramStats(n)
    if path.endswith('.tex'):
        contents = prepare_tex_file_contents(path)
    else:
        with open(path, 'r') as f:
            contents = f.read()
    tokenized = tokenize_text_nltk(contents)
    print(f'{len(tokenized)} sentences; collecting stats for {n}-grams')
    stats.add_tokenized_text(tokenized)
    print('collecting stats for words')
    stats.collect_stats_for_words()
    print('done')
    return stats
