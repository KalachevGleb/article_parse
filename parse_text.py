import abc

# import tools for natural language processing
import json
import multiprocessing
import os
import pickle
import re
from collections import defaultdict
from copy import copy
from typing import List, Tuple, Optional, Union, Dict

import yaml
from nltk.stem import *


# get infinitive form of a word using nltk
from texdocument import TexDocument, TextFragment


def get_infinitive(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word)


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

stop_words = set(stopwords.words('english'))


# replaces $ <formula> $ or $$ <formula> $$ or \[ <formula> \] or \begin{equation} <formula> \end{equation} by formula_i
# where i is the number of the formula
def replace_tex_formulas(text):
    # first find \begin{document} and \end{document} and take text between them
    begin_document = text.find(r'\begin{document}')
    end_document = text.find(r'\end{document}')
    if begin_document != -1 and end_document != -1:
        text = text[begin_document + len(r'\begin{document}'):end_document]
    # remove tex comments:  % ... \n where % is not escaped
    text = re.sub(r'(?<!\\)%.*?\n', '', text)
    text = text.replace(r'\%', '%')
    # remove tex comments of the form \iffalse ... \fi
    text = re.sub(r'\\iffalse.*\\fi', '', text)
    # remove tables and figures
    text = re.sub(r'\\begin[{]table[}].*?\\end[{]table[}]', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin[{]figure[}].*?\\end[{]figure[}]', '', text, flags=re.DOTALL)
    # remove labels
    text = re.sub(r'\\label[{].*?[}]', '', text)
    # replace ~ by space
    text = text.replace('~', ' ')
    # regular expression for tex formulas
    tex_formula_regex = re.compile(r'(\$(([^$]|\\\$)+)\$-?|\$\$([^$]+)\$\$|\\\[((?:[^\\]|\\[^\]])*)\\\]|'
                                   r'\\begin[{](equation|align|multline|eqnarray)[*]?[}](.*)'
                                   r'\\end[{](equation|align|multline|eqnarray)[*]?[}])', re.DOTALL)
    # replace formulas
    fragments = []
    formulas = {}
    prev_end = 0
    for match in tex_formula_regex.finditer(text):
        fragments.append(text[prev_end:match.start()])
        fragments.append("formula___" + str(len(formulas)))
        formula = text[match.start():match.end()]
        formulas["formula___" + str(len(formulas))] = formula
        znak = ''
        for c in formula[::-1]:
            if c == '}' or c.isalnum():
                break
            elif c in '.,;:':
                znak = c
                break
        if znak:
            fragments[-1] += ' ' + znak
        elif '.' in formula or formula[-1] != '$' or formula[-2] == '$':
            for j in range(match.end(), len(text)):
                if not text[j].isspace():
                    if text[j].isalpha() and text[j].isupper():
                        fragments[-1] += ' .'
                    break
        prev_end = match.end()
    fragments.append(text[prev_end:])
    return ' '.join(fragments), formulas


# text efficient multiple replacement
def replace_multiple(text: str, replacements):
    return re.subn('|'.join(re.escape(k) for k in replacements.keys()), lambda m: replacements[m.group(0)], text)


with open('dict_2-4gram_10000-f.yml', 'r') as pos_dict_file:
    pos_dict = yaml.load(pos_dict_file, Loader=yaml.CSafeLoader)


dict_pos_set = set(sum((list(v.keys()) for v in pos_dict.values()), []))


# sent_tokenize is one of instances of
# PunktSentenceTokenizer from the nltk.tokenize.punkt module
def tokenize_text_nltk(txt):
    res = []
    txt, formulas = replace_tex_formulas(txt)
    # remove all braces
    replacements = {'{': '', '}': '', 'i.e.': 'ie', 'e.g.': 'eg', 'et al.': 'et_al',
                    'Fig.': 'Fig', 'fig.': 'Fig', 'Ref.': 'Ref', 'ref.': 'ref', 'Eq.': 'Eq', 'eq.': 'Eq',
                    'resp.': 'respectively'}
    txt, _ = replace_multiple(txt, replacements)
    special_tokens = {'ie': 'IE', 'eg': 'EG', 'et_al': 'ET_AL'}

    tokenized = sent_tokenize(txt)
    # Word tokenizers is used to find the words
    # and punctuation in a string
    wordlists = [word_tokenize(x) for x in tokenized]
    for i, wordlist in enumerate(wordlists):
        while wordlist[0].startswith('footnote_'):
            wordlists[i-1].append(wordlist[0])
            wordlists[i] = wordlist = wordlist[1:]

    for wordslist in wordlists:
        # remove footnotes from wordlist and save positions of footnotes
        footnote_pos = [i for i, x in enumerate(wordslist) if x.startswith('footnote_')]
        wordslist1 = [x for i, x in enumerate(wordslist) if not x.startswith('footnote_')]

        # Using a Tagger. Which is part-of-speech
        # tagger or POS-tagger.
        tagged = nltk.pos_tag(wordslist1)
        # insert footnotes back into tagged list
        for i in footnote_pos:
            tagged.insert(i, (wordslist[i], 'FOOTNOTE'))
        if tagged[-1][1] in ['.', '?', '!']:
            tagged[-1] = (tagged[-1][0], 'EOL')
        else:
            tagged.append(('', 'EOL'))
        # make first letter of first word lowercase if not NNP
        if not tagged[0][1].startswith('NNP'):
            if tagged[0][0][1:].islower():
                tagged[0] = (tagged[0][0][0].lower() + tagged[0][0][1:], tagged[0][1])
            else:
                tagged[0] = (tagged[0][0], 'NNP')

        for i, (word, tag) in enumerate(tagged):
            if word and word[0].isupper() and not tag.startswith('NNP'):
                tagged[i] = (word, 'NNP')
            elif word in special_tokens:
                tagged[i] = (word, special_tokens[word])
            elif tag in dict_pos_set and word in pos_dict and pos_dict[word].get(tag, 0) < 5:
                best_tag = max(pos_dict[word].items(), key=lambda x: x[1])[0]
                tagged[i] = (word, best_tag)

        tagged = [('', 'SOL')] + tagged
        # find and replace formulas in tagged list
        for i in range(len(tagged)):
            if tagged[i][0].startswith('formula___'):
                formula = formulas[tagged[i][0]]
                if formula.endswith('-'):
                    tagged[i] = (formula, 'EQNP')
                else:
                    tagged[i] = (formulas[tagged[i][0]], 'EQN')
            elif tagged[i][0].startswith('equation_'):
                tagged[i] = (tagged[i][0], 'EQN')

        res.append(tagged)
    return res


class DataBase:
    def __init__(self):
        self.db = {}
        self.relations = {}


class PatternBase(abc.ABC):
    def __repr__(self):
        return str(self)

    @abc.abstractmethod
    def match(self, words, start_index):
        raise NotImplementedError()


class PatternVariable(PatternBase):
    def __init__(self, name, type=None):
        self.name = name
        self.type = {type} if isinstance(type, str) else set(type) if type is not None else None

    def __str__(self):
        return f'{self.name}: {self.type}'

    def match(self, words, start_index):
        if self.type is None or words[start_index][1] in self.type:
            yield (self.name, words[start_index]), start_index + 1


class PatternConst(PatternBase):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f'{self.value}'

    def match(self, words, start_index):
        if words[start_index][0] == self.value:
            yield None, start_index + 1


def compose_generators(generator_lambdas, start_index):
    if len(generator_lambdas) == 1:
        for result, index in generator_lambdas[0](start_index):
            yield (result,), index
    for result, index in generator_lambdas[0](start_index):
        for rest, last_index in compose_generators(generator_lambdas, index):
            yield result + rest, last_index


class PatternCanSkipWords(PatternBase):
    def __init__(self, max_skip_words=1, min_skip_words=0):
        self.min_skip_words = min_skip_words
        self.max_skip_words = max_skip_words

    def __str__(self):
        if self.max_skip_words is ...:
            return '...'
        return f'skip(<={self.max_skip_words})'

    def match(self, words, start_index):
        if self.max_skip_words is ...:
            for i in range(start_index + self.min_skip_words, len(words)):
                yield None, i
        else:
            for i in range(start_index + self.min_skip_words, min(start_index + self.max_skip_words, len(words))):
                yield None, i


# Class for natural language pattern
class PatternSequence(PatternBase):
    def __init__(self, elements: List[PatternBase]):
        self.elements = elements

    def __str__(self):
        return ' '.join(str(e) for e in self.elements)

    def match(self, words, start_index):
        for p, last_index in compose_generators([lambda index: e.match(words, index) for e in self.elements],
                                                start_index):
            yield {x[0]: x[1] for x in p if x is not None}, last_index


class Relation:
    def __init__(self, name, objects: dict):
        self.name = name
        self.objects = sorted(objects.items())

    def __str__(self):
        return f'{self.name}({self.objects})'


class RelationStats:  # class for statistics of relations
    def __init__(self):
        self.relation = None
        self.count = 0

    def add(self, relation):
        if self.relation is None:
            self.relation = relation
        self.count += 1


def parse_natural_sentense(sentence, relations: defaultdict, patterns: List[Tuple[str, PatternBase]]):
    words = sentence.split(' ')
    for (rel, pattern) in patterns:
        for result, last_index in pattern.match(words, 0):
            if result:
                r = Relation(rel, result)
                relations[str(r)].add(r)
    return len(relations)


def parse_pattern_item(item: str):
    if all(x == '_' for x in item):
        return PatternCanSkipWords(len(item))
    if item == '...':
        return PatternCanSkipWords(...)
    if item.endswith('_'):
        return PatternVariable(item[:-1])
    if '_' in item:
        v, suffix = item.split('_')
        return PatternVariable(v, suffix)
    return PatternConst(item)


def parse_pattern(pattern: str):
    return PatternSequence([parse_pattern_item(x) for x in pattern.split(' ')])


class ParseTreeNode:
    __slots__ = ['applied_rules', 'metadata', 'level', 'rule_num']

    def __init__(self):
        self.applied_rules = set()
        self.metadata = {}
        self.level = 0
        self.rule_num = 0

    def print_tree(self, prefix='', last=True, is_main=False):
        print(self.str_for_print(prefix, last, is_main), end='')

    def str_for_print(self, prefix='', last=True, is_main=False):
        return ''

    def __repr__(self):
        return str(self)

    @abc.abstractmethod
    def get_word(self) -> str:
        pass


# Token class, contains word and metadata from nltk tokenizer
class Token(ParseTreeNode):
    def __init__(self, word, pos, metadata):
        super().__init__()
        self.word = word
        self.pos = pos
        self.metadata = copy(metadata)
        self.metadata['leaf'] = 1

    def __str__(self):
        return f'{self.word}({self.metadata})'

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.word == other.word and self.metadata == other.metadata

    @property
    def token(self):
        return self

    def str_for_print(self, prefix='', last=True, is_main=False):
        return prefix + ('└──' if last else '├──') + ('*' if is_main else ' ') + str(self.word) +\
               '  (' + str(self.metadata) + ')\n'

    def is_leaf(self):
        return True

    def get_word(self):
        return self.word

    def __iter__(self):
        yield self


# Parse tree for natural language;
# each leaf is a token (with meta data)
# each internal node is a rule applied to a list of children; is can have main token
class ParseTree(ParseTreeNode):
    def __init__(self, token, rule=None, children=None, metadata=None, main_branch=None, level=None, rule_num=None):
        super().__init__()
        self.token: Optional[Token] = token if isinstance(token, Token) else None
        self.metadata = {} if token is None else copy(token.metadata if main_branch is None else children[main_branch].metadata)
        self.metadata['leaf'] = 0
        if metadata is not None:
            self.metadata.update(metadata)
        self.rule = rule
        self.main_branch = main_branch
        self.children = children if children is not None else []
        self.level = level
        self.rule_num = rule_num

    def __str__(self):
        if not self.children:
            return str(self.token)
        else:
            # if self.rule:
            #    return f'{self.rule}({",".join(map(str, self.children))})'
            # else:
            return f'({",".join(map(str, self.children))})'

    @property
    def main_child(self):
        return self.children[self.main_branch] if self.main_branch is not None else None

    def __iter__(self):
        yield self
        for c in self.children:
            yield from c

    def __len__(self):
        return 1 + sum(len(c) for c in self.children)

    def __getitem__(self, index):
        if self.rule is None:
            return self.token
        else:
            return self.children[index]

    def str_for_print(self, prefix='', last=True, is_main=False):
        node_str = f'[{self.token.word}]' if self.token else ''
        node_str += str(self.metadata) + '    \trule = ' + str(self.rule) + f'({self.level})'
        res = prefix + ('└──' if last else '├──') + ('*' if is_main else ' ') + node_str + '\n'
        for i, c in enumerate(self.children):
            res += c.str_for_print(prefix + ('  ' if last else '│  '), i == len(self.children) - 1, is_main=i == self.main_branch)
        return res

    def is_leaf(self):
        return self.metadata.get('leaf', 0) == 1

    def get_word(self):
        #return self.token.word if self.token and not self.is_leaf() else ' '.join(c.get_word() for c in self.children)
        return self.token.word if self.token else ' '.join(c.get_word() for c in self.children)

#
# def remove_tree_node_right(path_to_node: List[ParseTree]):
#     root = path_to_node[0]
#     if len(path_to_node) == 1:
#         return [root]
#     elif path_to_node[1] is root.main_child:
#         r = remove_tree_node_right(path_to_node[1:])
#
#
#
# def remove_tree_node(path_to_root: List[ParseTree]):
#     curr = path_to_root[0]
#     left_nodes = []
#     right_nodes = []
#     for i, c in enumerate(path_to_root[1:]):
#         main_branch = c.children[c.main_branch]
#         #rest_branches = c.children[:c.main_branch] + c.children[c.main_branch + 1:]
#         #pos = next(i for i, ch in enumerate(rest_branches) if ch is curr)
#         right_nodes.extend(c.children[c.main_branch + 1:])
#         left_nodes = c.children[:c.main_branch] + left_nodes
#


# Base class for grammar rule item.
class RuleItem(abc.ABC):
    @abc.abstractmethod
    def match(self, node: ParseTreeNode):
        raise NotImplementedError()


class RuleItemWithTags(RuleItem):
    def __init__(self, tags: dict):
        self.tags = {}
        self.replace_tags = {}
        if tags is not None:
            for tag, value in tags.items():
                if isinstance(value, str):
                    if value[0] == '=':
                        value = yaml.safe_load(value[1:])
                        self.replace_tags[tag] = value
                        continue
                    elif len(parts:=value.split('->')) > 1:
                        if len(parts) > 2:
                            raise ValueError(f'Invalid tag value {value}')
                        self.tags[tag] = yaml.safe_load(parts[0])
                        self.replace_tags[tag] = yaml.safe_load(parts[1])
                        continue
                self.tags[tag] = value

    # checks whether word tags match with this rule item
    def match(self, node: ParseTreeNode):
        return all(match_tag(node.metadata.get(tag, None), value) for tag, value in self.tags.items())

    def replace(self, node: ParseTreeNode):
        for tag, value in self.replace_tags.items():
            node.metadata[tag] = value


# Class for grammar rule item, which is constant word
class RuleItemConst(RuleItemWithTags):
    def __init__(self, word, tags=None):
        super().__init__(tags)
        self.word = {word} if isinstance(word, str) else set(word)

    def __str__(self):
        return '|'.join(self.word)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.word == other.word and self.tags == other.tags

    def match(self, node: ParseTreeNode):
        return node.get_word() in self.word and super().match(node)


def match_tag(tag, value):
    if isinstance(value, str):
        return tag == value
    if hasattr(value, '__iter__'):
        return tag in value
    return tag == value


# Class for grammar rule item, which is word with some constraints on tags
class RuleItemVariable(RuleItemWithTags):
    def __init__(self, name, tags: dict):
        super().__init__(tags)
        self.name = name

    def __str__(self):
        return f'{self.name}:{self.tags}'

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.tags == other.tags


class NegateRuleItem(RuleItem):
    def __init__(self, item: RuleItem):
        self.item = item
        if isinstance(item, RuleItemWithTags) and item.replace_tags:
            raise ValueError('! cannot be used with tag replacement')

    def __str__(self):
        return f'!{self.item}'

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, NegateRuleItem) and self.item == other.item

    def match(self, node: ParseTreeNode):
        return not self.item.match(node)


class GrammarRule:
    def __init__(self, name, items: List[RuleItem], main_token=None, metadata=None, interval=None, subtree=False, destruct=False, use_priority=False, text=""):
        self.text = text
        self.name = name
        self.items = items
        self.destruct = destruct
        self.main_token = main_token
        self.metadata = copy(metadata) if metadata is not None else {}
        self.interval = interval if interval is not None else (0, len(items))
        self.subtree = 0
        self.use_priority = use_priority
        if subtree:  # if main_token is first, then the first item can be a subtree
            if main_token == 0:
                self.subtree = 1  # 1 means left
            elif main_token == len(items) - 1:
                self.subtree = 2
            else:
                raise ValueError(
                    f'Subtree can only be first or last item but main_token={main_token} is not first or last')

        assert self.interval[0] >= 0 and self.interval[1] <= len(items)
        if self.main_token is None:
            if self.interval[1] - self.interval[0] == 1:
                self.main_token = self.interval[0]
            elif any(not isinstance(item, RuleItemConst) for item in self.items[self.interval[0]:self.interval[1]]):
                raise ValueError(f'Grammar rule {self.name} has no main token')
        if self.main_token is not None:
            assert self.interval[0] <= self.main_token < self.interval[1]

        if self.subtree and self.is_tag_add_rule():
            raise Exception(f"Subtree can not be used with tag add rule {self}")

    def __str__(self):
        if self.text:
            return self.text
        res = " ".join([("[" if i == self.interval[0] else "") + str(item) + ("*" if i == self.main_token else "") + (
            "]" if i == self.interval[1] - 1 else "") for i, item in enumerate(self.items)])
        if self.name:
            res = f'{self.name}({res})'
        if self.metadata:
            res += str(self.metadata)
        if self.subtree:
            res = '^' + res
        return res

    def __repr__(self):
        return str(self)

    def is_tag_add_rule(self):
        if self.interval[1] - self.interval[0] > 1:
            return False
        if self.main_token is None:
            return False
        return True

    # Update metadata and returns true if metadata changed
    def add_tags(self, node: ParseTreeNode) -> bool:
        old_metadata = copy(node.metadata)
        node.metadata.update(self.metadata)
        return old_metadata != node.metadata

    def replace_items_tags(self, nodes: List[ParseTreeNode]):
        assert len(self.items) == len(nodes)
        for item, node in zip(self.items, nodes):
            if isinstance(item, RuleItemWithTags):
                item.replace(node)


# Grammar class, consists of several sets of rules, each set has its level
class Grammar:
    def __init__(self, rules: List[Tuple[int, GrammarRule]]):
        self.num_levels = max(x[0] for x in rules) + 1
        self.rules = [[] for _ in range(self.num_levels)]
        for level, rule in rules:
            self.rules[level].append(rule)

    def __str__(self):
        lines = []
        for level in range(self.num_levels):
            lines.append(f'# level {level}')
            for rule in self.rules[level]:
                lines.append(f'\t{rule}')
        return '\n'.join(lines)

    def __repr__(self):
        return str(self)

    def __iter__(self):
        yield from self.rules

    def __getitem__(self, index) -> List[GrammarRule]:
        return self.rules[index]


def parse_sentense(tokens, grammar, debug=True) -> List[ParseTreeNode]:
    # create parse tree for each token
    parse_trees = [Token(word, i, {'ps': part_of_speech}) for i, (word, part_of_speech) in enumerate(tokens)]

    for level in range(0, grammar.num_levels):
        if debug:
            print(f'level {level}')
        changed = True
        while changed:
            changed = False
            for rule_num, rule in enumerate(grammar[level]):
                #if rule.text == "(is|are|am|was|were|can|could|will|would|have|has|had|did|does|do* not){neg:1, leaf:1}":
                #    print(rule.text)
                for i in range(len(parse_trees) - len(rule.items) + 1):
                    if rule.subtree == 1:
                        curr = parse_trees[i]
                        left_branch = [curr]
                        while not curr.is_leaf():
                            curr = curr.children[-1]
                            left_branch.append(curr)

                        found = None
                        for idx, prev, curr in zip(range(len(left_branch)), left_branch[0:-1], left_branch[1:]):
                            # take into account rule priority
                            if rule.use_priority and found and (prev.level, prev.rule_num) < (level, rule_num):
                                break
                            if rule.items[0].match(curr) and all(x.match(parse_trees[i + j + 1]) for j, x in enumerate(rule.items[1:])):
                                found = (prev, curr, idx)
                        if found:
                            prev, curr, i0 = found
                            b, e = rule.interval
                            assert b == 0
                            rule.replace_items_tags([curr] + parse_trees[i + 1:i + len(rule.items)])
                            new_tree = ParseTree(curr.token, str(rule), [curr] + parse_trees[i + 1:i + e],
                                                 metadata=rule.metadata, main_branch=rule.main_token - b, level=level, rule_num=rule_num)
                            if debug:
                                print(f'apply rule {rule} to {[curr] + parse_trees[i + 1:i + e]}: {new_tree}')
                            prev.children[-1] = new_tree
                            del parse_trees[i + 1:i + e]
                            for node, next in zip(left_branch[i0::-1], left_branch[i0 + 1::-1]):
                                if node.main_child is next:
                                    for tag in next.metadata:
                                        if tag not in node.metadata:
                                            node.metadata[tag] = next.metadata[tag]
                            changed = True
                            break
                    elif rule.subtree == 2:
                        curr = parse_trees[i + len(rule.items) - 1]
                        right_branch = [curr]
                        while not curr.is_leaf():
                            curr = curr.children[0]
                            right_branch.append(curr)

                        found = None
                        for idx, prev, curr in zip(range(len(right_branch)), right_branch[0:-1], right_branch[1:]):
                            # take into account rule priority
                            if rule.use_priority and found and (prev.level, prev.rule_num) < (level, rule_num):
                                break
                            if rule.items[-1].match(curr) and all(x.match(parse_trees[i + j]) for j, x in enumerate(rule.items[:-1])):
                                found = (prev, curr, idx)
                        if found:
                            prev, curr, i0 = found
                            b, e = rule.interval
                            assert e == len(rule.items)
                            rule.replace_items_tags(parse_trees[i:i + e - 1] + [curr])
                            new_tree = ParseTree(curr.token, str(rule), parse_trees[i + b:i + e - 1] + [curr],
                                                 metadata=rule.metadata, main_branch=rule.main_token - b, level=level, rule_num=rule_num)
                            if debug:
                                print(f'apply rule {rule} to {parse_trees[i + b:i + e - 1] + [curr]}: {new_tree}')
                            prev.children[0] = new_tree
                            del parse_trees[i + b:i + e - 1]
                            for node, next in zip(right_branch[i0::-1], right_branch[i0 + 1::-1]):
                                if node.main_child is next:
                                    for tag in next.metadata:
                                        if tag not in node.metadata:
                                            node.metadata[tag] = next.metadata[tag]
                            changed = True
                            break

                    if all(x.match(parse_trees[i + j]) for j, x in enumerate(rule.items)):
                        if rule.main_token is None:
                            b, e = rule.interval
                            if all(isinstance(x, Token) for x in parse_trees[i + b:i + e]):
                                new_token = Token(' '.join(x.get_word() for x in parse_trees[i + b:i + e]),
                                                  parse_trees[i + b].pos, metadata=rule.metadata)
                                if debug:
                                    print(f'apply rule {rule} to {parse_trees[i + b:i + e]}: {new_token}')
                                parse_trees[i + b:i + e] = [new_token]
                                changed = True
                                break
                        elif rule.is_tag_add_rule():
                            rule_key = str(rule)
                            if rule_key not in parse_trees[i + rule.interval[0]].applied_rules:
                                parse_trees[i + rule.interval[0]].applied_rules.add(rule_key)
                                rule.replace_items_tags(parse_trees[i:i + len(rule.items)])
                                subtree = parse_trees[i + rule.interval[0]]
                                if debug:
                                    print(f'add tags ({rule}) {rule.metadata} to {subtree}: ', end='')
                                tags_changed = rule.add_tags(subtree)
                                changed = tags_changed or changed
                                if debug:
                                    print(subtree, f'  (changed={tags_changed})')
                        else:
                            b, e = rule.interval
                            rule.replace_items_tags(parse_trees[i:i + len(rule.items)])
                            new_tree = ParseTree(parse_trees[i + rule.main_token].token, str(rule),
                                                 parse_trees[i + b:i + e],
                                                 metadata=rule.metadata, main_branch=rule.main_token - b, level=level, rule_num=rule_num)
                            if debug:
                                print(f'apply rule {rule} to {parse_trees[i + b:i + e]}: {new_tree}')
                            parse_trees[i + b:i + e] = [new_tree]
                            changed = True
                            break
                if changed:
                    if debug:
                        print('changed')
                    break
    return parse_trees


def parse_item(item: str) -> Tuple[RuleItem, bool]:
    if item[0] == '!':
        it = parse_item(item[1:])
        return NegateRuleItem(it[0]), it[1]
    selected = False
    if item.endswith('*'):
        selected = True
        item = item[:-1]
    if item.count('_'):
        pos_ = item.index('_')
        v = item[:pos_]
        suffix = item[pos_ + 1:]
        if suffix.count('{'):
            suffix = suffix.replace(':', ': ')
            pos = suffix.index('{')
            tags = yaml.safe_load(suffix[pos:])
            suffix = suffix[:pos]
            if '|' in suffix:
                suffix = set(suffix.split('|'))
            if suffix:
                tags['ps'] = suffix
            return RuleItemVariable(v, tags), selected
        if '|' in suffix:
            suffix = set(suffix.split('|'))
        return RuleItemVariable(v, {'ps': suffix} if suffix else {}), selected
    else:
        tags = {}
        if item.count('{'):
            item = item.replace(':', ': ')
            pos = item.index('{')
            tags = yaml.safe_load(item[pos:])
            item = item[:pos]
        return RuleItemConst(item.split('|'), tags=tags), selected


def parse_rule(rule: str):
    text = rule.strip()
    if rule[0] == '^':
        rule = rule[1:].strip()
        subtree = True
    else:
        subtree = False
    metadata = {}
    rule = rule.replace('[', ' [ ')
    rule = rule.replace(']', ' ] ')
    rule_pattern = re.compile(r'^([^()]*)\(((?:[^()]|\\\(|\\\))*[^\\])\)(.*)$')
    match = rule_pattern.match(rule)
    if match:
        name = match.group(1).strip()
        items = match.group(2)
        suf = match.group(3).strip()
        if suf and suf[0] == '{':
            suf = suf.replace(':', ': ')
            metadata = yaml.safe_load(suf)
    else:
        name = ''
        items = rule
    items = items.replace('\\(', '(')
    items = items.replace('\\)', ')')
    items = items.split()
    cntopen = items.count('[')
    cntclose = items.count(']')
    if cntopen != cntclose or cntopen > 1:
        raise ValueError(f'Invalid rule {rule}: mismatched brackets')
    # remove brackets and store their positions
    open_pos = 0
    close_pos = len(items)
    if cntopen == 1:
        open_pos = items.index('[')
        close_pos = items.index(']')
        items = items[:open_pos] + items[open_pos + 1:close_pos] + items[close_pos + 1:]
        close_pos -= 1

    items = [parse_item(item) for item in items]
    pos = [i for i, x in enumerate(items) if x[1] is True]
    if len(pos) == 0:
        main_token = None
    elif len(pos) == 1:
        main_token = pos[0]
    else:
        raise Exception('Multiple main tokens in rule')
    return GrammarRule(name, [x[0] for x in items], main_token, metadata, interval=(open_pos, close_pos),
                       subtree=subtree, use_priority=True, text=text)


def parse_grammar(grammar: str):
    lines = [line.strip() for line in grammar.split('\n')]
    level = 0
    rules = []
    for line in lines:
        # remove comments
        line = line.split('%')[0].strip()
        # skip empty lines
        if line == '':
            continue
        # '#' means new level
        if line[0] == '#':
            level += 1
        else:  # parse rule
            rules.append((level, parse_rule(line)))
    return Grammar(rules)


grammar_str = r"""
(_SOL [note*] that){ps:VB}
(induces|implies|coincides|corresponds|varies|decreases|increases){ps:VBZ}
(commute|denote|recall|decrease|increase){ps:VB}
(explicit|quantum|interesting){ps:JJ}
(code|fiber|complex|soundness){ps:NN}
(codes){ps:NNS}
(random){ps:JJ}
(at random*){leaf:1}
(in turn){ps:RB}

(whether){ps:WHETHER}
(if){ps:IF}
(a|an|the){ps:ART}
(that){ps:THAT}
(then){ps:THEN}
(there){ps:THERE}
(which){ps:WHICH}
(since){ps:SINCE}
(due to){ps:IN, due_to:1}
(such that){ps:SUCHTHAT}
(of){ps:OF,of_type:1}
(up to){ps:IN}
(because of){ps:IN}

(_{}* _FOOTNOTE){footnote:1}
(@ cite*){ps:CITE, subj:1, aux:1,leaf:1}
(@ ref|eqref*){ps:CD, subj:1, ref:1, aux:1,leaf:1}
(@ cref*){ps:NNP, subj:1, ref:1, aux:1,leaf:1}
(\( _NNP* \)){ps:IABBR}
(_FW _NNP*){leaf:1}
(_NNP* _ET_AL){leaf:1}
(' s*){ps: POS, pos: 1}

(_EQNP* th){ps:JJ,leaf:1}
(_EQNP _{}*){eqnp:1,leaf:1}

(_CD* % ){leaf:1}

(_NN|NNP*){gsp: NN, subj: 1, many: 0}
(_NNS|NNPS*){gsp: NN, subj: 1, many: 1}
(_EQN*){gsp:NN, subj:1}
(is|are|am|do|does*){gsp: VB, whole:0} 
(!is|are|am|was|were|be|been [_VBG*]){subj: 1, many: 0}  % ...ing used as a noun
(is|are|was|were* _VBG{gsp:=VB,subj:=0})
(by [_VBG*]){subj: 0, gsp: VB}
(_VB|VBP|VBD|VBZ*){gsp: VB, whole:0}  % verb
(_{gsp:VB}*){accept_to:1}
(be|been|is|are|am|was|were* _VBD|VBN){passive: 1} 
(is|are|am|was|were|can|could|will|would|have|has|had|did|does|do* not){neg:1, leaf:1} 
(of|in|on|over*){of_type: 1}
(_{gsp:NN}* _POS){nnpos: 1}
(he|she|it|they|we|I|this*){can_p:1}
(_NNP* -|-- _NNP){many_names:1,leaf:1}
(_SOL [_NN|NNS -|-- _NNP*]){ps:NNP,many_names:1,leaf:1}
(_{many_names:1}* _IABBR){new_abbr:1,leaf:1}
(_JJ|DT [one*]){ps:NN,gsp:NN,subj:1,many:0}
(one*){subj:1,many:0}

(for* a long time){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}
(in* fact){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}
(to* this end){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}
(on* the one hand){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}
(on* the other hand){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}
(in* this case){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}
(in* particular){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}
(for* example){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}
([as* follows] :|.){ps:STD,gsp:RB,has_prep:1,leaf:1}
(_SOL [in* words]){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}
(yet another*){leaf:1}
(turn|turns|turned* out){leaf:1}
(that is){ps:IE}
(with respect to){ps:IN}

#
(\( _{}* \)){optional:1}
(_IE{zpt:null}* ,){zpt:1}

# 
_RBS _JJ*
least _JJ*
(_NNP _NNP*){cmpxnnp: 1}
(_CD* -|-- _CD){cdrange:1}
#
_CITE* , _CITE
_CITE* and _CITE
_CITE* , and _CITE
(_NNP _{subj:1,aux:null}*){named: 1}
(_NNP* _CITE){cite: 1}
(_NNP* _CD){numbered: 1}
(_{named:1}* _CD){numbered: 1}
(_{gsp:NN}* due to _NNP|NNPS){author:1}
(more|less|fewer|greater* than _CD|EQN){ps:JJ, compare:1}
(_RB _CD*)

#
(`` _{}* ''){quote:1, leaf:1}
(` _{}* '){quote:1, leaf:1}
(\( see* \)){ps:CITE}
(not _IN*){neg:1}
(at most|least _CD|EQN*){ps:JJ}

(_{gsp:VB,what:null} [that*]){that:can_join}
^(_JJ* , _JJ){comma:1}
^(_JJ* and _JJ){and: 1}
^(_JJ* or _JJ){or: 1}
^(_JJ and _JJ*){and: 1}
^(_JJ or _JJ*){or: 1}

^(_JJ* \( _JJ \)){comment:1}
 
_JJ|JJS|JJR _NN|NNS|NNP|NNPS*
_{nnpos:1} _{subj:1}*
its|his|her|their|our|my _{subj:1}* 

(_{gsp:NN}* _CITE){cite:1}
(_{gsp:VB}* _CITE){cite:1}

(_{gsp:NN}* _EQN){with_formula:1}

^(_NN* \( _NN|NNP \)){comment:1}
^(_NNS* \( _NNS|NNPS \)){comment:1}
^(_{gsp:NN}* \( _{gsp:NN} \)){comment:1}
^(_NN* \( _IE _NN|NNP \)){comment:1}
^(_NNS* \( _IE _NNS|NNPS \)){comment:1}
^(_{gsp:NN}* \( _IE _{gsp:NN} \)){comment:1}
^([_NN* , _IE _NN|NNP] _EOL){comment:1}
^([_NNS* , _IE _NNS|NNPS] _EOL){comment:1}
^([_{gsp:NN}* , _IE _{gsp:NN}] _EOL){comment:1}
^(_NN* , _IE _NN|NNP ,){comment:1}
^(_NNS* , _IE _NNS|NNPS ,){comment:1}
^(_{gsp:NN}* , _IE _{gsp:NN} ,){comment:1}

_VBN _NN|NNS*

(_{gsp:NN} _{gsp:NN,the:null}*){no_of:1}
^(_{gsp:NN} _{gsp:NN,pod:1,the:null}*){no_of:1}
(_{subj:1}* _IABBR){new_abbr: 1}

(both _VBG|NN|NNS|NNP|NNPS{many:1}*){gsp:NN,both:1,subj:1}
(_DT|PDT _VBG|NN|NNS|NNP|NNPS*){gsp:NN,subj:1}
(the _NN|NNS|EQN*){the:1}
(the _NNP|NNPS*){the:1, err:1}
(a|an _NN|NNP|EQN*){the:0}
(a|an _NNS|NNPS*){the:0, err:1}
(some _NN|NNS*){exists:1,quant:1}
(some|any|all|each|many _EQN*){quant:1}
(some|any|all|each|many|one of _{subj:1}*){quant:1}
(some|any|all|each|many|one of us|them*){quant:1,subj:1,gsp:NN}
(for [all _{subj:1,many:1}*]){all:1,quant:1}
(for [any|each|every _{subj:1}*]){all:1,quant:1}
(the _{subj:1}*){the:1}


_{subj:1}* _VBN|VBG{subj_last:1}

(in* comparison to){ps:null,join_right:subj}

(and* , _RB{leaf:1} ,)
(or* , _RB{leaf:1} ,)

%(!and|or [_NN|NNS|NNP|NNPS{and:null,or:null} , _NN|NNS|NNP|NNPS*]){comma: 1}
(_NN|NNS|NNP|NNPS{comma:1,and:null,or:null} , and _NN|NNS|NNP|NNPS*){and: 1, many: 1}
(_NN|NNS|NNP|NNPS{comma:1,and:null,or:null} , or _NN|NNS|NNP|NNPS*){or: 1}
^(!and|or [_NN|NNS|NNP|NNPS{the:null} and _NN|NNS|NNP|NNPS{the:null}*]){and: 1, many: 1}
^(!and|or [_NN|NNS|NNP|NNPS{the:{0,1}} and _NN|NNS|NNP|NNPS{the:{0,1}}*]){and: 1, many: 1}
^(!and|or [_NN|NNS|NNP|NNPS{the:null} or _NN|NNS|NNP|NNPS{the:null}*]){or: 1}
^(!and|or [_NN|NNS|NNP|NNPS{the:{0,1}} or _NN|NNS|NNP|NNPS{the:{0,1}}*]){or: 1}
^([_NN|NNS|NNP|NNPS{the:null,and:null,or:null}* and _NN|NNS|NNP|NNPS{the:null}] !_VB|VBD|VBZ|VBP){and: 1, many: 1}
^([_NN|NNS|NNP|NNPS{the:{0,1},and:null,or:null}* and _NN|NNS|NNP|NNPS{the:{0,1}}]  !_VB|VBD|VBZ|VBP){and: 1, many: 1}
^([_NN|NNS|NNP|NNPS{the:null,and:null,or:null}* or _NN|NNS|NNP|NNPS{the:null}]  !_VB|VBD|VBZ|VBP){or: 1}
^([_NN|NNS|NNP|NNPS{the:{0,1},and:null,or:null}* or _NN|NNS|NNP|NNPS{the:{0,1}}]  !_VB|VBD|VBZ|VBP){or: 1}

(!and|or [_VBN{and:null,or:null} , _VBN*]){comma: 1}
(_VBN{comma:1,and:null,or:null} , and _VBN*){and: 1}
(_VBN{comma:1,and:null,or:null} , or _VBN*){or: 1}
(_VBN{and:null,or:null} and _VBN*){and: 1}
^(_VBN{and:null,or:null}* and _VBN){and: 1}
(_VBN or _VBN*){or: 1}
^(_VBN* or _VBN){or: 1}

^(!and|or [_EQN{and:null,or:null} , _EQN*]){comma: 1}
^(_EQN{and:null,or:null}* , _EQN){comma: 1}
^(_EQN{comma:1,and:null,or:null} , and _EQN*){and: 1, many: 1}
^(_EQN{comma:1,and:null,or:null} , or _EQN*){or: 1}
^(_EQN{comma:null} , and _EQN*){and: 1, many: 1, err: "redundant ',' before and"}
^(_EQN{comma:null} , or _EQN*){or: 1, err: "redundant ',' before or"}
^(!and|or [_EQN and _EQN*]){and: 1, many: 1}
^(!and|or [_EQN or _EQN*]){or: 1}
^(_EQN* and _EQN){and: 1, many: 1}
^(_EQN* or _EQN){or: 1}
(either _{or:1}*)

([_{subj:1}*] !_{of_type:1}){of_arg:1}
(_{gsp:NN}* _{of_type:1} _{of_arg:1}){of_arg:1}

^(_{and:1}* , respectively){resp:1}
^(_{or:1}* , respectively){resp:1}
^(_{and:1}* respectively){resp:1,err:"',' expected before 'respectively'"}
^(_{or:1}* respectively){resp:1,err:"',' expected before 'respectively'"}

(_JJ|DT [_VBG _{subj:1}*])

^(_{gsp:VB,what:null}* _DT){what:1}
^(_VBD{what:null}* _NN){what:1,canbe:NN}
^(_VBD{what:null}* _NNS){what:1,canbe:NNS}
^(_VBD{what:null}* _NNP){what:1,canbe:NNP}
^(_VBD{what:null}* _NNPS){what:1,canbe:NNPS}
^(_{gsp:VB,what:null}* _{subj:1}){what:1}
^(_{gsp:VB,whole:0}* : _EQN){what:1}
^(_{gsp:NN}* : _{subj:1,comma:1})
^(_{gsp:NN}* : _{subj:1,and:1})
(_{subj:1}*){can_be_what:1}
(us|me|him|her|it|them*){can_be_what:1}
^([_VBG{what:null}* _RB] _{can_be_what:1})
^(_VBG{what:null}* _{subj:1}){what:1}
^(_{gsp:VB,what:null}* us|me|him|her|it|them|itself|himself|myself|herself|themselves){whom:1}
^(_VBG{what:null}* us|me|him|her|it|them|itself|himself|myself|herself|themselves){whom:1}

_{gsp:VB}* _VBG
_MD _{gsp:VB}*
has|have{leaf:1}* _VBN|VBD
(has|have{leaf:1}* _{gsp:VB,whole:0}){err: "possible wrong verb form after 'has/have'"}

^(is|are|am|be|been|was|were{leaf:1}* _JJ|JJR|JJS|VBN){jprop:1}
^(do|does|was|were{leaf:1}* _VB|VBP|VBZ|VBD)

^_{accept_to:1}* _TO _{subj:1}
^_{accept_to:1}* _TO us|me|him|her|it|them
(_CD _{subj:1,the:null}*){count:1}
(_IN [_EQN{leaf:1} _{subj:1,many:1,the:null}*]){count:1}
(_IN [_EQN{leaf:1} _{subj:1,many:0,the:null}*]){count:1,err:"'s' expected after unknows count specified by formula"}
(one of _{subj:1}*){many:0}

^_{gsp:{VB},that:null}* _RB _IN _{subj:1}
^_{gsp:{VB,NN},that:null}* _IN _{subj:1}
^_{gsp:{VB}}* _RB _RP _{subj:1}
^_{gsp:{VB}}* _RP _{subj:1}
^_{gsp:{VB},that:null}* , like _{subj:1}
^_{gsp:{NN},that:null}* _OF _{subj:1}
^_{gsp:{VB,NN},that:null}* _IN|OF _DT of _{subj:1}
^_{gsp:VB}* _STD{gsp:RB,has_prep:1}
^_{gsp:{VB,JJ}}* _{join_right:{nn,subj}} _{subj:1}
^_{gsp:{VB,NN},that:null}* _IN|OF _VBG
^_{gsp:NN}* _VBG|VBN _IN|OF _{subj:1}
^_{gsp:NN}* , _VBG|VBN _IN|OF _{subj:1}
^_{gsp:NN}* _VBG|VBN{what:1}
%^_{gsp:NN}* _VBG _{subj:1}
^_VBG* _TO _{subj:1}
^(_{gsp:VB}* _TO _VB)

^(_{gsp:VB,what:null}* _{subj:1}){what:1}
^(_{gsp:VBG,what:null}* _{subj:1}){what:1}

(_RB and _RB*){and: 1}

_RB _{gsp:VB}*
^(_RB _VBG|VBN*)
^(_RB _JJ|JJR|RB|RBR*)
^(_RB _IN|OF*)

(_JJR{than:null}* than _{subj:1}){than:1}
(more|less{than:null}* _JJ than _{subj:1}){ps:JJR, than:1}
(more|less{than:null}* _RB than _{subj:1}){than:1}
(_JJR{than:null}* then _{subj:1}){than:1,err:"'than' expected instead of 'then'"}
^(_{subj:1}* _JJR{than:1})
^(_RBR _JJ|JJR|RB*)

^(_NN|NNS|NNP|NNPS{with_formula:null,with_number:null}* _CD){with_number:1}
^(_{subj:1}* but _{subj:1}){butsubj:1}

(_JJ _EQN*){subj:1,gsp:NN}

(_{}* _CITE){cite:1}
%#
(_VB , _VB*){comma: 1}
(_VB{comma:1} , and _VB*){and: 1}
(_VB{comma:1} , or _VB*){or: 1}
(_VB and _VB*){and: 1}
(_VB or _VB*){or: 1}
(_VB or _VB*){or: 1}
^(_VBG{gsp:VB}* and _VBG{gsp:=VB,accept_to:=1}){and:1}
^(_VBG{gsp:VB}* or _VBG{gsp:=VB,accept_to:=1}){or:1}
(_VBG{gsp:null}* and _VBG{gsp:null}){and:1}
(_VBG{gsp:null}* or _VBG{gsp:null}){or:1}

%#
let* _{gsp:VB,whole:0}
%#
([_{canbe:NN}*] _{gsp:VB,whole:0}){ps:NN,subj:1}
([_{canbe:NNS}*] _{gsp:VB,whole:0}){ps:NNS,subj:1}
([_{canbe:NNP}*] _{gsp:VB,whole:0}){ps:NNP,subj:1}
([_{canbe:NNPS}*] _{gsp:VB,whole:0}){ps:NNPS,subj:1}

([_{subj:1}* _VBD{whole:0,ps:=VBN}] _VB|VBZ|VBP{whole:0})
(no _{subj:1} _{gsp:VB,whole:0}*){whole:1,neg:1}
(_{subj:1} _{gsp:VB,whole:0}*){whole:1}
%^(_{subj:1,that_arg:1}* _{gsp:VB,whole:0}){whole:1}
(no _{can_p:1} _{gsp:VB,whole:0}*){whole:1,neg:1}
(_{can_p:1} _{gsp:VB,whole:0}*){whole:1}
(_JJ is|are*){whole:1,def:1}
(_SOL [_VB|VBP|VBD|VBZ{gsp:VB,whole:0}*]){whole:1}

^(_{gsp:VB,whole:1}* ,|and therefore|hence|thus|so _{at_end:1}){dep:hence,at_end:1}
(therefore|thus|hence _{gsp:VB,whole:1}*){hence:1}
(therefore|thus|hence , _{gsp:VB,whole:1}*){hence:1}
(_SOL [then _{gsp:VB,whole:1}*]){hence:1}
%#
(_SOL [_RB , _{gsp:VB}*])
(, _RB , _{gsp:VB}*)
_{vvod:1} , _{gsp:VB}*
(_{vvod:1} _{gsp:VB}*){err:"',' expected"}
%#
_IN _{subj:1} _VB|VBP|VBZ|VBD{whole:1}*
(_SOL [_IN _{subj:1} , _VB|VBP|VBZ|VBD{whole:1}*])
(_{subj:1} [, _IN _{subj:1} , _VB|VBP|VBZ|VBD{whole:0}*])
(which [, _IN _{subj:1} , _VB|VBP|VBZ|VBD{whole:0}*])

(there _{gsp:VB}*){whole:1}
(_{gsp:NN}*){accept_to:1}
(_{gsp:VB}* _RB)
%#
(_VB , _VB*){comma: 1}
(_VB{comma:1} , and _VB*){and: 1}
(_VB{comma:1} , or _VB*){or: 1}
^(_{gsp:VB,whole:0}* and _{gsp:VB,whole:0}){and: 1}
^(_{gsp:VB,whole:0}* , and _{gsp:VB,whole:0}){and: 1}
(_{gsp:VB,whole:1} and _{gsp:VB,whole:1}*){and: 1}
^(_{gsp:VB,whole:0}* or _{gsp:VB,whole:0}){or: 1}
^(_{gsp:VB,whole:0}* , or _{gsp:VB,whole:0}){or: 1}
(_{gsp:VB,whole:1} or _{gsp:VB,whole:1}*){or: 1}
%(_VB or _VB*){or: 1}
^(_NN|NNS|NNP|NNPS{and:null,or:null}* , _NN|NNS|NNP|NNPS){comma: 1}
^(_NN|NNS|NNP|NNPS* and _NN|NNS|NNP|NNPS){and: 1, many: 1}
^(_NN|NNS|NNP|NNPS* or _NN|NNS|NNP|NNPS){or: 1}
^(_NN|NNS|NNP|NNPS* , and _NN|NNS|NNP|NNPS){and: 1, many: 1}
^(_NN|NNS|NNP|NNPS* , or _NN|NNS|NNP|NNPS){or: 1}


^(_NN|NNS{which:null}* which _{gsp:VB}){which:1}
^(_NNP|NNPS{which:null}* who _{gsp:VB,whole:0}){which:1}
^(_{subj:1,which:null}* which _{gsp:VB}){which:1}
^(_NN|NNS{which:null}* , which _{gsp:VB}){which:1}
^(_NNP|NNPS{which:null}* , who _{gsp:VB,whole:0}){which:1}
^(_{subj:1,which:null}* , which _{gsp:VB}){which:1}

([_{whole:1}*] _EOL){at_end:1}
(_{gsp:VB,whole:0}* how to _{gsp:VB}){dep:howto}
(_{gsp:VB,whole:0}* how _{whole:1}){dep:how}
(_{gsp:VB,whole:0}* , how _{whole:1}){dep:how}
^(_{that:can_join}* _{whole:1}){that:none,dep:that}
^(_{gsp:VB,whole:0}* that _{whole:1}){dep:that}
^(_{gsp:NN}* that _{gsp:VB})
^(_{gsp:VB,whole:0}* when _{whole:1}){dep:when}
(_{gsp:VB,whole:1}* , so that _{whole:1}){dep:sothat}
(_{gsp:VB,whole:1}* so that _{whole:1}){dep:sothat}
^(_{gsp:NN}* , _SUCHTHAT _{at_end:1}){at_end:1,dep:suchthat}
^(_{gsp:NN}* _SUCHTHAT _{at_end:1}){at_end:1,dep:suchthat}
^(_{gsp:VB,whole:0}* whether _{at_end:1}){dep:whether}

^(that _VBG|NN|NNS|NNP|NNPS{the:null}*){gsp:NN,subj:1,that_arg:1}

(if|when _{whole:1}* then _{whole:1}){dep:if_then}
(if|when _{whole:1}* , then _{whole:1}){dep:if_then}
(when _{whole:1}* , _{whole:1}){dep:if_then}
(when _{whole:1}* _{whole:1}){dep:if_then}
(if _{whole:1}* _{whole:1}){dep:if_then,err:"',', 'or', 'and' or 'then' expected"}
(if _{whole:1}* , _{whole:1}){dep:if_then,err:"',', 'or', 'and' or 'then' expected"}
([_{gsp:VB,whole:1}* if _{whole:1}] !then){dep:if}
([_{gsp:VB,whole:1}* , if _{whole:1}] !then){dep:if}

(_{gsp:VB,whole:1}* though|before|after|as _{gsp:VB,whole:1,at_end:1}){dep:infix,at_end:1}
([_{gsp:VB,whole:1}* as _EQN] _EOL){dep:as_limit,at_end:1}
(though|before|after|as _{whole:1} _{whole:1}*){dep:prefix}
(though|before|after|as _{whole:1} , _{whole:1}*){dep:prefix}

(_{gsp:VB,whole:1}* where _{whole:1}){dep:where}
(_{gsp:VB,whole:1}* , where _{whole:1}){dep:where}
(_{gsp:VB,whole:1}* since _{gsp:VB,whole:1,at_end:1}){dep:since,at_end:1}
(_SOL [since _{whole:1} _{whole:1}*]){dep:since}
(_SOL [since _{whole:1} , _{whole:1}*]){dep:since}
(_SOL [since _{whole:1} then _{whole:1}*]){dep:since}
(_SOL [since _{whole:1} , then _{whole:1}*]){dep:since}
(_{gsp:VB,whole:1}* _{due_to:1} _{gsp:VB,whole:1,at_end:1}){dep:dueto,at_end:1}
(_{gsp:VB,whole:1}* , _{due_to:1} _{gsp:VB,whole:1,at_end:1}){dep:dueto,at_end:1}
(_{gsp:VB,whole:1}* because _{gsp:VB,whole:1,at_end:1}){dep:because,at_end:1}
(_{gsp:VB,whole:1}* , because _{gsp:VB,whole:1,at_end:1}){dep:because,at_end:1}
(_{gsp:VB,whole:1}* in order to _{gsp:VB}){dep:goal}
(_{gsp:VB,whole:1}* , in order to _{gsp:VB}){dep:goal}

(is* why _{whole:1}){dep:reason}

(for _{subj:1} _{gsp:VB,whole:1}*){for:1}
(for _{subj:1} , _{gsp:VB,whole:1}*){for:1}
(to _VB _{gsp:VB,whole:1}*){to:1}
(to _VB , _{gsp:VB,whole:1}*){to:1}
(_{whole:1} but* _{whole:1}){whole:1,dep:but}
(_{whole:1} , but* _{whole:1}){whole:1,dep:but}
(_{whole:1} , while* _{whole:1}){whole:1,dep:while}
(_{whole:1} while* _{whole:1}){whole:1,dep:while}
([_{whole:1}* , _VBG] _EOL){comment:1}
(_VBG , _{whole:1}*){how:1}
(_VBN{whole:0} _{whole:1}*){dep:cond}
(_VBN{whole:0} , _{whole:1}*){dep:cond}
([_{whole:1}* , _IE _{whole:1}] _EOL){ie:1}
(_{whole:1}* , _IE _{whole:1} ,){ie:1}
([_{whole:1}* _IE _{whole:1}] _EOL){ie:1,err:"',' before 'that is' or 'i.e.' expected"}

(_{whole:1} , _{whole:1}*){comma:1}
(_{whole:1}* : _{whole:1}){compound:colon}
(_{whole:1}* ; _{whole:1}){compound:semicolon}
(_{whole:1} , and _{whole:1}*){and:1}
(_{whole:1} , or _{whole:1}*){or:1}
(_{whole:1} and _{whole:1}*){and:1}
(_{whole:1} or _{whole:1}*){or:1}
(let*){whole:1}

^(_{gsp:NN}* _SUCHTHAT _{whole:1}){dep:suchthat}
^(_{gsp:NN}* _SUCHTHAT _EQN){dep:suchthat}
(_VBG _{gsp:VB,whole:1}*){dep:how}
(_EQN*){whole:1}

(_{vvod:1} _{whole:1}*)
(_{vvod:1} , _{whole:1}*)
(_IN _{subj:1} , _VB|VBP|VBZ|VBD{whole:1}*)

(_CD){subj:1}
^(_{gsp:VB,what:null}* that){what:1,that:1}
(which*){ps:DT}
(of){ps:IN}
(_{}* \( _{} \)){comment:1,err:cannot determine comment type}
(_{}* \( _{} _{} \)){comment:1,err:comment not fully parsed}

^(_EQN* _CD)
^(_EQN _CD*)
^(_CD _EQN*)
^(_CD* _EQN)
(: _EQN*){eqcolon:1}
%(_VBD{whole:0}){ps:VBN}
(_VB|VBP{whole:0}){whole:1}
#
([_{gsp:VB,whole:1}* _CITE] _EOL){cite:1}
(_SOL _{whole:1}* _EOL){sentense:1}
(_SOL _{}* _EOL){sentense:1,err:sentense is incomplete}
"""


math_env_names = {"equation", "equation*", "align", "align*", "eqnarray", "eqnarray*", "multline", "multline*"
                  "tikzcd", "tikzcd*", "gather", "gather*"}


grammar = parse_grammar(grammar_str)


def test_full_tex_file(file_name, max_fails=100, pr=True, pickle_file=None):
    document = TexDocument(filename=file_name)
    if pr:
        document.print_document_info()
    parsed = []
    failed = []
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

    text = " ".join(text_segments)

    #for fragment in document.text_fragments():
    sent_tokens = tokenize_text_nltk(text)
    for i, sent in enumerate(sent_tokens):
        if pr:
            print(f"{i+1}. {' '.join(x[0] for x in sent)}")
        parse_trees = parse_sentense(sent, grammar, debug=False)
        if pickle_file:
            with open(pickle_file, 'wb') as f:
                pickle.dump(parse_trees, f)
        if len(parse_trees) != 1:
            with open('failed.txt', 'a' if len(failed) else 'w') as f:
                f.write(f"{len(failed)+1}. {' '.join(x[0] for x in sent)}\n")
                for tree in parse_trees:
                    f.write(tree.str_for_print())
                f.write('\n=========================================================\n')
            failed.append((i, ' '.join(x[0] for x in sent), parse_trees))
        else:
            with open('parsed.txt', 'a' if len(parsed) else 'w') as f:
                f.write(f"{len(parsed)+1}. {' '.join(x[0] for x in sent)}\n")
                f.write(parse_trees[0].str_for_print())
                f.write('\n=========================================================\n')
            parsed.append((i, ' '.join(x[0] for x in sent), parse_trees))
        if len(failed) >= max_fails:
            break

    if pr:
        for i, sent, parse_trees in failed:
            print(f'{i}. {sent}')
            for tree in parse_trees:
                tree.print_tree()
            print()

    if pr:
        print("Parsed:", len(parsed))
        print("Failed:", len(failed))
    return len(parsed), len(failed)


def test_text(text, pr, pickle_file=None):
    failed = 0
    parsed = 0
    sent_tokens = tokenize_text_nltk(text)
    all_trees = []
    for i, sent in enumerate(sent_tokens):
        if pr:
            print(f"{i+1}. {' '.join(x[0] for x in sent)}")
        if len(sent) > 100:
            print(f"Too long sentense: len = {len(sent)}")
            continue
        parse_trees = parse_sentense(sent, grammar, debug=False)
        all_trees.append(parse_trees)
        if len(parse_trees) != 1:
            failed += 1
        else:
            parsed += 1

    if pickle_file:
        # if needed create path to file
        path = os.path.dirname(pickle_file)
        if path and not os.path.exists(path):
            os.makedirs(path)
        with open(pickle_file, 'wb') as f:
           pickle.dump(all_trees, f)

    if pr:
        print("Parsed:", parsed)
        print("Failed:", failed)
    return parsed, failed


# iterate over files in .tar file
import tarfile
def iterate_tar_contents(filename):
    tar = tarfile.open(filename)
    try:
        while member := tar.next():
            if member.isdir():
                continue
            content = tar.extractfile(member).read().decode("utf-8")
            yield member.name, content
    finally:
        tar.close()


def test_article_dir(dir_name, max_fails=100, pr=True):
    # find .tex file that contains \documentclass
    main_file = None
    for file_name in os.listdir(dir_name):
        if file_name.endswith('.tex'):
            with open(os.path.join(dir_name, file_name), 'rb') as f:
                if b'\\documentclass' in f.read():
                    #print(file_name)
                    main_file = file_name
                    break
    if main_file is None:
        return "main document not found"
    dir_last_name = dir_name.split('/')[-1]
    return test_full_tex_file(os.path.join(dir_name, main_file), max_fails, pr, 'results/' + dir_last_name + '.pickle')


def test_article_dir_tuple(args):
    try:
        return test_article_dir(*args)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return str(e)


def test_text_tuple(args):
    try:
        return args[0], test_text(*args[1:])
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return args[0], str(e)

# tests each directory in the given directory
def test_dir_with_articles(dir_name):
    parsed = 0
    total = 0
    dir_list = []
    for name in os.listdir(dir_name):
        path = os.path.join(dir_name, name)
        if os.path.isdir(path):
            dir_list.append(name)
    dir_list = sorted(dir_list)
    print(f'{len(dir_list)} directories found')

    with open('log.txt', 'w') as f:
        for name in dir_list:
            path = os.path.join(dir_name, name)
            print(f'Test {name}', end=': ')
            try:
                res = test_article_dir(path,10000,False)
                if isinstance(res, tuple):
                    print(f'{res[0]}/{res[0]+res[1]} ({res[0]/(res[0]+res[1]):.0%}) parsed')
                    f.write(f'{name}: {res[0]}/{res[0]+res[1]} ({res[0]/(res[0]+res[1]):.0%})\n')
                    parsed += res[0]
                    total += res[0] + res[1]
                else:
                    print(f'{res}')
                    f.write(f'{name}: {res}\n')
            except Exception as e:
                if e is KeyboardInterrupt:
                    raise
                print(f'{e}')
                f.write(f'{name}: {e}\n')
            f.flush()
        print(f'Total: {parsed}/{total} ({parsed/total:.0%}) parsed')
        f.write(f'Total: {parsed}/{total} ({parsed/total:.0%}) parsed\n')
    return parsed, total


# similar procedure but uses multiprocessing to test each directory in parallel
def test_dir_with_articles_parallel(dir_name):
    parsed = 0
    total = 0
    dir_list = []
    for name in os.listdir(dir_name):
        path = os.path.join(dir_name, name)
        if os.path.isdir(path):
            dir_list.append(name)
    dir_list = sorted(dir_list)
    print(f'{len(dir_list)} directories found')

    with open('log.txt', 'w') as f:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
            for name, res in zip(dir_list, pool.imap(test_article_dir_tuple, [(os.path.join(dir_name, name), 10000, False) for name in dir_list])):
                if isinstance(res, tuple):
                    if res[0]+res[1]==0:
                        print(f'{name}: no sentenses found')
                        f.write(f'{name}: no sentenses found\n')
                    else:
                        print(f'{res[0]}/{res[0]+res[1]} ({res[0]/(res[0]+res[1]):.0%}) parsed')
                        f.write(f'{name}: {res[0]}/{res[0]+res[1]} ({res[0]/(res[0]+res[1]):.0%})\n')
                    parsed += res[0]
                    total += res[0] + res[1]
                else:
                    print(f'{res}')
                    f.write(f'{name}: {res}\n')
                f.flush()
            print(f'Total: {parsed}/{total} ({parsed/total:.0%}) parsed')
            f.write(f'Total: {parsed}/{total} ({parsed/total:.0%}) parsed\n')
    return parsed, total


# similar procedure but uses multiprocessing to test each directory in parallel
def test_tar_with_articles_parallel(tar_name, max_number_of_articles=100000):
    tar_file_name = tar_name.split('/')[-1]

    parsed = 0
    total = 0

    tar_iterator = iterate_tar_contents(tar_name)
    with open(f'out_{tar_file_name}.txt', 'w') as f:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
            for name, res in pool.imap(test_text_tuple, ((name, contents, False, f'results_tar/{i//1000*1000}-{i//1000*1000+999}/{name}.pickle')
                                                                for i, (name, contents) in zip(range(max_number_of_articles), tar_iterator))):
                if isinstance(res, tuple):
                    if res[0]+res[1]==0:
                        print(f'{name}: no sentenses found')
                        f.write(f'{name}: no sentenses found\n')
                    else:
                        print(f'{res[0]}/{res[0]+res[1]} ({res[0]/(res[0]+res[1]):.0%}) parsed')
                        f.write(f'{name}: {res[0]}/{res[0]+res[1]} ({res[0]/(res[0]+res[1]):.0%})\n')
                    parsed += res[0]
                    total += res[0] + res[1]
                else:
                    print(f'{res}')
                    f.write(f'{name}: {res}\n')
                f.flush()
            print(f'Total: {parsed}/{total} ({parsed/total:.0%}) parsed')
            f.write(f'Total: {parsed}/{total} ({parsed/total:.0%}) parsed\n')
    return parsed, total


# read parse trees from files from directory results/*.pickle
def read_trees(dir, pr=False):
    trees = []
    files = [f for f in os.listdir(dir) if f.endswith('.pickle')]
    if pr:
        bar = progressbar.ProgressBar(maxval=len(files))
        bar.start()
    for fi, f in enumerate(files):
        if f.endswith('.pickle'):
            with open(os.path.join(dir, f), 'rb') as f:
                trees.append(pickle.load(f))
            if pr:
                bar.update(fi)
    if pr:
        bar.finish()
    return trees


# read parse trees from files from directory results/*.pickle
def read_trees_lazy(dir, pr=False):
    trees = []
    files = [f for f in os.listdir(dir) if f.endswith('.pickle')]
    if pr:
        bar = progressbar.ProgressBar(maxval=len(files))
        bar.start()
    for fi, f in enumerate(files):
        if f.endswith('.pickle'):
            with open(os.path.join(dir, f), 'rb') as f:
                yield pickle.load(f)
            if pr:
                bar.update(fi)
    if pr:
        bar.finish()


import progressbar


stop_words.update(['.',',','$eqn$','*','-','(',')','[',']'])


def test_trees(repeat_threshold=10):
    # enumerate directories in results_tar/
    dirs = []
    for d in os.listdir('results_tar'):
        if os.path.isdir(os.path.join('results_tar', d)):
            dirs.append(d)

    print(f"{len(dirs)} dirs found")
    # create progress bar to show progress

    combinations = defaultdict(lambda: 0)
    subtrees = defaultdict(lambda: 0)

    for di, d in enumerate(sorted(dirs)):
        print(f'processing dir {d} ...')

        trees = read_trees_lazy(os.path.join('results_tar', d, 'documents'), True)
        #print(f'{d}: {len(trees)} files with trees found')

        #bar = progressbar.ProgressBar(maxval=len(trees))
        #bar.start()

        for i, tree_file in enumerate(trees):
            for sentense in tree_file:
                for tree in sentense:
                    for node in tree:
                        if isinstance(node, ParseTree):
                            tokens = [x.get_word() if x.metadata['ps'] != 'EQN' else '$eqn$' for x in node.children]
                            leaves = [x.get_word() if x.metadata['ps'] != 'EQN' else '$eqn$' for x in node if x.is_leaf()]
                            if sum(1 for x in tokens if x not in stop_words) > 1:
                                combinations[tuple(tokens)] += 1
                            if sum(1 for x in leaves if x not in stop_words) > 1:
                                if len(leaves) <= 6:
                                    subtrees[tuple(leaves)] += 1
            #bar.update(i+1)
        #bar.finish()

        # convert combinations to list sorted by count in descending order
        combination_list = sorted((x for x in combinations.items() if x[1] >= repeat_threshold), key=lambda x: x[1], reverse=True)
        # convert subtrees to list sorted by count in descending order
        subtree_list = sorted((x for x in subtrees.items() if x[1] >= repeat_threshold), key=lambda x: x[1], reverse=True)
        # save results to files
        with open('combinations.txt', 'w') as f:
            for comb in combination_list:
                f.write(f'{" ".join(comb[0])}: {comb[1]}\n')

        with open('subtrees.txt', 'w') as f:
            for subt in subtree_list:
                f.write(f'{" ".join(subt[0])}: {subt[1]}\n')


if __name__ == '__main__':
    test_trees()
    #test_full_tex_file('tests/main.tex')
    #test_full_tex_file('tests/balanced_product/balanced_product_codes.tex')
    #test_article_dir("/Users/gleb/PycharmProjects/ineq-prover/prover/arxiv/0705/0705.0968")
    #test_dir_with_articles_parallel("/Users/gleb/PycharmProjects/ineq-prover/prover/arxiv/0705")
    #test_tar_with_articles_parallel("/Users/Gleb/Desktop/Solver/2020-09-08-arxiv-extracts-nofallback-until-2007-068.tar", 100000)
    exit(0)
    grammar = parse_grammar(grammar_str)
    print(grammar)
    test_sentenses = [
        r"now consider codes $\mathcal{C}_1$ and $\mathcal{C}_2$ such that they have property $(*)$ , and the dual product code $\mathcal{C}_1\boxplus \mathcal{C}_2$ does not have codewords of small weight and large rank .",
        # 'the cat is in the box',
        # "Also note that we can always remove the logarithm in the above inequalities by using its concavity and Jensen's inequality",
        # "This can be fixed by proving a multivariate extension of the ALT inequality based on pinching(see Theorem 2.3).",
        # "However, we only recover the result for q using complex interpolation theory.",
        # "Next, recall the multivariate Lie-Trotter product formula in (28)",
        # "For positive semi-definite operators $\\rho$ and $\\sigma$ , the Hermitian operators $\\sigma$ , $\\rho$ and $\\sigma$ are well-defined under the convention $X$."
    ]
    for sentense in test_sentenses:
        print("=========================================================")
        print(f"Test sentense: {sentense}\n")
        tokens = tokenize_text_nltk(sentense)[0]
        print(f"tokens = {tokens}")
        print('\nParsing...')
        parse_trees = parse_sentense(tokens, grammar)
        print("Parse trees:")
        for tree in parse_trees:
            tree.print_tree()


# # Test function for parse_natural_text
# def test_parse_natural_text():
#     relations = defaultdict(lambda: RelationStats())
#     patterns = [  # list of english language standard relations and their patterns
#         ('part_of', "... x_NN is _ part of _ y_NN"),
#         ('is_a', "... x_NN is _ y_NN"),
#         ('can_be', "... x_NN can be _ y_NN"),
#         ('has_a', "... x_NN has a _ y_NN"),
#         ('has_many', "... x_NN has many _ y_NNS"),
#     ]
