import abc

# import tools for natural language processing
import multiprocessing
import os
import pickle
import re
import shutil
import sys
import time
from collections import defaultdict
from copy import copy
from typing import List, Tuple, Optional, Union, Dict

import yaml
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize


# get infinitive form of a word using nltk
from parse_formula import parse_eqn, Variable, EqnList, ExprTokenSeq, ExprElement, tex_rel_bin_operators
from stdenvs import BibliographyEnvironment
from tagger import dictionaries, stop_words
from texdocument import TexDocument, TextFragment, MathEnv
from texstream import reset_tex_errors, tex_error, tex_print_errors, tex_warning, tex_error_context


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
        prev_end = match.end()
        symb = ''
        for c in formula[::-1]:
            if c == '}' or c.isalnum():
                break
            elif c in '.,;:':
                symb = c
                break
        if symb:
            fragments[-1] += ' ' + symb
        elif '.' in formula or formula[-1] != '$' or formula[-2] == '$':
            for j in range(match.end(), len(text)):
                if not text[j].isspace():
                    if text[j].isalpha() and text[j].isupper():
                        fragments[-1] += ' .'
                    break
        prev_end = match.end()
    fragments.append(text[prev_end:])
    return re.sub(r'\\[a-zA-Z]*[*]?', ' ', ' '.join(fragments)), formulas


# text efficient multiple replacement
def replace_multiple(text: str, replacements):
    return re.subn('|'.join(re.escape(k) for k in replacements.keys()), lambda m: replacements[m.group(0)], text)


def tokenize_text_nltk(txt):
    dicts = dictionaries()
    res = []
    txt, formulas = replace_tex_formulas(txt)
    # remove all braces
    replacements = {'{': '', '}': '', 'i.e.': 'ie', 'i. e.': 'ie', 'e.g.': 'eg', 'e. g.': 'eg', 'et al.': 'et_al', 'cf.': 'cf',
                    'Fig.': 'Fig', 'fig.': 'Fig', 'Ref.': 'Ref', 'ref.': 'ref', 'Eq.': 'Eq', 'eq.': 'Eq',
                    'resp.': 'respectively'}
    txt, _ = replace_multiple(txt, replacements)

    tokenized = sent_tokenize(txt)
    for i, sent in enumerate(tokenized):
        fst = sent.find(' ')
        first_word = sent[:fst]
        if first_word.lower() in stop_words:
            tokenized[i] = first_word.lower()+sent[fst:]

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
            tagged[i] = dicts.get_pos(word, tag)

        tagged = [('', 'SOL')] + tagged
        # find and replace formulas in tagged list
        for i in range(len(tagged)):
            if tagged[i][0].startswith('formula___'):
                formula = formulas[tagged[i][0]]
                if formula.endswith('-'):
                    tagged[i] = (formula, 'EQNP')
                else:
                    tagged[i] = (formula, 'EQN')
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

    @property
    @abc.abstractmethod
    def text(self) -> str:
        pass

    @property
    def arg_name(self):
        return self.metadata.get('arg', None)

    @property
    def ps(self):
        return self.metadata.get('ps', None)


def get_parsed_eqn_metadata(eqn_parsed, eqn):
    res = {}
    if isinstance(eqn_parsed, Variable):
        res['variable'] = 1
        res['many'] = 0
    elif isinstance(eqn_parsed, EqnList):
        if len(eqn_parsed.eqns) > 1:
            res['many'] = 1
            res['comma'] = 1
        has_rel = 0
        has_not_rel = 0
        for eq in eqn_parsed.eqns:
            eq_res = get_parsed_eqn_metadata(eq, eqn)
            if eq_res.get('whole', 0):
                has_rel += 1
            else:
                has_not_rel += 1
            if not eq_res.get('variable', 0):
                res['variable'] = 0
        #if has_rel > 0 and has_not_rel > 0:
        #    tex_warning(f'Equation {eqn} has both relations and non-relations')
        if has_rel > 0:
            res['whole'] = 1
        else:
            res['whole'] = 0
    elif isinstance(eqn_parsed, Relation):
        res['whole'] = 1
    elif isinstance(eqn_parsed, ExprTokenSeq):
        if any(isinstance(x, ExprElement) and x.token in tex_rel_bin_operators for x in eqn_parsed.tokens):
            res['whole'] = 1
    res.setdefault('variable', 0)
    res.setdefault('many', 0)
    res.setdefault('whole', 0)
    return res


def get_eqn_metadata(eqn):
    if eqn[:2] == '$$' and eqn[-2:] == '$$':
        eqn = eqn[2:-2]
    if eqn[:1] == '$' and eqn[-1] == '$':
        eqn = eqn[1:-1]
    eqn_parsed = parse_eqn(eqn)
    return get_parsed_eqn_metadata(eqn_parsed, eqn)


# Token class, contains word and metadata from nltk tokenizer
class Token(ParseTreeNode):
    def __init__(self, word, pos, metadata):
        super().__init__()
        self.word = word
        self.pos = pos
        self.metadata = copy(metadata)
        if self.ps in ('IN', 'TO'):
            self.metadata.update(dictionaries().prep_types.get(word, {}))
        self.metadata['leaf'] = 1
        if self.ps == 'EQN' and self.word[-1] != '-' and (self.word[0] == '$' or self.word[:2] == '\\['):
            self.metadata.update(get_eqn_metadata(self.word))

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

    @property
    def text(self):
        return self.word


# Parse tree for natural language;
# each leaf is a token (with meta data)
# each internal node is a rule applied to a list of children; is can have main token
class ParseTree(ParseTreeNode):
    def __init__(self, token, rule=None, children=None, metadata=None, main_branch=None, level=None, rule_num=None,
                 chech_func=None, create_func=None):
        super().__init__()
        self.token: Optional[Token] = token if isinstance(token, Token) else None
        self.metadata = {} if token is None else copy(token.metadata if main_branch is None else children[main_branch].metadata)
        self.metadata['leaf'] = 0
        if 'arg' in self.metadata:
            del self.metadata['arg']
        if metadata is not None:
            self.metadata.update(metadata)
        self.rule = rule
        self.main_branch = main_branch
        self.children = children if children is not None else []
        self.level = level
        self.rule_num = rule_num
        self.check_func = chech_func
        if create_func is not None:
            create_func(self)

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

    @property
    def text(self):
        return " ".join(c.text for c in self.children)

    def check(self):
        for c in self.children:
            if isinstance(c, ParseTree):
                c.check()
        if self.check_func is not None:
            args = [c for c in self.children if c.arg_name is None]
            kwargs = {c.arg_name: c for c in self.children if c.arg_name is not None and c.arg_name != '_'}
            if isinstance(self.check_func, list):
                for func in self.check_func:
                    func(self, *args, **kwargs)
            else:
                self.check_func(self, *args, **kwargs)


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
                    elif len(parts:=value.split('->', 2)) > 1:
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
        if isinstance(item, RuleItemWithTags) and item.replace_tags and tuple(item.replace_tags.keys()) != ('arg',):
            raise ValueError('! cannot be used with tag replacement')

    def __str__(self):
        return f'!{self.item}'

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, NegateRuleItem) and self.item == other.item

    def match(self, node: ParseTreeNode):
        return not self.item.match(node)

    def replace(self, node: ParseTreeNode):
        if isinstance(self.item, RuleItemWithTags):
            self.item.replace(node)


class GrammarRule:
    def __init__(self, name, items: List[RuleItem], main_token=None, metadata=None, interval=None, subtree=False,
                 destruct=False, use_priority=False, text="", cond_func=None, postprocess_func=None, create_func=None):
        self.text = text
        self.name = name
        self.items = items
        self.destruct = destruct
        self.main_token = main_token
        self.metadata = copy(metadata) if metadata is not None else {}
        self.interval = interval if interval is not None else (0, len(items))
        self.subtree = 0
        self.use_priority = use_priority
        self.cond_func = cond_func
        self.postprocess_func = postprocess_func
        self.create_func = create_func
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

    def match_partial(self, *nodes: ParseTreeNode):
        return all(item.match(node) for item, node in zip(self.items, nodes) if node is not None)

    def match(self, *nodes: ParseTreeNode) -> Union[bool, List[ParseTreeNode]]:
        if any(not x.match(node) for node, x in zip(nodes, self.items)):
            return False
        if self.cond_func:
            args = [node for node in nodes if node.arg_name is None]
            kwargs = {node.metadata['arg']: node for node in nodes if node.arg_name not in (None,'_')}
            return self.cond_func(*args, **kwargs)
        return True


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
                    segment = parse_trees[i:i + len(rule.items)]
                    if len(segment) > 2 and not rule.match_partial(None, *segment[1:-1], None):
                        continue
                    if rule.subtree == 1:
                        if not rule.items[-1].match(segment[-1]):
                            continue
                        curr = segment[0]
                        left_branch = [curr]
                        while not curr.is_leaf():
                            curr = curr.children[-1]
                            left_branch.append(curr)

                        found = None
                        for idx, prev, curr in zip(range(len(left_branch)), left_branch[0:-1], left_branch[1:]):
                            # take into account rule priority
                            if rule.use_priority and found and (prev.level, prev.rule_num) < (level, rule_num):
                                break
                            if mres := rule.match(curr, *segment[1:]):
                            # if rule.items[0].match(curr) and all(x.match(parse_trees[i + j + 1]) for j, x in enumerate(rule.items[1:])):
                                found = (prev, curr, idx, mres)
                        if found:
                            prev, curr, i0, mres = found
                            b, e = rule.interval
                            assert b == 0
                            rule.replace_items_tags([curr] + segment[1:])
                            new_tree = ParseTree(curr.token, str(rule), [curr] + segment[1:e],
                                                 metadata=rule.metadata, main_branch=rule.main_token - b,
                                                 level=level, rule_num=rule_num,
                                                 chech_func=rule.postprocess_func, create_func=rule.create_func)
                            if debug:
                                print(f'apply rule {rule} to {[curr] + segment[1:e]}: {new_tree}')
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
                        if not rule.items[0].match(segment[0]):
                            continue
                        curr = segment[-1]
                        right_branch = [curr]
                        while not curr.is_leaf():
                            curr = curr.children[0]
                            right_branch.append(curr)

                        found = None
                        for idx, prev, curr in zip(range(len(right_branch)), right_branch[0:-1], right_branch[1:]):
                            # take into account rule priority
                            if rule.use_priority and found and (prev.level, prev.rule_num) < (level, rule_num):
                                break
                            if mres := rule.match(*segment[:-1], curr):
                                found = (prev, curr, idx, mres)
                        if found:
                            prev, curr, i0, mres = found
                            b, e = rule.interval
                            assert e == len(rule.items)
                            rule.replace_items_tags(segment[:e-1] + [curr])
                            new_tree = ParseTree(curr.token, str(rule), segment[b:e-1] + [curr],
                                                 metadata=rule.metadata, main_branch=rule.main_token - b,
                                                 level=level, rule_num=rule_num,
                                                 chech_func=rule.postprocess_func, create_func=rule.create_func)
                            if debug:
                                print(f'apply rule {rule} to {segment[b:e-1] + [curr]}: {new_tree}')
                            prev.children[0] = new_tree
                            del parse_trees[i + b:i + e - 1]
                            for node, next in zip(right_branch[i0::-1], right_branch[i0 + 1::-1]):
                                if node.main_child is next:
                                    for tag in next.metadata:
                                        if tag not in node.metadata:
                                            node.metadata[tag] = next.metadata[tag]
                            changed = True
                            break

                    if mres := rule.match(*segment):
                        #if all(x.match(parse_trees[i + j]) for j, x in enumerate(rule.items)):
                        b, e = rule.interval
                        if rule.main_token is None:
                            if all(isinstance(x, Token) for x in segment[b:e]):
                                new_token = Token(' '.join(x.get_word() for x in segment[b:e]),
                                                  segment[b].pos, metadata=rule.metadata)
                                if debug:
                                    print(f'apply rule {rule} to {segment[b:e]}: {new_token}')
                                parse_trees[i+b:i+e] = [new_token]
                                changed = True
                                break
                        elif rule.is_tag_add_rule():
                            rule_key = str(rule)
                            if rule_key not in segment[b].applied_rules:
                                segment[b].applied_rules.add(rule_key)
                                rule.replace_items_tags(segment)
                                subtree = segment[b]
                                if debug:
                                    print(f'add tags ({rule}) {rule.metadata} to {subtree}: ', end='')
                                tags_changed = rule.add_tags(subtree)
                                changed = tags_changed or changed
                                if debug:
                                    print(subtree, f'  (changed={tags_changed})')
                        else:
                            rule.replace_items_tags(segment)
                            new_tree = ParseTree(segment[rule.main_token].token, str(rule),
                                                 segment[b:e],
                                                 metadata=rule.metadata, main_branch=rule.main_token - b, level=level,
                                                 rule_num=rule_num,
                                                 chech_func=rule.postprocess_func, create_func=rule.create_func)
                            if debug:
                                print(f'apply rule {rule} to {segment[b:e]}: {new_tree}')
                            parse_trees[i + b:i + e] = [new_tree]
                            changed = True
                            break
                if changed:
                    if debug:
                        print('changed')
                    break

    for tree in parse_trees:
        if isinstance(tree, ParseTree):
            tree.check()

    return parse_trees


def parse_item(item: str) -> Tuple[RuleItem, bool]:
    if item[0] == '!':
        it = parse_item(item[1:])
        return NegateRuleItem(it[0]), it[1]
    selected = False
    if item.endswith('*'):
        selected = True
        item = item[:-1]

    tags = {}
    if item.count('{'):
        item = item.replace(':', ': ')
        pos = item.index('{')
        tags = yaml.safe_load(item[pos:])
        item = item[:pos]

    if item.count('_'):
        pos_ = item.index('_')
        v = item[:pos_]
        suffix = item[pos_ + 1:]
        if '|' in suffix:
            suffix = set(suffix.split('|'))
        if suffix:
            tags['ps'] = suffix
        return RuleItemVariable(v, tags), selected
    else:
        return RuleItemConst(item.split('|'), tags=tags), selected


def parse_rule(rule: str, cond_func=None, check_func=None, create_func=None) -> GrammarRule:
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
                       subtree=subtree, use_priority=True, text=text,
                       cond_func=cond_func, postprocess_func=check_func, create_func=create_func)


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


def parse_grammar_list(grammar: List[Tuple]):
    level = 0
    rules = []
    for rulestr, *func in grammar:
        line = rulestr.strip()
        # skip empty lines
        if line == '':
            raise Exception('Empty rule in grammar')
        # '#' means new level
        if line[0] == '#':
            level += 1
        else:  # parse rule
            rules.append((level, parse_rule(line, *func)))
    return Grammar(rules)


from grammar import grammar_str
grammar = parse_grammar_list(grammar_str)


math_env_names = {"equation", "equation*", "align", "align*", "eqnarray", "eqnarray*", "multline", "multline*",
                  "tikzcd", "tikzcd*", "gather", "gather*"}


sufficient_pos_set = {'FW', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'RB', 'RBR', 'RBS', 'RP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}


def test_full_tex_file(file_name, max_fails: Optional[int] = 100, pr=1, pickle_file=None):
    dicts = dictionaries()
    reset_tex_errors()
    if pr:
        print(f'Preprocessing {file_name} ... ')
    document = TexDocument(filename=file_name)
    if pr:
        print('Preprocessing done')
    if pr > 1:
        document.print_document_info()
    parsed = []
    failed = []
    # collect text and math environments
    text_segments = []
    eqn_counter = 0
    for env in document.items_and_envs([TextFragment, BibliographyEnvironment], math_env_names):
        if isinstance(env, TextFragment):
            text_segments.append(env.remove_formatting_macros())
        elif isinstance(env, BibliographyEnvironment):
            pass
        else:
            assert isinstance(env, MathEnv)
            eqn_counter += 1

            text_segments.append(f"equation_{eqn_counter}")
            last_frag = env.frag
            if isinstance(last_frag, TextFragment) and last_frag.text.count('.'):
                text_segments.append('.')

    text = " ".join(text_segments)
    with open('before_parse.txt', 'w') as f:
        f.write(text)

    # copy files parsed.txt, failed.txt and parse_trees.txt (if exist) to files with the same name with .old extension
    for file in ['parsed.txt', 'failed.txt', 'parse_trees.txt']:
        if os.path.isfile(file):
            shutil.copy(file, file + '.old')


    # for fragment in document.text_fragments():
    sent_tokens = tokenize_text_nltk(text)
    uncommon_words = defaultdict(lambda: 0)
    errors = []

    progress = progressbar.ProgressBar(maxval=len(sent_tokens))
    if pr <= 1:
        tex_error_context.print_level = 4
    else:
        tex_error_context.print_level = 0

    if pr == 1:
        print(f'Parsing ...')
        progress.start()

    for i, sent in enumerate(sent_tokens):
        if pr >= 2:
            print(f"{i+1}. {' '.join(x[0] for x in sent)}")
        for token, pos in sent:
            if token and token[0].isalpha() and token[0].islower() and pos in sufficient_pos_set \
                    and pos in dicts.dict_pos_set \
                    and not token in stop_words:
                if token not in dicts.pos_dict:
                    uncommon_words[token] += 1

        parse_trees = parse_sentense(sent, grammar, debug=False)
        if pickle_file:
            with open(pickle_file, 'wb') as f:
                pickle.dump(parse_trees, f)
        for tree in parse_trees:
            if isinstance(tree, ParseTree):
                for subtree in tree:
                    if isinstance(subtree, ParseTree) and subtree.metadata.get('err', None) is not None:
                        err = subtree.metadata['err']
                        if all(x.metadata.get('err', None) != err for x in subtree.children):
                            stxt = " ".join(x.get_word() for x in tree if isinstance(x, Token))
                            #if pr:
                            #    print(f'error: {err}; in "{stxt}")')
                            tex_error(f'{err}; in "{stxt}"')
                            errors.append([err, stxt])
        if len(parse_trees) != 1:
            with open('failed.txt', 'a' if len(failed) else 'w', encoding='utf-8') as f:
                f.write(f"{len(failed)+1}. {' '.join(x[0] for x in sent)}\n")
                for tree in parse_trees:
                    f.write(tree.str_for_print())
                f.write('\n=========================================================\n')
            failed.append((i, ' '.join(x[0] for x in sent), parse_trees))
        else:
            with open('parsed.txt', 'a' if len(parsed) else 'w', encoding='utf-8') as f:
                f.write(f"{len(parsed)+1}. {' '.join(x[0] for x in sent)}\n")
                f.write(parse_trees[0].str_for_print())
                f.write('\n=========================================================\n')
            parsed.append((i, ' '.join(x[0] for x in sent), parse_trees))
        with open('parse_trees.txt', 'a' if len(parsed)+len(failed) > 1 else 'w', encoding='utf-8') as f:
            f.write(f"{len(failed) + len(parsed)}. {' '.join(x[0] for x in sent)}\n")
            for tree in parse_trees:
                f.write(tree.str_for_print())
            f.write('\n=========================================================\n')
        if max_fails is not None and len(failed) >= max_fails:
            break
        if pr == 1:
            progress.update(i+1)

    if pr == 1:
        progress.finish()
        print(f'Parsing done')

    if pr:
        print("Parsed:", len(parsed))
        print("Failed:", len(failed))
        #     for i, sent, parse_trees in failed:
        #         print(f'{i}. {sent}')
        #         for tree in parse_trees:
        #             tree.print_tree()
        #         print()

        if uncommon_words:
            print('\nFollowing uncommon words were used:')
            for word, cnt in uncommon_words.items():
                if cnt > 1:
                    print(f'{word} ({cnt} times)')
                else:
                    print(word)
            print()

        if errors:
            print('Following problems detected:')
            tex_print_errors(3-pr)
            # for err, sent in errors:
            #     print(f'  {err}: "{sent}"')
            # print()

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


def test_article_dir(dir_name, max_fails: Optional[int] = 100, pr=1, pickle_dir: Optional[str] = 'results'):
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
    if pickle_dir and not os.path.exists(pickle_dir):
        # if needed, create dir to save pickle files
        os.makedirs(pickle_dir)

    return test_full_tex_file(os.path.join(dir_name, main_file), max_fails, pr, f'{pickle_dir}/{dir_last_name}.pickle' if pickle_dir else None)


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


# read command line argument: path to main tex file
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: python3 main.py <path to main tex file>')
        exit(1)
    curr_time = time.time()
    filename = sys.argv[1]
    if os.path.isfile(filename):
        test_full_tex_file(filename, max_fails=None)
    elif os.path.isdir(filename):
        test_article_dir(filename, max_fails=None, pickle_dir=None)
    else:
        print(f'{filename} is not a file or directory')
        exit(1)
    print(f'{time.time()-curr_time:.2f} seconds')
