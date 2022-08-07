import re
from collections import defaultdict
from typing import List, Dict, Union

from texstream import TexStream


class TexItem:

    def print_structure(self, indent, counters: defaultdict):
        pass

    # yields all text fragments in the item
    def text_fragments(self):
        if hasattr(self, 'items'):
            for item in self.items:
                yield from item.text_fragments()

    def items_and_envs(self, item_classes, env_names):
        if any(isinstance(self, cls) for cls in item_classes):
            yield self
        elif hasattr(self, 'items'):
            for item in self.items:
                yield from item.items_and_envs(item_classes, env_names)


class TexEnvBase(TexItem):
    def __init__(self, env_name, parent):
        self.env_name = env_name
        self.parent = parent

    def print_structure(self, indent, counters: defaultdict):
        counters[self.env_name] += 1
        print(indent + f'{self.env_name} {counters[self.env_name]}')

    def items_and_envs(self, item_classes, env_names):
        if self.env_name in env_names or any(isinstance(self, cls) for cls in item_classes):
            yield self
        elif hasattr(self, 'items'):
            for item in self.items:
                yield from item.items_and_envs(item_classes, env_names)


class UnknownEnvironment(TexEnvBase):
    def __init__(self, env_name, args, items):
        super().__init__(env_name, None)
        assert all(isinstance(item, TexItem) for item in items)
        self.env_name = env_name
        self.args = args
        self.items = items

    def __repr__(self):
        res = 'Unknown environment ' + self.env_name
        if self.args:
            res += ' with args ' + ' '.join(map(str, self.args))
        res += '\n'
        for item in self.items:
            res += str(item) + '\n'
        return res

    def print_structure(self, indent, counters: defaultdict):
        counters[self.env_name] += 1
        print(indent + f'Unknown environment {self.env_name} {counters[self.env_name]}')
        for item in self.items:
            item.print_structure(indent + '  ', counters)


class StandardEnvironment:
    def __init__(self, name, args: List[Union[bool,str]], env_cls=None, verbatim=False):
        self.name = name
        self.env_cls = env_cls
        self.args = args
        self.verbatim = verbatim

    def read_args(self, text: TexStream, as_str=False):
        return text.read_args(self.args, as_str)

    def __call__(self, parent, items, args):
        return self.env_cls(self.name, parent, items, *args)


# later environment classes will be registered in this dictionary
standard_environments = {}  # type: Dict[str, StandardEnvironment]


def register_standard_environment(names, args, env_class, verbatim=False):
    for name in names:
        standard_environments[name] = StandardEnvironment(name, args, env_class, verbatim)
