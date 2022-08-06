import re
from typing import List, Dict, Optional, Union


class TexError(Exception):
    pass


class TexStream:
    def __init__(self, text: Union[str, List[str]], defined_commands):
        self.text = text
        self.defined_commands = defined_commands
        self.pos = 0
        # token is a symbol or \[a-zA-Z]+ or \[^a-zA-Z] or #[1-9] or #
        if type(text) == str:
            token_regexp = re.compile(r'[^\\#]|\\[a-zA-Z]+|\\[^a-zA-Z]|#[1-9]?')
            self.tokens = token_regexp.findall(text)
        else:
            self.tokens = text

    def _read_token(self):
        if self.pos >= len(self.tokens):
            return None
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def read_token(self):
        token = self._read_token()
        if token == '{':
            self.defined_commands.begin_scope()
        elif token == '}':
            self.defined_commands.end_scope()
        return token

    def read_group(self):  # read { ... } or one symbol; {} can be nested;
        old_pos = self.pos
        tok = self._read_token()
        if tok is None:
            raise TexError('unexpected end of input')
        if tok == '}':
            raise TexError(f'Unexpected }} at position {old_pos}')
        if tok != '{':
            return [tok]
        opened = 1
        while opened > 0:
            next = self._read_token()
            if next == '{':
                opened += 1
            elif next == '}':
                opened -= 1
            if next is None:
                raise TexError(f'Unexpected end of input at position {old_pos}')
        return self.tokens[old_pos:self.pos]

    def read_required_arg(self):
        self.skip_ws()
        res = self.read_group()
        if res[0] == '{':
            return res[1:-1]
        return res

    def skip_ws(self):  # skip whitespace
        while self.pos < len(self.tokens) and self.tokens[self.pos].isspace():
            self.pos += 1
        return self

    @property
    def next_token(self):
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def read_optional_arg(self):  # reads [ ... ] or one symbol; [] inside {} group are ignored;
        old_pos = self.pos
        self.skip_ws()
        if self.next_token != '[':
            self.pos = old_pos
            return None
        while self.read_group() != [']']:
            pass
        return self.tokens[old_pos+1:self.pos-1]

    def read_args(self, arg_types: List[Union[bool,str]], as_str=None):  # arg_types[i] = True if arg[i] is required,
        # arg[i] is optional if False,
        # arg[i] is optional 1-character string if str
        args = []
        for arg_type in arg_types:
            if arg_type is True:
                args.append(self.read_required_arg())
            elif arg_type is False:
                args.append(self.read_optional_arg())
            else:
                assert isinstance(arg_type, str)
                if self.next_token == arg_type:
                    args.append(self.read_token())
                else:
                    args.append(None)
        if as_str:
            if as_str is True:
                as_str = range(len(args))
            for i in as_str:
                if args[i] is not None:
                    args[i] = tokens_to_text(args[i])
        return args

    def replace(self, start_pos, end_pos, text):
        if not isinstance(text, TexStream):
            text = TexStream(text, self.defined_commands)
        self.tokens[start_pos:end_pos] = text.tokens
        self.pos = start_pos
        return self

    def __getitem__(self, item):
        return self.tokens[item]


def tokens_to_text(tokens: List[str]):
    tok1 = tokens
    for i in range(len(tok1)-1):
        if tok1[i][0] == '\\' and tok1[i+1][0].isalpha():
            tok1[i] += ' '
    return ''.join(tok1)


def macro_replace_args(text: List[str], num_args, args, name_for_error):
    # replace # by arg[0], #i by arg[i-1] for i>=1
    if not num_args:
        return text
    output = []
    if len(args) != num_args:
        raise Exception(f'Wrong number of arguments for {name_for_error}')
    for token in text:
        if token == '#':
            output.extend(args[0])
        elif token[0] == '#':
            try:
                output.extend(args[int(token[1:])-1])
            except IndexError:
                raise Exception(f'Wrong number of arguments for {name_for_error}')
        else:
            output.append(token)
    return output


# TeX macros
class TexMacro:
    def __init__(self, name, num_args, optional_arg, body):
        self.name = name
        self.num_args = num_args
        self.optional_arg = optional_arg if type(optional_arg) == list else [optional_arg] if optional_arg is not None else []
        if sum(1 for x in self.optional_arg if x is not None) > self.num_args:
            raise Exception(f'Too many optional arguments for {name}')
        while self.optional_arg and self.optional_arg[-1] is None:
            self.optional_arg.pop()
        self.body = body

    def __str__(self):
        res = self.name
        if self.num_args:
            res += f'[{self.num_args}]'
            if self.optional_arg:
                res += f'[{self.optional_arg}]'
        res += f'{{{self.body}}}'
        return res

    def expand(self, args):
        # for i in range(len(self.optional_arg)):
        #    if args[i] is None and self.optional_arg[i] is not None:
        #        args[i] = self.optional_arg[i]
        return macro_replace_args(self.body, self.num_args, args, f"macro {self.name}")


# class for storing definitions by newenvironment
class TexEnvironment:
    def __init__(self, name, num_args, optional_arg, begin_def, end_def):
        self.name = name
        self.num_args = num_args
        self.optional_arg = optional_arg if type(optional_arg) == list else [optional_arg] if optional_arg is not None else []
        if sum(1 for x in self.optional_arg if x is not None) > self.num_args:
            raise Exception(f'Too many optional arguments for {name}')
        while self.optional_arg and self.optional_arg[-1] is None:
            self.optional_arg.pop()
        self.begin_def = begin_def
        self.end_def = end_def

    def __str__(self):
        res = self.name
        if self.num_args:
            res += f'[{self.num_args}]'
            if self.optional_arg:
                res += f'[{self.optional_arg}]'
        res += f'{{{self.begin_def}}}{{{self.end_def}}}'
        return res

    def expand_beg(self, args):
        return macro_replace_args(self.begin_def, self.num_args, args, f"environment {self.name}")

    def expand_end(self, args):
        return macro_replace_args(self.end_def, self.num_args, args, f"environment {self.name}")


# theorem environment introduces by newtheorem or newtheorem* command
# \newtheorem{<env_name>}[<inherirs>]{<name>}
class TexTheoremDef:
    def __init__(self, env_name, name, inherits=None):
        self.name = name
        self.env_name = env_name
        self.inherits = inherits

    def __str__(self):
        res = f'\\newtheorem{{{self.env_name}}}'
        if self.inherits:
            res += f'[{self.inherits}]'
        res += f'{{{self.name}}}'
        return res


# Stores all defined macros, environments and theorems
class TexDefinedCommands:
    class Scope:
        def __init__(self, parent: 'Optional[TexDefinedCommands.Scope]' = None):
            self.parent = parent
            self.macros = {}
            self.environments = {}
            self.theorems = {}

        def add_macro(self, macro):
            self.macros[macro.name] = macro

        def add_environment(self, environment):
            self.environments[environment.name] = environment

        def add_theorem(self, theorem: TexTheoremDef):
            self.theorems[theorem.env_name] = theorem

        def get_macro(self, name):
            if name in self.macros:
                return self.macros[name]
            if self.parent:
                return self.parent.get_macro(name)
            return None

        def get_environment(self, name):
            if name in self.environments:
                return self.environments[name]
            if self.parent:
                return self.parent.get_environment(name)
            return None

        def get_theorem(self, name):
            if name in self.theorems:
                return self.theorems[name]
            if self.parent:
                return self.parent.get_theorem(name)
            return None

    def __init__(self, dir_path=None):
        self.scopes = [TexDefinedCommands.Scope(None)]
        self.dir_path = dir_path
        self.labels = {}  # label name -> labeled TexItem
        self.title = None  # title of the document
        self.author = None  # author of the document
        self.authors = None  # authors of the document
        self.footnotes = []

    def add_footnote(self, text):
        self.footnotes.append(text)
        return f'footnote_{len(self.footnotes)}'

    def add_macro(self, macro):
        self.scopes[-1].add_macro(macro)

    def add_environment(self, environment):
        self.scopes[-1].add_environment(environment)

    def add_theorem(self, theorem):
        self.scopes[-1].add_theorem(theorem)

    def begin_scope(self):
        self.scopes.append(TexDefinedCommands.Scope(self.scopes[-1]))

    def end_scope(self):
        if len(self.scopes) == 1:
            raise TexError('Cannot end scope, already at top level')
        self.scopes.pop()

    def get_macro(self, name):
        return self.scopes[-1].get_macro(name)

    def get_environment(self, name):
        return self.scopes[-1].get_environment(name)

    def get_theorem(self, name):
        return self.scopes[-1].get_theorem(name)




