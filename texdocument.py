import os
import re
from collections import defaultdict
from typing import Optional, List, Dict

from parse_formula import parse_eqn
from stdenvs import get_standard_env, TextFragmentBase, AbstractEnvironment, BibliographyEnvironment
from texbase import TexEnvBase, TexItem, UnknownEnvironment, register_standard_environment
from texstream import TexDefinedCommands, TexStream, TexEnvironment, TexTheoremDef, TexMacro, TexError, tokens_to_text


def read_macro_args(text: TexStream, num_args, optional_args):
    arg_types = [True] * num_args
    if optional_args is not None:
        for i, arg in enumerate(optional_args):
            if arg is not None:
                arg_types[i] = False
        #arg_types[0] = False
    res = text.read_args(arg_types)
    if optional_args is not None:
        for i, arg in enumerate(optional_args):
            if arg is not None and res[i] is None:
                res[i] = arg
        # res[0] = optional_arg
    return res


def read_unknown_macro_args(text: TexStream):
    text.skip_ws()
    args = []
    while text.next_token in ['[','{']:
        if text.next_token == '[':
            args.append(text.read_optional_arg())
        else:
            args.append(text.read_required_arg())
    return args


# remove comments processes \input and \include and expand macros
def preprocess_tex_file(text: str, commands: TexDefinedCommands, env_stack=None):
    # remove tex comments:  % ... \n
    if isinstance(text, str):
        # find comments started from % which is not escaped
        text = re.sub(r'(?<!\\)%.*?\n', '', text)
        #text = re.sub(r'%.*\n', '', text)
        # remove tex comments of the form \iffalse ... \fi
        # text = re.sub(r'(?<!\\)\\iffalse.*\\fi', '', text, flags=re.DOTALL)
    text = TexStream(text, commands)
    old_pos = 0
    if env_stack is None:
        env_stack = []
    output = []
    while text.next_token is not None:
        old_pos = text.pos
        tok = None
        while (tok := text.read_token()) is not None:
            if tok[0] == '\\' and tok[1].isalpha():
                break
        else:
            break
        output.append(text.tokens[old_pos:text.pos-1])
        old_pos = text.pos-1
        command = tok[1:]
        if command == 'input' or command == 'include':
            # find {filename}
            try:
                filename, = text.read_args([True], as_str=True)  # type: str
                if '.' not in filename:
                    filename += '.tex'
                if commands.dir_path is not None:
                    filename = f'{commands.dir_path}/{filename}'
                with open(filename, 'r') as f:
                    text.replace(old_pos, text.pos, f.read())
            except TexError:
                print('Error: missing filename in \\' + command)
        elif command == 'newcommand' or command == 'renewcommand':
            try:
                # find *{name}[num of args][optional arg]{args}{body}
                star, name, num_args, *optional_arg, body = text.read_args(['*', True, False, False, False, False, False, False, True], as_str=[1,2])
                if not num_args:
                    num_args = 0
                else:
                    num_args = int(num_args)
                commands.add_macro(TexMacro(name[1:], num_args, optional_arg, body))
            except TexError:
                print('Error: cannot read arguments in \\' + command)
        elif command == 'def':
            try:
                # find \name#1...#n{body}
                name, = text.read_args([True], as_str=True)
                nargs = 0
                while text.next_token[0] == '#':
                    nargs += 1
                    arg = text.read_token()
                    if int(arg[1:]) != nargs:
                        print(f'Error: argument {arg} number mismatch in \\def')
                body, = text.read_args([True])
                commands.add_macro(TexMacro(name[1:], nargs, None, body))
            except TexError:
                print('Error: cannot read arguments in \\def')
        elif command == 'let':
            try:
                # find \name[[<spaces>]=]body
                name, = text.read_args([True], as_str=True)
                text.skip_ws()
                if text.next_token == '=':
                    text.read_token()
                old_command, = text.read_args([True], as_str=False)
                old_command = preprocess_tex_file(old_command, commands, env_stack)
                commands.add_macro(TexMacro(name[1:], 0, None, old_command))
            except TexError:
                print('Error: cannot read arguments in \\let')
        elif command == 'newenvironment' or command == 'renewenvironment':
            # find *{name}[num of args][optional arg]{begin def}{end def}
            try:
                star, name, num_args, optional_arg, begin_def, end_def = text.read_args(['*', True, False, False, True, True], as_str=[1,2])
                if not num_args:
                    num_args = 0
                else:
                    num_args = int(num_args)
                commands.add_environment(TexEnvironment(name, num_args, optional_arg, begin_def, end_def))
            except TexError:
                print('Error: cannot read arguments in \\' + command)
        elif command == 'newtheorem':
            # find *{name}[inherits]{body}
            try:
                star, env_name, inherits, name = text.read_args(['*', True, False, True], as_str=True)
                commands.add_theorem(TexTheoremDef(env_name, name, inherits))
            except TexError:
                print('Error: cannot read arguments in \\' + command)
        elif command == 'begin':  # begin environment
            # find {name}
            try:
                env_name, = text.read_args([True], as_str=True)
                commands.begin_scope()
                env_stack.append([env_name, None])
                if env := commands.get_environment(env_name):
                    args = read_macro_args(text, env.num_args, env.optional_arg)
                    if args is not None:
                        # expand macro in args
                        # args = [preprocess_tex_file(arg, commands) for arg in args]
                        expanded = preprocess_tex_file(env.expand_beg(args), commands, env_stack)
                        output.append(expanded)
                        env_stack[-1][1] = preprocess_tex_file(env.expand_end(args), commands, env_stack)
                    else:
                        print('Error: cannot read arguments of environment ' + env_name)
                else:  # save \begin{env} in output
                    output.append(text[old_pos:text.pos])
            except TexError:
                print('Error: cannot read environment name in \\' + command)
        elif command == 'end':  # end environment
            # find {name}
            try:
                env_name, = text.read_args([True], as_str=True)
                if env_stack and env_stack[-1][0] == env_name:
                    commands.end_scope()
                    if env_stack[-1][1] is not None:
                        output.append(env_stack[-1][1])
                    else:
                        output.append(text[old_pos:text.pos])
                    env_stack.pop()
                else:
                    print('Error: cannot end environment ' + env_name + ' because it is not started')
            except TexError:
                print('Error: cannot read environment name in \\' + command)
        elif macro := commands.get_macro(command):  # user-defined macro
            args = read_macro_args(text, macro.num_args, macro.optional_arg)
            if args is not None:
                # expand macro in args
                # args = [preprocess_tex_file(arg, commands) for arg in args]
                expanded_macro = preprocess_tex_file(macro.expand(args), commands, env_stack)
                output.append(expanded_macro)
            else:
                print('Error: cannot read arguments of macro \\' + command)
        elif command in ['author', 'authors', 'title']:
            # find {body}
            try:
                body, = text.read_args([True], as_str=True)
                setattr(commands, command, body)
            except TexError:
                print('Error: cannot read body of \\' + command)
        elif command == 'footnote':
            # find {body}
            try:
                body, = text.read_args([True], as_str=True)
                body = preprocess_tex_file(body, commands)
                output.append([' ', commands.add_footnote(TextFragment(body)), ' '])
            except TexError:
                print('Error: cannot read body of \\' + command)
        elif command == 'iffalse':
            while (tok:=text.read_token()) != '\\fi':
                if tok is None:
                    print('Error: \\iffalse is not closed')
                    break
        else:
            output[-1].append(tok)
        old_pos = text.pos

    output.append(text.tokens[old_pos:])
    return sum(output, [])


formatting_tex_macros_1arg = {"\\textbf", "\\textit", "\\texttt", "\\textsc", "\\textsf", "\\textsl", "\\textrm",
                              "\\textup", "\\textmd", "\\textnormal", "\\emph"}
formatting_tex_macros_no_arg = {"\\textcompwordmark", "\\sf", "\\sl", "\\,", "\\!", "\\'", '\\"', "\\`", "\\^", "\\~", "\\-","\\\\"}


math_env_names = {"equation", "equation*", "align", "align*", "eqnarray", "eqnarray*", "multline", "multline*",
                  "tikzcd", "tikzcd*", "gather", "gather*"}


class MathEnv(TexEnvBase):
    def __init__(self, parent, env_name, items=None):
        super().__init__(parent, env_name)
        if items is not None and len(items) != 1 or not isinstance(items[0], TextFragment):
            raise TexError(f'Error: invalid content in math environment {env_name}')
        self.frag = items[0] if items else None
        self.eqn = parse_eqn(self.frag.text_str()) if self.frag else None


register_standard_environment(list(math_env_names), [], MathEnv)


class TextFragment(TextFragmentBase):
    def __init__(self, text: str):
        super().__init__(text)
        self.sentenses = []
        self.parsed_sentenses = []

    def parse_envs(self, commands: TexDefinedCommands):
        # works like preprocessing; finds all environments in text and builds hierarchy of environments
        text = TexStream(self.text, commands)
        verbatim = False
        env_stack = [['', None, [], []]]
        while text.next_token is not None:
            old_pos = text.pos
            while text.next_token is not None:
                tok = text.read_token()
                if tok in ('\\begin', '\\end'):
                    break
            else:
                if any(not x.isspace() for x in text.tokens[old_pos:]):
                    env_stack[-1][-1].append(TextFragment(text.tokens[old_pos:]))
                break
            curr_pos = text.pos-1
            try:
                env_name, = text.read_args([True], as_str=True)
                if tok == '\\begin':
                    if verbatim:
                        continue
                    # create text fragment from current position to begin of environment
                    if any(not x.isspace() for x in text.tokens[old_pos:curr_pos]):
                        env_stack[-1][-1].append(TextFragment(text.tokens[old_pos:curr_pos]))

                    env = get_standard_env(commands, env_name)
                    if env is None:
                        print('Warning: unknown environment ' + env_name)
                        args = read_unknown_macro_args(text)
                        env_stack.append([env_name, None, args, []])
                    else:
                        if env.verbatim or env_name in math_env_names:
                            verbatim = True
                        args = env.read_args(text)
                        env_stack.append([env_name, env, args, []])
                else:
                    assert tok == '\\end'
                    # create text fragment from current position to end of environment
                    if any(not x.isspace() for x in text.tokens[old_pos:curr_pos]):
                        env_stack[-1][-1].append(TextFragment(text.tokens[old_pos:curr_pos]))
                    if env_stack and env_stack[-1][0] == env_name:
                        name, env, args, items = env_stack[-1]
                        env_stack.pop()
                        if env is not None:  # if environment is standard
                            env_stack[-1][-1].append(env(None, items, args))
                        else:
                            env_stack[-1][-1].append(UnknownEnvironment(name, args, items))
                        verbatim = False
                    elif not verbatim:
                        print('Error: cannot end environment ' + env_name + ' because it is not started')
            except TexError:
                print('Error: cannot read environment name')
                break
        if len(env_stack) > 1:
            # close all environments
            while len(env_stack) > 1:
                name, env, args, items = env_stack.pop()
                print(f'Error: environment {name} not closed')
                if env is not None:
                    env_stack[-1][-1].append(env(None, items, args))
                else:
                    env_stack[-1][-1].append(UnknownEnvironment(name, items, args))

        return env_stack[0][-1]

    def text_fragments(self):
        yield self

    def text_str(self):
        return self.text if isinstance(self.text, str) else ''.join(self.text)

    def __str__(self):
        return self.text_str()

    def remove_formatting_macros(self):
        # remove formatting macros
        res_tokens = []
        text = TexStream(self.text, TexDefinedCommands())
        try:
            while text.next_token is not None:
                tok = text.read_token(skip_scope=True)
                if tok in formatting_tex_macros_1arg:
                    arg, = text.read_args([True])
                    res_tokens.extend(arg)
                elif tok in formatting_tex_macros_no_arg:
                    if tok[-1].isalpha():
                        text.skip_ws()
                else:
                    res_tokens.append(tok)
        except TexError as e:
            print(f'Error: {e}')
        return tokens_to_text(res_tokens)


section_types = ['document', 'part', 'chapter', 'section', 'subsection', 'subsubsection', 'paragraph']


class TexSection(TexItem):
    def __init__(self, type, name, items, label=None):
        self.type = type
        self.name = name
        self.items = items
        self.label = label

    def __str__(self):
        return self.name + '\n' + ''.join(map(str, self.items()))

    def __iter__(self):
        return self.items.__iter__()

    @property
    def level(self):
        return section_types.index(self.type)

    def parse_envs(self, commands: TexDefinedCommands):
        for i in range(len(self.items)):
            if isinstance(self.items[i], TexSection):
                self.items[i].parse_envs(commands)
            elif isinstance(self.items[i], TextFragment):
                self.items[i] = self.items[i].parse_envs(commands)  # returns list of TexItems
        new_items = []
        for i in range(len(self.items)):
            if isinstance(self.items[i], list):
                new_items.extend(self.items[i])
            else:
                new_items.append(self.items[i])
        self.items = new_items
        return self.items

    def print_structure(self, indent, counters: Dict[str, int]):
        counters[self.type] += 1
        # reset counters for subsections
        level = section_types.index(self.type)
        for i in range(level+1, len(section_types)):
            counters[section_types[i]] = 0
        # print all numbers (section.subsection.subsubsection)
        num = ''
        for i in range(section_types.index('section'), level+1):
             num += str(counters[section_types[i]]) + '.'
        print(indent + self.type + ' ' + num + ': ' + self.name)
        for item in self.items:
            item.print_structure(indent + '  ', counters)


def parse_tex_structure(text: str):
    text = TexStream(text, TexDefinedCommands())
    stack = [TexSection('document', '', [])]
    old_pos = 0
    while text.next_token is not None:
        while text.next_token is not None:
            tok = text.read_token()
            if tok[0] == '\\' and tok[1:] in section_types[1:]:
                break
        else:
            break
        curr_pos = text.pos-1
        sec_type = tok[1:]
        try:
            star, caption = text.read_args(['*', True], as_str=True)
            if any(not x.isspace() for x in text.tokens[old_pos:curr_pos]):
                stack[-1].items.append(TextFragment(text.tokens[old_pos:curr_pos]))
            sec = TexSection(sec_type, caption, [])
            while sec.level <= stack[-1].level:
                stack.pop()
            stack[-1].items.append(sec)
            stack.append(sec)
            old_pos = text.pos
            # search \label{<label>} after spaces
            if text.skip_ws().next_token == '\\label':
                text.read_token()
                sec.label, = text.read_args([True], as_str=True)
        except TexError:
            print('Error: cannot read section name')

    if any(not x.isspace() for x in text.tokens[old_pos:text.pos]):
        stack[-1].items.append(TextFragment(text.tokens[old_pos:text.pos]))
    return stack[0]


def find_label(text: str):  # find \label{<label>} in text
    match = re.search(r'\\label[{](.*?)[}]', text)
    if match:
        return match.group(1)
    else:
        return None


# some macro definitions for formatting macros
predefined_tex_macros = r"""
\newcommand{\textbf}[1]{#1}
\newcommand{\textit}[1]{#1}
\newcommand{\textsc}[1]{#1}
\newcommand{\textsf}[1]{#1}
\newcommand{\em}{}
\newcommand{\bf}{}
\newcommand{\it}{}
\newcommand{\sc}{}
\newcommand{\IEEEpeerreviewmaketitle}{}
\newcommand{\maketitle}{}
\newcommand{\IEEEPARstart}[2]{#1#2}
% following commands should be replaced
\newcommand{\cite}[2][]{@cite }
\newcommand{\ref}[1]{@ref }
\newcommand{\eqref}[1]{@eqref }
\newcommand{\Cref}[1]{@cref }
\newcommand{\label}[1]{}
\newcommand{\setcounter}[2]{}
\newcommand{\citep}[3][][]{\cite[#1]{#3}}
\newcommand{\citet}[2][]{\cite[#1]{#2}}
\newcommand{\institute}[1]{}
\newcommand{\date}[1]{}
\newcommand{\offprints}[1]{}
\newcommand{\abstract}{}
\newcommand{\email}[1]{}
\newcommand{\affil}[1]{}
\newcommand{\affiliation}[1]{}
\newcommand{\pacs}[1]{}
\newcommand{\vspace}[1]{}
\newcommand{\hspace}[1]{}
\newcommand{\vfill}{}
\newcommand{\hfill}{}
\newcommand{\newline}{}
\newcommand{\pagebreak}{}
\newcommand{\S}{Paragraph}
\newcommand{\usetikzlibrary}[2][]{}
\newcommand{\tikzset}[1]{}
\newcommand{\theoremstyle}[1]{}
\newcommand{\hphantom}[1]{}
\newcommand{\vphantom}[1]{}
\newcommand{\bibliographystyle}[1]{}
\newcommand{\bibliography}[1]{}
"""


# Structure of the tex document
class TexDocument(TexItem):
    def __init__(self, /, text=..., filename=None):
        self.filename = filename
        self.commands = None
        self.theorem_envs = ["theorem", "lemma", "corollary", "definition", "remark", "example", "exercise", "claim"]
        self.title = ""
        self.authors = []
        self.abstract = ""
        self.body = TextFragment("")
        self.keywords = ""
        self.references = []
        self.footnotes = []
        self.figures = {}
        self.tables = {}
        self.equations = []
        self.sections = None
        self.appendices = None

        if text is ...:  # read file
            with open(filename, 'rb') as f:
                text = f.read().decode('utf-8', errors='ignore')

        self.parse_text(text)

    def parse_text(self, text):
        # get path to file
        if self.filename is None:
            path = None
        else:
            path = os.path.dirname(self.filename)

        # First step is preprocessing of the text: remove comments and expand user-defined macros
        self.commands = TexDefinedCommands(dir_path=path)
        # load predefined macros
        preprocess_tex_file(predefined_tex_macros, self.commands)
        # preprocess text
        text = tokens_to_text(preprocess_tex_file(text, self.commands))
        with open('preprocessed.tex', 'w') as f:
            f.write(text)
        # parse text
        # find begin document using regex
        begin_match = re.search(r'\\begin[ \t\n\r]*[{]document[}]', text)
        if begin_match is None:
            begin_document = -1
        else:
            begin_document = begin_match.end()
        # find end document using regex
        end_match = re.search(r'\\end[ \t\n\r]*[{]document[}]', text)
        if end_match is None:
            end_document = -1
        else:
            end_document = end_match.start()

        preamble = ""
        if begin_document != -1 and end_document != -1:
            preamble = text[:begin_match.start()]
            text = text[begin_document:end_document]
        else:
            print('Error: cannot find begin{document} or end{document}')

        # remove tables and figures from text and save them in self.tables and self.figures
        table_regexp = r'\\begin[{]table[*]?[}](.*)\\end[{]table[*]?[}]'
        for match in re.finditer(table_regexp, text, re.DOTALL):
            table_name = find_label(match.group(1))
            if table_name is None:
                # find first unused table name
                for i in range(1, 10000):
                    table_name = f'table:{i}'
                    if table_name not in self.tables:
                        break
            self.tables[table_name] = match.group(1)
        text = re.sub(table_regexp, '', text, flags=re.DOTALL)
        figure_regexp = r'\\begin[{]figure[*]?[}](.*)\\end[{]figure[*]?[}]'
        for match in re.finditer(figure_regexp, text, re.DOTALL):
            figure_name = find_label(match.group(1))
            if figure_name is None:
                # find first unused figure name
                for i in range(1, 10000):
                    figure_name = f'figure:{i}'
                    if figure_name not in self.figures:
                        break
            self.figures[figure_name] = match.group(1)
        text = re.sub(r'\\begin[ \t\n\r]*[{]figure[*]?[}].*?\\end[ \t\n\r]*[{]figure[*]?[}]', '', text, flags=re.DOTALL)
        # remove labels
        text = re.sub(r'\\label[ \t\n\r]*[{].*?[}]', '', text)
        # replace ~ by space
        text = text.replace('~', ' ')

        # find \appendix command and split text into main part and appendix
        appendix_pos = text.find(r'\appendix')
        if appendix_pos != -1:
            appendix = text[appendix_pos + len(r'\appendix'):]
            text = text[:appendix_pos]
        else:
            appendix = ""
        # split text into sections
        self.sections = parse_tex_structure(text)
        self.appendices = parse_tex_structure(appendix)

        self.sections.parse_envs(self.commands)
        self.appendices.parse_envs(self.commands)

        self.title = self.commands.title
        self.authors = self.commands.authors or self.commands.author

        self.footnotes = self.commands.footnotes

        # find abstract in sections
        for item in self.sections.items:
            if isinstance(item, AbstractEnvironment):
                self.abstract = item
                break

        bibligraphy = list(self.items_and_envs([BibliographyEnvironment], ()))
        if len(bibligraphy) > 0:
            self.references = bibligraphy[0]

        return text

    def print_structure(self, indent, counters: defaultdict):
        print(indent + "TexDocument:")
        print(indent + f"  title: {self.title}")
        print(indent + f"  authors: {self.authors}")
        if self.abstract:
            print(indent + f"  abstract: {self.abstract}")
        if self.keywords:
            print(indent + "  keywords: " + self.keywords)
        print(indent + "  sections:")
        self.sections.print_structure(indent + "    ", counters)
        print(indent + "  appendices:")
        self.appendices.print_structure(indent + "    ", counters)
        # if self.references:
        #     print(indent + "  references:")
        #     for ref in self.references:
        #         print(indent + "    " + ref)

    def print_document_info(self):
        self.print_structure("", defaultdict(lambda: 0))

    def text_fragments(self):
        yield from self.sections.text_fragments()
        yield from self.appendices.text_fragments()
        for footnote in self.footnotes:
            yield from footnote.text_fragments()

    def items_and_envs(self, item_classes, env_names):
        yield from self.sections.items_and_envs(item_classes, env_names)
        yield from self.appendices.items_and_envs(item_classes, env_names)
        for footnote in self.footnotes:
            yield from footnote.items_and_envs(item_classes, env_names)
