from typing import Optional, List

from texbase import register_standard_environment, TexEnvBase, TexItem, StandardEnvironment, standard_environments
from texstream import TexStream, TexDefinedCommands, TexError


def roman_number(num):
    """
    Convert an integer to a roman numeral.
    """
    if num < 1 or num > 3999:
        raise ValueError('Roman number out of range')
    roman_numbers = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    roman_values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    res = ''
    for i in range(len(roman_values)):
        while num >= roman_values[i]:
            res += roman_numbers[i]
            num -= roman_values[i]
    return res


class TextFragmentBase(TexItem):
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


#==============================================================================
# standard LaTeX environments:
# - abstract
# - enumerate, itemize
# - formatting: center, flushleft, flushright
# - floats: figure, table
# - math: equation, equation*, align, align*, multline, multline*, eqnarray, eqnarray*
# - misc: quote, quote*
# - theorem environments defined by \newtheorem
# - proof: proof
# - verbatim
# - thebibliography


class EnumerateItem(TexItem):
    def __init__(self, parent, items: Optional[List[TexItem]] = None):
        self.parent = parent
        self.items = items if items is not None else []

    def __str__(self):
        return ''.join(map(str, self.items))


def split_list(lst, x):
    """
    Split a list into lists separated by elements equal to x.
    """
    res = []
    cur = []
    for item in lst:
        if item == x:
            res.append(cur)
            cur = []
        else:
            cur.append(item)
    res.append(cur)
    return res


class EnumerateEnvironment(TexEnvBase):
    def __init__(self, name, parent, items=None):
        super().__init__(name, parent)
        # convert tex items to enumerate items; split TextFragments by \item
        if items is None:
            items = []
        self.items = []
        for item in items:
            if isinstance(item, TextFragmentBase):
                parts = split_list(item.text, '\\item')  # list of lists of strings
                for i in range(len(parts)):
                    if i == 0 and any(not p.isspace() for p in parts[0]):
                        if not self.items:
                            print(f'Warning: some text {"".join(parts[0])} before first \\item of enumerate environment')
                            self.items.append(EnumerateItem(self, [type(item)(parts[0])]))
                        else:
                            self.items[-1].items.append(type(item)(parts[0]))
                    else:
                        if parts[i][0] == '[':
                            close = parts[i].index(']')
                            self.items.append(EnumerateItem(self, [type(item)(parts[i][close:])]))
                        else:
                            self.items.append(EnumerateItem(self, [type(item)(parts[i])]))
            else:
                if not self.items:
                    print('Warning: some text before first \\item of enumerate environment')
                    self.items.append(EnumerateItem(self, [item]))
                else:
                    self.items[-1].items.append(item)
        self.itemize = name == 'itemize'

    def __str__(self):
        tab = ' '
        level = 1
        while isinstance(self.parent, EnumerateEnvironment):
            tab += '  '
            if not self.parent.itemize:
                level += 1

        def counter(level, i):
            if self.itemize:
                return '*'
            elif level == 1:  # first level -- arabic numbers
                return str(i + 1) + '.'
            elif level == 2:  # second level -- lower case alphabetical letters a-z
                return chr(i + ord('a')) + '.'
            elif level == 3:  # third level -- roman numbers in lower case
                return roman_number(i + 1).lower() + '.'
            else:
                return counter(level-3, i)

        res = 'Enumerate environment\n'
        for i, item in enumerate(self.items):
            res += tab + counter(level, i) + ' ' + str(item) + '\n'
        return res


register_standard_environment(['enumerate', 'itemize'], [], EnumerateEnvironment)


class BibItem(TexItem):
    def __init__(self, parent, label, items: Optional[List[TexItem]] = None):
        self.parent = parent
        self.items = items if items is not None else []
        self.label = label

    def __str__(self):
        return ''.join(map(str, self.items))

    def text_fragments(self):
        pass

    def items_and_envs(self, item_classes, env_names):
        if any(isinstance(self, cls) for cls in item_classes):
            yield self


class BibliographyEnvironment(TexEnvBase):
    def __init__(self, name, parent, items=None, approx_num_items=9):
        super().__init__(name, parent)
        # convert tex items to bib items; split TextFragments by \item
        if items is None:
            items = []
        self.approx_num_items = approx_num_items
        self.item_by_label = {}
        self.items = []
        for item in items:
            if isinstance(item, TextFragmentBase):
                parts = split_list(item.text, '\\bibitem')  # list of lists of strings
                for i in range(len(parts)):
                    if i == 0 and any(not p.isspace() for p in parts[0]):
                        if not self.items:
                            print(f'Warning: some text {"".join(parts[0])} before first \\bibitem of bibliography environment')
                            self.items.append(BibItem(self, "", [type(item)(parts[0])]))
                        else:
                            self.items[-1].items.append(type(item)(parts[0]))
                    else:
                        try:
                            stream = TexStream(parts[i], TexDefinedCommands())
                            label, = stream.read_args([True], True)
                            self.items.append(BibItem(self, label, [type(item)(stream.tokens[stream.pos:])]))
                        except TexError:
                            print(f'Error: cannot read bibitem label')
                            self.items.append(BibItem(self, "", [type(item)(parts[i])]))
            else:
                if not self.items:
                    print('Warning: some text before first \\item of bibliography environment')
                    self.items.append(BibItem(self, [item]))
                else:
                    self.items[-1].items.append(item)

        for item in self.items:
            if item.label in self.item_by_label:
                print(f'Warning: duplicate bibitem label {item.label}')
            self.item_by_label[item.label] = item

    def __str__(self):
        res = 'Bibliography environment\n'
        for i, item in enumerate(self.items):
            res += '  ' + str(i + 1) + ' ' + str(item) + '\n'
        return res


register_standard_environment(['thebibliography'], [True], BibliographyEnvironment)


# Class for formatting environments
class FormattingEnvironment(TexEnvBase):
    def __init__(self, name, parent, items=None):
        super().__init__(name, parent)
        self.name = name
        self.items = items if items is not None else []

    def __str__(self):
        return 'Formatting environment\n' + ''.join(map(str, self.items))


register_standard_environment(['center', 'flushleft', 'flushright'], [], FormattingEnvironment)


class AbstractEnvironment(TexEnvBase):
    def __init__(self, name, parent, items=None):
        super().__init__(name, parent)
        self.name = name
        self.items = items if items is not None else []

    def __str__(self):
        return 'Abstract:\n' + ''.join(map(str, self.items))


register_standard_environment(['abstract'], [], AbstractEnvironment)


class VerbatimEnvironment(TexEnvBase):
    def __init__(self, name, parent, items=None):
        super().__init__(name, parent)
        self.name = name
        self.items = items if items is not None else []

    def __str__(self):
        return 'Verbatim {\n' + ''.join(map(str, self.items))+'\n}\n'


register_standard_environment(['verbatim'], [], VerbatimEnvironment, verbatim=True)


class TheoremStatement(TexEnvBase):
    def __init__(self, type, parent, items, comment, label=None):
        super().__init__(type, parent)
        self.type = type
        self.comment = comment
        self.items = items
        self.label = label
        self.proof = None

    def __str__(self):
        res = f'{self.type}'
        if self.label:
            res += f'[{self.label}]'
        if self.comment:
            res += f'({self.comment})'

        res += '\n' + ''.join(map(str, self.items))
        return res


class TheoremProof(TexEnvBase):
    def __init__(self, parent, items: List[TexItem], comment, statement: Optional[TheoremStatement] = None):
        super().__init__('proof', parent)
        self.items = items
        self.comment = comment
        self.statement = statement

    def __str__(self):
        res = f'Proof (\n{self.comment})'
        res += '\n'.join(map(str, self.items))
        res += "\nEnd Proof\n"
        return res


def create_proof_env(name, parent, items, comment=None):
    return TheoremProof(parent, items, comment)


register_standard_environment(['proof'], [False], create_proof_env)


def get_standard_env(defined_macros, name):
    # if env is theorem, return theorem env
    if thm := defined_macros.get_theorem(name):
        return StandardEnvironment(thm.env_name, [False], TheoremStatement)
    # if env is standard env, return standard env
    return standard_environments.get(name, None)