import math
import re
from typing import List

from texstream import TexError, TexStream, TexDefinedCommands, tex_weak_warning, tex_warning, tex_error

tex_math_funcs = {
    r'\sin': math.sin,
    r'\cos': math.cos,
    r'\tan': math.tan,
    r'\tg':  math.tan,
    r'\cot': lambda x: 1/math.tan(x),
    r'\ctg': lambda x: 1/math.tan(x),
    r'\sec': lambda x: 1/math.cos(x),
    r'\csc': lambda x: 1/math.sin(x),
    r'\cosh': math.cosh,
    r'\ch':   math.cosh,
    r'\sinh': math.sinh,
    r'\sh':   math.sinh,
    r'\tanh': math.tanh,
    r'\coth': lambda x: 1/math.tanh(x),
    r'\exp': math.exp,
    r'\ln': math.log,
    r'\lg': lambda x: math.log10(x),
}

tex_bin_operators = {
    r'\cup',
    r'\cap',
    r'\sqcup',
    r'\sqcap',
    r'\setminus',
    r'\oplus',
    r'\ominus',
    r'\otimes',
    r'\boxplus',
    r'+',
    r'-',
    r'\times',
    r'\cdot',
    r'\ast',
    r'\star',
    r'/',
    r'\wedge',
    r'\vee',
}

tex_rel_bin_operators = {
    '>': '>',
    '<': '<',
    '=': '=',
    ':=': ':=',
    r'\coloneq': ':=',
    r'\le': '<=',
    r'\leq': '<=',
    r'\leqslant': '<=',
    r'\ge': '=>',
    r'\geq': '=>',
    r'\geqslant': '=>',
    r'\ll': '<<',
    r'\gg': '>>',
    r'\neq': '!=',
    r'\ne': '!=',
    r'\in': 'in',
    r'\ni': 'ni',
    r'\notin': '!in',
    r'\notni': '!ni',
    r'\not': '!',
    r'\approx': '~=',
    r'\sim': '~',
    r'\cong': 'isomorphic',
    r'\equiv': 'equiv',
    r'\Leftrightarrow': '<==>',
    r'\Leftarrow': '<==',
    r'\Rightarrow': '==>',
    r'\leftrightarrow': '<->',
    r'\leftarrow': '<-',
    r'\rightarrow': '->',
    r'\to': '->',
}

math_un_operators = {
    r'\neg': '-',
    r'\pm': '+-',
    r'\mp': '-+',
    '-': '-',
    '+': '+',
}

math_big_operators = {
    r'\sum': 'sum',
    r'\prod': 'prod',
    r'\int': 'int',
    r'\oint': 'oint',
    r'\bigcup': 'cup',
    r'\bigcap': 'cap',
    r'\bigwedge': 'wedge',
    r'\bigvee': 'vee',
    r'\bigoplus': 'oplus',
    r'\bigotimes': 'otimes',
    r'\bigodot': 'odot',
    r'\bigominus': 'ominus',
    r'\iint': 'iint',
    r'\iiint': 'iiint',
    # r'\min': 'min',
    # r'\max': 'max',
    # r'\sup': 'sup',
    # r'\inf': 'inf',
    # r'\lim': 'lim',
}

tex_paren_pairs = {
    (r'(', r')'),
    (r'[', r']'),
    (r'\{', r'\}'),
    (r'\lbrace', r'\rbrace'),
    (r'\lbrack', r'\rbrack'),
    (r'\langle', r'\rangle'),
    (r'|', r'|'),
    (r'\|', r'\|'),
    (r'\lvert', r'\rvert'),
    (r'\lVert', r'\rVert'),
    (r'\lceil', r'\rceil'),
    (r'\lfloor', r'\rfloor'),
}

tex_paren_pairs_ext = {
    *tex_paren_pairs,
    (r'<', r'>'),
    (r'\angle', r'\angle'),
    (r'|', r'\rangle'),
    (r'\langle', r'|'),
    (r'\lvert', r'\rangle'),
    (r'\langle', r'\rvert'),
}

tex_paren_weak_pairs = {
    ('(', ']'),
    ('[', ')'),
    (r'|', r'\rangle'),
    (r'\langle', r'|'),
    (r'\lvert', r'\rangle'),
    (r'\langle', r'\rvert'),
}
all_paren_elems = set(sum(tex_paren_pairs, ()))
left_paren_elems = set(x for x,y in tex_paren_pairs)
right_paren_elems = set(y for x,y in tex_paren_pairs)
left2right_map = {x:y for x,y in tex_paren_pairs}
right2left_map = {y:x for x,y in tex_paren_pairs}
left2right_map_ext = {x:y for x,y in tex_paren_pairs_ext}
right2left_map_ext = {y:x for x,y in tex_paren_pairs_ext}
left2right_weak_set = set(tex_paren_pairs) | {(y,x) for x,y in tex_paren_pairs} | set(tex_paren_weak_pairs)

tex_paren_sizes = {
    r'\big': 2,
    r'\Big': 3,
    r'\bigg': 4,
    r'\Bigg': 5,
}

math_fonts = {
    r'\mathrm',
    r'\mathbf',
    r'\mathit',
    r'\mathsf',
    r'\mathtt',
    r'\mathcal',
    r'\mathfrak',
    r'\mathscr',
    r'\mathbb',
}

math_decorators = {
    r'\underline',
    r'\vec',
    r'\widetilde',
    r'\widehat',
    r'\tilde',
    r'\hat',
}

math_dec_operators = {
    r'\overline',
    r'\bar',
    r'\dot',
    r'\ddot',
    r'\dddot',
}


class EqnBase:
    def estimated_size(self):
        return 1


class EqnList(EqnBase):
    def __init__(self, eqns):
        self.eqns = eqns

    def __str__(self):
        return ','.join(str(eqn) for eqn in self.eqns)

    def __repr__(self):
        return f'EqnList({self.eqns})'

    def __len__(self):
        return len(self.eqns)

    def __getitem__(self, key):
        return self.eqns[key]

    def __iter__(self):
        return iter(self.eqns)

    def __contains__(self, item):
        return item in self.eqns

    def __add__(self, other):
        return EqnList(self.eqns + other.eqns)

    def __eq__(self, other):
        return self.eqns == other.eqns

    def __ne__(self, other):
        return self.eqns != other.eqns

    def __delitem__(self, key):
        del self.eqns[key]

    def estimated_size(self):
        return max((eqn.estimated_size() for eqn in self.eqns), default=1)


class Relation(EqnBase):
    def __init__(self, rel, *args):
        self.rel = rel
        self.args = args

    def __str__(self):
        if len(self.args) == 1:
            return f'{self.rel} {self.args[0]}'
        elif len(self.args) == 2:
            return f'{self.args[0]} {self.rel} {self.args[1]}'
        else:
            return f'{self.rel}({", ".join(self.args)})'

    def __repr__(self):
        return f'Relation({self.rel}, {", ".join(self.args)})'

    def __eq__(self, other):
        return self.rel == other.rel and self.args == other.args

    def __ne__(self, other):
        return self.rel != other.rel or self.args != other.args

    def requires_lr(self):
        return len(self.args) == 2 and self.args[1] is None and self.args[0] is None

    def requires_l(self):
        return len(self.args) == 2 and self.args[0] is None

    def requires_r(self):
        return 1 <= len(self.args) <= 2 and self.args[-1] is None

    def complete(self):
        return all(arg is not None for arg in self.args)

    def estimated_size(self):
        return max((arg.estimated_size() for arg in self.args), default=1)


class Expr(EqnBase):
    def estimated_size(self):
        return 1


class SubSupScript(Expr):
    def __init__(self, base, sub=None, sup=None):
        self.base = base
        self.sub = sub
        self.sup = sup

    def __str__(self):
        res = str(self.base)
        if self.sub is not None:
            res += f'_{{{str(self.sub)}}}'
        if self.sup is not None:
            res += f'^{{{str(self.sup)}}}'
        return res

    def estimated_size(self):
        return self.base.estimated_size() + (1 if self.sub is not None and self.sub.estimated_size() > 1 or
                                                  self.sup is not None and self.sup.estimated_size() > 1 else 0)


class Variable(Expr):
    def __init__(self, name, font=None, decorations=None):
        self.name = name
        self.font = font
        self.decorations = sorted(decorations) if decorations else []

    def __str__(self):
        if self.font is None and self.decorations == []:
            return self.name
        else:
            return f'{"".join(self.decorations)}{{{self.name}}}'

    def __repr__(self):
        return f'Variable({self.name},{self.font},{self.decorations})'

    def __eq__(self, other):
        return self.name == other.name and self.font == other.font and self.decorations == other.decorations

    def __ne__(self, other):
        return self.name != other.name or self.font != other.font or self.decorations != other.decorations


class FuncNode(Expr):
    def __init__(self, name, args, increase_size=False):
        self.name = name
        self.args = args
        self.increase_size = increase_size

    def __str__(self):
        return f'{self.name}({", ".join(map(str, self.args))})'

    def __repr__(self):
        return f'FuncNode({self.name}, {", ".join(map(str, self.args))}, {self.increase_size})'

    def __eq__(self, other):
        return self.name == other.name and self.args == other.args

    def __ne__(self, other):
        return self.name != other.name or self.args != other.args

    def estimated_size(self):
        res = max(arg.estimated_size() for arg in self.args) + (1 if self.increase_size else 0)
        if isinstance(self.name, EqnBase):
            res = max(res, self.name.estimated_size())
        return res


class Parenthesis(Expr):
    def __init__(self, expr, left, right, size, text=""):
        self.expr = expr
        self.left = left
        self.right = right
        self.size = size
        if isinstance(expr, ExprTokenSeq) and len(expr.tokens) == 0:
            tex_warning(f'empty parenthesis {self} in "{"".join(text)}"')
        elif self.size == 1 and self.expr.estimated_size() > 1:
            tex_warning(f'expression {str(self.expr)} of big height is enclosed in small parentheses {left} {right} in "{"".join(text)}"')
            self.size = self.expr.estimated_size() + 1

    def __str__(self):
        return f'{self.left}{self.expr}{self.right}'

    def __repr__(self):
        return f'{self.left}{self.expr}{self.right}'

    def __eq__(self, other):
        return self.expr == other.expr and self.left == other.left and self.right == other.right

    def __ne__(self, other):
        return self.expr != other.expr or self.left != other.left or self.right != other.right

    def estimated_size(self):
        return max(self.expr.estimated_size(), self.size if isinstance(self.size, int) else 1)


class ExprTokenSeq(Expr):
    def __init__(self, tokens):
        self.tokens = tokens

    def __str__(self):
        return ''.join(str(token) for token in self.tokens)

    def __repr__(self):
        return f'ExprTokenSeq({self.tokens})'

    def __eq__(self, other):
        return self.tokens == other.tokens

    def __ne__(self, other):
        return self.tokens != other.tokens

    def estimated_size(self):
        return max((token.estimated_size() for token in self.tokens), default=1)


class ExprElement(Expr):
    def __init__(self, token, size=1):
        self.token = token
        self.size = size

    def __str__(self):
        return self.token

    def __repr__(self):
        return self.token

    def estimated_size(self):
        return self.size


class FracNode(Expr):
    def __init__(self, numer, denom):
        self.numer = numer
        self.denom = denom

    def __str__(self):
        return f'\\frac{{{self.numer}}}{{{self.denom}}}'

    def __repr__(self):
        return f'FracNode({self.numer}, {self.denom})'

    def __eq__(self, other):
        return self.numer == other.numer and self.denom == other.denom

    def __ne__(self, other):
        return self.numer != other.numer or self.denom != other.denom

    def estimated_size(self):
        return self.numer.estimated_size() + self.denom.estimated_size()


class SqrtNode(Expr):
    def __init__(self, expr, power=None):
        self.expr = expr
        self.power = power

    def __repr__(self):
        if self.power is None:
            return f'sqrt({self.expr})'
        else:
            return f'sqrt[{self.power}]({self.expr})'

    def __eq__(self, other):
        return self.expr == other.expr and self.power == other.power

    def __ne__(self, other):
        return self.expr != other.expr or self.power != other.power

    def estimated_size(self):
        return self.expr.estimated_size()


def make_node(seq: List[EqnBase]):
    if len(seq) == 1:
        return seq[0]
    else:
        return ExprTokenSeq(seq)


class MboxNode(Expr):
    def __init__(self, expr):
        self.expr = expr

    def __str__(self):
        return f'\\mbox{{{self.expr}}}'

    def __repr__(self):
        return f'MboxNode({self.expr})'

    def __eq__(self, other):
        return self.expr == other.expr

    def __ne__(self, other):
        return self.expr != other.expr


def read_sub_sup_expr(stream, tok, start_pos, text):
    stream.skip_ws()
    gr = stream.next_token == '{'
    pos0 = stream.pos
    if stream.next_token[0] == '\\':
        arg = stream.read_token()
        if arg in math_fonts:
            rarg = stream.read_required_arg()
            if len(rarg) > 1:
                tex_error(f'missed {{ }} around {"".join(stream.tokens[pos0:stream.pos])} in {"".join(stream.tokens[start_pos:stream.pos])} in "{"".join(text)}"')
            if all(x.isalnum() for x in rarg):
                return Variable(''.join(rarg), arg)
            return parse_eqn(rarg)
        else:
            if stream.next_token == '{':
                tex_error(f'missed {{ }} around {"".join(stream.tokens[pos0:stream.pos])}... in {"".join(stream.tokens[start_pos:stream.pos])}... in "{"".join(text)}"')
            return ExprElement(arg)

    arg = ''.join(stream.read_required_arg())

    if not gr and not (stream.pos+1 < len(stream.tokens) and stream.tokens[stream.pos+1] == tok):
        if stream.next_token and stream.next_token.isdigit() and arg.isdigit():
            tex_error(f'probably {{ }} missed around a number in {"".join(stream.tokens[start_pos:stream.pos])}{stream.next_token}... in "{"".join(text)}"')
        if stream.next_token and stream.next_token.isalpha() and arg.isalpha():
            tex_warning(f'possibly {{ }} missed around letters in {"".join(stream.tokens[start_pos:stream.pos])}{stream.next_token}... in "{"".join(text)}"')

    arg = parse_eqn(arg)

    return arg


# parse latex equation (extract variables, parentheses, commas, relations)
def parse_eqn(text):
    if text is None:
        return None
    stream = TexStream(text, TexDefinedCommands())
    par_stack = []
    expr_stack = [[]]
    curr_stack = [[]]
    while stream.next_token:
        if stream.next_token == '{':
            arg = parse_eqn(stream.read_group()[1:-1])
            curr_stack[-1].append(arg)
            continue

        token = stream.read_token()
        if token == ',':
            expr_stack[-1].append(ExprTokenSeq(curr_stack[-1]) if len(curr_stack[-1]) != 1 else curr_stack[-1][0])
            curr_stack[-1] = []
        elif token == '\\left':
            par = stream.read_token()
            if par not in left2right_map_ext:
                tex_warning(f'unknown left parenthesis {par} in "{"".join(text)}"')
            par_stack.append((par, 0))
            expr_stack.append([])
            curr_stack.append([])
        elif token == '\\right':
            rpar = stream.read_token()
            if rpar not in right2left_map_ext:
                tex_warning(f'unknown right parenthesis {rpar} in "{"".join(text)}"')
            while par_stack and par_stack[-1][1] != 0:
                par, _ = par_stack.pop()
                if par != '|':
                    tex_warning(f'no pair for opening parenthesis {par} in "{"".join(text)}"')
                if expr_stack[-1]:
                    expr_stack[-1].append(ExprTokenSeq([ExprElement(par)] + curr_stack[-1]))
                    expr_stack[-2] += expr_stack[-1]
                else:
                    curr_stack[-2] += [ExprElement(par)] + curr_stack[-1]
                curr_stack.pop()
                expr_stack.pop()

            if not par_stack:
                tex_error(f'no pair for closing parenthesis \\right{rpar} in "{"".join(text)}"')
                curr_stack[-1].append(ExprElement(rpar))
                continue

            par, size = par_stack.pop()
            expr_stack[-1].append(ExprTokenSeq(curr_stack[-1]) if len(curr_stack[-1]) != 1 else curr_stack[-1][0])
            curr_stack.pop()
            curr_stack[-1].append(Parenthesis(EqnList(expr_stack[-1]) if len(expr_stack[-1]) != 1 else expr_stack[-1][0], par, rpar, 0, text))
            expr_stack.pop()
            # print(f'{par} {rpar}')
            if par != '.' and rpar != '.' and (par, rpar) not in tex_paren_pairs_ext:  # left2right_map_ext.get(par,None) != rpar and right2left_map_ext.get(rpar,None) != par:
                if (par, rpar) in tex_paren_weak_pairs:
                    tex_weak_warning(f'rare parenthesis pair {par} {rpar} in "{"".join(text)}"')
                else:
                    tex_warning(f'left parenthesis {par} and right parenthesis {rpar} do not match in "{"".join(text)}"')
        elif (is_par := token in all_paren_elems) or (token[-1] in 'rlRL' and token[:-1] in tex_paren_sizes) or token in tex_paren_sizes:
            if is_par:
                left = token in left_paren_elems and (not token in right_paren_elems or not par_stack or par_stack[-1][0] != right2left_map[token] or par_stack[-1][1] != 1)
                par = token
                size = 1
            elif token in tex_paren_sizes:
                par = stream.read_token()
                left = par in left_paren_elems and (not par in right_paren_elems or not par_stack or par_stack[-1][0] != right2left_map[par] or par_stack[-1][1] != 1)
                size = tex_paren_sizes[token]
            else:
                left = token[-1] in 'lL'
                par = stream.read_token()
                size = tex_paren_sizes[token[:-1]]
            if left:
                if stream.next_token and stream.next_token in '_^':
                    curr_stack[-1].append(ExprElement(token))
                    if token != '|':
                        tex_warning(f'_ or ^ after opening parenthesis {par} in "{"".join(text)}"')
                else:
                    par_stack.append((par, size))
                    expr_stack.append([])
                    curr_stack.append([])
            else:
                if not par_stack:
                    tex_warning(f'no pair for closing parenthesis {par} in "{"".join(text)}"')
                    curr_stack[-1].append(ExprElement(par))
                    continue

                lpar, left_size = par_stack.pop()
                if left2right_map[lpar] != par:
                    if (lpar, par) in tex_paren_weak_pairs:
                        tex_weak_warning(f'rare parenthesis pair {lpar} {par} in "{"".join(text)}"')
                    else:
                        if lpar != '|' or not par_stack or (par_stack[-1][0], par) not in (tex_paren_weak_pairs | tex_paren_pairs):
                            tex_warning(f'left parenthesis {lpar} and right parenthesis {par} do not match in "{"".join(text)}"')
                        else:
                            if expr_stack[-1]:
                                expr_stack[-1].append(ExprTokenSeq([ExprElement(par)] + curr_stack[-1]))
                                expr_stack[-2] += expr_stack[-1]
                            else:
                                curr_stack[-2] += [ExprElement(lpar)] + curr_stack[-1]
                            curr_stack.pop()
                            expr_stack.pop()
                            lpar, left_size = par_stack.pop()
                if left_size != size:
                    tex_warning(f'left and right parentheses {lpar} {par} have different size in "{"".join(text)}"')

                expr_stack[-1].append(ExprTokenSeq(curr_stack[-1]) if len(curr_stack[-1]) != 1 else curr_stack[-1][0])
                curr_stack.pop()
                curr_stack[-1].append(Parenthesis(EqnList(expr_stack[-1]) if len(expr_stack[-1]) != 1 else expr_stack[-1][0], lpar, par, max(left_size, size), text))
                expr_stack.pop()
        elif token == '\\frac' or token == '\\dfrac':
            numer, denom = stream.read_args([True, True])
            numer = parse_eqn(numer)
            denom = parse_eqn(denom)
            curr_stack[-1].append(FracNode(numer, denom))
        elif token == '\\sqrt':
            pow, expr = stream.read_args([False, True])
            if pow is not None:
                pow = parse_eqn(pow)
            expr = parse_eqn(expr)
            curr_stack[-1].append(SqrtNode(expr, pow))
        elif token == '&' or token == '\\\\':
            continue
        elif token in '_^':
            start_pos = stream.pos-2
            stream.skip_ws()
            supscript = None
            subscript = None
            arg = read_sub_sup_expr(stream, token, start_pos, text)
            if token == '_': subscript = arg
            else: supscript = arg
            if stream.next_token and stream.next_token in '_^':
                second_token = stream.read_token()
                second_arg = read_sub_sup_expr(stream, second_token, start_pos, text)
                if second_token == '_': subscript = second_arg
                else: supscript = second_arg
            if len(curr_stack[-1]) == 0:
                SubSupScript(ExprElement(''), subscript, supscript)
                tex_error(f'open parenthesis before subscript "{"".join(stream.text[start_pos:stream.pos])}" in "{"".join(text)}"')
            else:
                curr_stack[-1][-1] = SubSupScript(curr_stack[-1][-1], subscript, supscript)
        elif token in math_fonts:
            arg = stream.read_required_arg()
            if all(x.isalnum() for x in arg):
                arg = Variable(''.join(arg), token)
            else:
                arg = parse_eqn(arg)
            curr_stack[-1].append(arg)
        elif token in math_decorators or token in math_dec_operators:
            arg = parse_eqn(stream.read_required_arg())
            if isinstance(arg, Variable):
                arg.decorations.append(token)
            else:
                arg = FuncNode(token, [arg])
            curr_stack[-1].append(arg)
        elif token == '\\mbox' or token == '\\text':
            arg = ''.join(stream.read_required_arg())
            curr_stack[-1].append(MboxNode(arg))
        elif token.startswith('\\big') or token in math_big_operators:
            curr_stack[-1].append(ExprElement(token, 2))
        else:
            curr_stack[-1].append(ExprElement(token))

    # check whether all parenthesis are closed
    while len(par_stack) > 0:
        par, was_left = par_stack.pop()
        if par != '|':
            tex_warning(f'no pair for opening parenthesis {par} in "{"".join(text)}"')
        if expr_stack[-1]:
            expr_stack[-1].append(ExprTokenSeq([ExprElement(par)] + curr_stack[-1]))
            expr_stack[-2] += expr_stack[-1]
        else:
            curr_stack[-2] += [ExprElement(par)] + curr_stack[-1]
        curr_stack.pop()
        expr_stack.pop()

    if len(curr_stack[-1]) >= 1:
        expr_stack[-1].append(ExprTokenSeq(curr_stack[-1]) if len(curr_stack[-1]) != 1 else curr_stack[-1][0])
    return EqnList(expr_stack[-1]) if len(expr_stack[-1]) != 1 else expr_stack[-1][0]
