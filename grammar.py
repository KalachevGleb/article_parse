def multi_arg_enumeration(root, left, op, right, comma=None):
    if not op.is_leaf():
        print(f"Error: '{op.get_word()}' is not a leaf")
    op = op.get_word()
    if op in ('and', 'or'):
        if comma is None and left.metadata.get('comma', 0):
            print(f"Error: a comma expected before '{op}' in '{left.text} {op} {right.text}'")
            root.metadata['err'] = f"a comma expected before '{op}'"
        if comma is not None and not comma.metadata.get('comma', 0):
            print(f"Warning: a comma maybe redundant before '{op}' in '{left.text} {op} {right.text}'")
            root.metadata['warn'] = f"a comma maybe redundant before '{op}'"


def check_many(root, noun, verb):
    vbz = verb.metadata.get('vbz')
    if vbz == 1:
        if noun.metadata.get('many', 0) == 1:
            print(f"Warning: '{verb.get_word()}' should not be in 3rd person singular form after {noun.text}")
            root.metadata['err'] = f"'{verb.get_word()}' should not be in 3rd person singular"
    elif vbz == 0:
        if noun.metadata.get('many', 0) == 0:
            print(f"Warning: '{verb.get_word()}' should be in 3rd person singular form after {noun.text}")
            root.metadata['err'] = f"'{verb.get_word()}' should be in 3rd person singular form"


math_complete_verbs = {'fix', 'put', 'let', 'consider', 'denote', 'define', 'introduce', 'find',
                       'imagine', 'note', 'observe', 'suppose'}


def check_complete(verb):
    if verb.get_word() not in math_complete_verbs:
        verb.metadata['incomplete'] = 1


def check_modal_verb(root, modal, verb):
    if verb.ps not in ('VB', 'VBP') or verb.get_word() in ('is', 'are', 'am'):
        root.metadata['err'] = f"{verb.get_word()} in wrong form after '{modal.get_word()}', should be infinitive"


def check_let_clause(root, let, verb, subj=None):
    check_modal_verb(root, let, verb)


def check_the(root, art, noun):
    if noun.metadata.get('the', None) is not None:
        if art.pos == 'ART':
            root.metadata['err'] = f"'{art.text}' is redundant before {noun.text}"
        else:
            root.metadata['err'] = f"article before noun is redundant after {art.text}"


def check_the_the_nn(root, *args, nn1=None, nn2=None):
    if nn1 is not None and nn1.metadata.get('gsp',None) == 'NN' and nn1.metadata.get('the', None) != 1:
        nn1.metadata['err'] = f"'the' before '{nn1.get_word()}' expected in the ... the ... comparison '{root.text}'"
    if nn2 is not None and nn2.metadata.get('gsp',None) == 'NN' and nn2.metadata.get('the', None) != 1:
        nn2.metadata['err'] += f"  'the' before '{nn2.get_word()}' expected in the ... the ... comparison '{root.text}'"


grammar_str = [
    ("(‘ ‘){leaf:1,ps:QOPEN}", None),
    ("(’ ’){leaf:1,ps:QCLOSE}", None),

    (r"""(_SOL [note*] that){ps:VB}""", None),
    (r"""(induces|implies|coincides|corresponds|varies|decreases|increases|belongs){ps:VBZ}""", None),
    (r"""(commute|denote|recall|decrease|increase){ps:VB}""", None),
    (r"""(explicit|quantum|interesting){ps:JJ}""", None),
    (r"""(code|fiber|complex|soundness){ps:NN}""", None),
    (r"""(codes){ps:NNS}""", None),
    (r"""(random){ps:JJ}""", None),
    (r"""(at random*){leaf:1}""", None),
    (r"""(in turn){ps:RB}""", None),

    (r"""(whether){ps:WHETHER}""", None),
    (r"""(if|iff){ps:IF}""", None),
    (r"""(if and only if){ps:IF}""", None),
    (r"""(a|an|the){ps:ART}""", None),
    (r"""(that){ps:THAT}""", None),
    (r"""(then){ps:THEN}""", None),
    (r"""(there){ps:THERE}""", None),
    (r"""(which){ps:WHICH}""", None),
    (r"""(since){ps:SINCE}""", None),
    (r"""(due to){ps:IN, due_to:1}""", None),
    (r"""(such that){ps:SUCHTHAT}""", None),
    (r"""(of){ps:OF,of_type:1}""", None),
    (r"""(up to){ps:IN}""", None),
    (r"""(because of){ps:IN}""", None),
    (r"""(cf){ps:VB}""", None),
    (r"""(almost all*){ps:DT}""", None),
    (r"""(as well as*){ps:IN,left_nn:1,right_nn:1,left_vb:1,right_vb:1,right_whole:1,before:0,leaf:1}""", None),
    (r"""(as well*){ps:RB}""", None),

    (r"""(@ [cite|ref|eqref|cref]){ps:null}""", None),
    (r"""(_{}* _FOOTNOTE){footnote:1}""", None),
    (r"""(@ cite*){ps:CITE, subj:1, aux:1,leaf:1}""", None),
    (r"""(@ ref|eqref*){ps:CD, subj:1, ref:1, aux:1,leaf:1}""", None),
    (r"""(@ cref*){ps:NNP, subj:1, ref:1, aux:1,leaf:1}""", None),
    (r"""(\( _NNP* \)){ps:IABBR}""", None),
    (r"""(_FW _NNP*){leaf:1}""", None),
    (r"""(_NNP* _ET_AL){leaf:1}""", None),
    (r"""(' s*){ps: POS, pos: 1}""", None),

    (r"""(_EQNP* th){ps:JJ,leaf:1}""", None),
    (r"""(_EQNP _{}*){eqnp:1,leaf:1}""", None),

    (r"""(_CD* % ){leaf:1}""", None),

    (r"""(_NN|NNP*){gsp: NN, subj: 1, many: 0}""", None),
    (r"""(_NNS|NNPS*){gsp: NN, subj: 1, many: 1}""", None),
    (r"""(_EQN*){gsp:NN, subj:1}""", None),
    (r"""(is|are|am|do|does*){gsp: VB, whole:0}""", None),
    (r"""(!is|are|am|was|were|be|been [_VBG*]){subj: 1, many: 0}""", None),
    (r"""(is|are|was|were* _VBG{gsp:=VB,subj:=0})""", None),
    (r"""(by [_VBG*]){subj: 0, gsp: VB}""", None),
    ("(_VBZ){vbz:1}", None),
    ("(_VB|VBP){vbz:0}", None),
    ("(is|does|was|has){vbz:1}", None),
    ("(am|are|do|were|have){vbz:0}", None),
    ("(can|may|could|might|will|would|shall|should|may|might){vbz:_}", None),
    (r"""(_VB|VBP|VBD|VBZ*){gsp: VB, whole:0}""", None),
    (r"""(_VB|VBP|VBD|VBZ|VBN|VBG*){accept_to:1}""", None),
    (r"""(be|been|is|are|am|was|were* _VBD|VBN){passive: 1}""", None),
    (r"""(is|are|am|was|were|can|could|will|would|have|has|had|did|does|do* not){neg:1, leaf:1}""", None),
    (r"""(of|in|on|over*){of_type: 1}""", None),
    (r"""(_{gsp:NN}* _POS){nnpos: 1}""", None),
    (r"""(he|she|it|this*){many:0}""", None),
    (r"""(they|we|I|you|these*){many:1}""", None),
    (r"""(he|she|it|they|we|I|you*){can_p:1,no_prep:1}""", None),
    (r"""(this|these*){can_p:1}""", None),
    (r"""(he|she|it|they|we|I|you [_VBN*]){ps:VBD}""", None),
    (r"""(_NNP* -|-- _NNP){many_names:1,leaf:1}""", None),
    (r"""(_SOL [_NN|NNS -|-- _NNP*]){ps:NNP,many_names:1,leaf:1}""", None),
    (r"""(_{many_names:1}* _IABBR){new_abbr:1,leaf:1}""", None),
    (r"""(_JJ|DT [one*]){ps:NN,gsp:NN,subj:1,many:0}""", None),
    (r"""([one*] _MD){ps:NN,gsp:NN,subj:1,many:0,restrict_nn_join:1}""", None),
    (r"""(one*){subj:1,many:0}""", None),

    (r"""(for* a long time){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}""", None),
    (r"""(in* fact){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}""", None),
    (r"""(to* this end){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}""", None),
    (r"""(on* the one hand){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}""", None),
    (r"""(on* the other hand){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}""", None),
    (r"""(in* this case){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}""", None),
    (r"""(in* particular){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}""", None),
    (r"""(for* example){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}""", None),
    (r"""([as* follows] :|.){ps:STD,gsp:RB,has_prep:1,leaf:1}""", None),
    (r"""(_SOL [in* words]){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}""", None),
    (r"""(yet another*){leaf:1}""", None),
    (r"""(turn|turns|turned* out){leaf:1}""", None),
    (r"""(that is){ps:IE}""", None),
    (r"""(with respect to){ps:IN}""", None),
    (r"""(in order to*){leaf:1}""", None),
    (r"""([in general] !_{gsp:NN}){ps:STD,gsp:RB,vvod:1,has_prep:1,leaf:1}""", None),

    (r"""#""", None),
    (r"""(\( _{}* \)){optional:1}""", None),
    (r"""(_IE{zpt:null}* ,){zpt:1}""", None),

    (r"""#""", None),
    (r"""_RBS _JJ*""", None),
    (r"""least _JJ*""", None),
    (r"""(at least|most _{subj:1}*)""", None),
    (r"""(_NNP _NNP*){cmpxnnp: 1}""", None),
    (r"""(_CD* -|-- _CD){cdrange:1}""", None),
    (r"""#""", None),
    (r"""(_CITE* , _CITE){comma:1}""", None),
    (r"""_CITE* and _CITE""", None),
    (r"""_CITE{comma:1}* , and _CITE""", None),
    (r"""(_CD{ref:1}* , _CD{ref:1}){comma:1}""", None),
    (r"""_CD{ref:1}* and _CD{ref:1}""", None),
    (r"""_CD{ref:1,comma:1}* , and _CD{ref:1}""", None),
    (r"""(_NNP _{subj:1,aux:null}*){named: 1}""", None),
    (r"""(_NNP* _CITE){cite: 1}""", None),
    (r"""(_NNP* _CD){numbered: 1}""", None),
    (r"""(_{named:1}* _CD){numbered: 1}""", None),
    (r"""(_{gsp:NN}* due to _NNP|NNPS){author:1}""", None),
    (r"""(more|less|fewer|greater* than _CD|EQN){ps:JJ, compare:1}""", None),
    (r"""(_RB _CD*)""", None),

    (r"""#""", None),
    (r"""(`` _{}* ''){quote:1, leaf:1}""", None),
    (r"""(_QOPEN _{}* _QCLOSE){quote:1, leaf:1}""", None),
    (r"""(` _{}* '){quote:1, leaf:1}""", None),
    (r"""(\( see* \)){ps:CITE}""", None),
    (r"""(not _IN*){neg:1}""", None),
    (r"""(at most|least _CD|EQN*){ps:JJ}""", None),

    (r"""(_{gsp:VB,what:null} [that*]){that:can_join}""", None),
    (r"""^(_JJ* , _JJ){comma:1}""", None),
    (r"""^(_JJ* and _JJ){and: 1}""", None),
    (r"""^(_JJ* or _JJ){or: 1}""", None),
    (r"""^(_JJ and _JJ*){and: 1}""", None),
    (r"""^(_JJ or _JJ*){or: 1}""", None),

    (r"""^(_JJ* \( _JJ \)){comment:1}""", None),

    (r"""(the _JJR*){ps:JJRT,whole:0}""", None),
    (r"""(the _RBR*){ps:RBRT,whole:0}""", None),
    (r"""(_{gsp:NN}){jjrt_arg:1}""", None),
    (r"""(_{gsp:VB,whole:1}){jjrt_arg:1}""", None),

    (r"""^(_JJ|JJS|JJR _NN|NNS|NNP|NNPS*){left_jj:1}""", None),
    (r"""^(_NN|NNS|NNP|NNPS* _JJ|JJS|JJR{compare:1})""", None),
    (r"""_{nnpos:1} _{subj:1}*""", None),
    (r"""(its|his|her|their|our|my _{subj:1}*){the:1}""", None, check_the),

    (r"""(_{gsp:NN}* _CITE){cite:1}""", None),
    (r"""(_{gsp:VB}* _CITE){cite:1}""", None),

    (r"""(_{gsp:NN}* _EQN){with_formula:1}""", None),

    (r"""^(_NN* \( _NN|NNP \)){comment:1}""", None),
    (r"""^(_NNS* \( _NNS|NNPS \)){comment:1}""", None),
    (r"""^(_{gsp:NN}* \( _{gsp:NN} \)){comment:1}""", None),
    (r"""^(_NN* \( _IE _NN|NNP \)){comment:1}""", None),
    (r"""^(_NNS* \( _IE _NNS|NNPS \)){comment:1}""", None),
    (r"""^(_{gsp:NN}* \( _IE _{gsp:NN} \)){comment:1}""", None),
    (r"""^([_NN* , _IE _NN|NNP] _EOL){comment:1}""", None),
    (r"""^([_NNS* , _IE _NNS|NNPS] _EOL){comment:1}""", None),
    (r"""^([_{gsp:NN}* , _IE _{gsp:NN}] _EOL){comment:1}""", None),
    (r"""^(_NN* , _IE _NN|NNP ,){comment:1}""", None),
    (r"""^(_NNS* , _IE _NNS|NNPS ,){comment:1}""", None),
    (r"""^(_{gsp:NN}* , _IE _{gsp:NN} ,){comment:1}""", None),

    (r"""(_VBN _NN|NNS*){left_jj:1}""", None),

    (r"""(_{subj:1} [_IN{left_nn:1}]){has_left:1}""", None),
    (r"""(_{gsp:VB} [_IN{left_vb:1}]){has_left:1}""", None),
    (r"""(_IN{left_vb:1,right_nn:1,has_left:null,before:null} _{gsp:NN} [_{gsp:NN}*] _MD|VB|VBP|VBZ{whole:0}){obj:1}""", None),
    (r"""(_{gsp:NN,has_right:null} _{gsp:NN,the:null,obj:null,restrict_nn_join:null,left_jj:null}*){no_of:1}""", None),
    (r"""(_EQN|CD{has_right:null} _{gsp:NN,the:null,obj:null,restrict_nn_join:null}*){no_of:1}""", None),
    (r"""(_{subj:1}* _IABBR){new_abbr: 1}""", None),

    (r"""(_JJ|PDT|DT|ART|CD [_VBG|VBN _{subj:1}*])""", None),
    (r"""(_JJ|PDT|DT|ART|CD [_VBD{ps:=VBN} _{subj:1}*])""", None),

    (r"""(both _EQN|VBG|NN|NNS|NNP|NNPS{many:1}*){gsp:NN,both:1,subj:1,left_jj:1}""", None),
    (r"""(_DT|PDT _VBG|NN|NNS|NNP|NNPS*){gsp:NN,subj:1,left_jj:1}""", None),
    (r"""(the _NN|NNS|NNP|NNPS|EQN*){the:1,left_jj:1}""", None, check_the),
    (r"""(a|an _NN|NNP|EQN*){the:0,left_jj:1}""", None, check_the),
    (r"""(a|an _NNS|NNPS*){the:0,left_jj:1, err:'a|an before plural'}""", None),
    (r"""(some _NN|NNS*){exists:1,quant:1,left_jj:1}""", None),
    (r"""(some|any|all|each|many _EQN*){quant:1,left_jj:1}""", None),
    (r"""(any|each|one of _{subj:1}*){quant:1,many:0,left_jj:1}""", None),
    (r"""(some|all|many of _{subj:1}*){quant:1,many:1,left_jj:1}""", None),
    (r"""(any|each|one of us|them*){quant:1,subj:1,gsp:NN,left_jj:1,many:0}""", None),
    (r"""(some|all|many of us|them*){quant:1,subj:1,gsp:NN,left_jj:1,many:1}""", None),
    (r"""(for [all _{subj:1,many:1}*]){all:1,quant:1,left_jj:1}""", None),
    (r"""(for [any|each|every _{subj:1}*]){all:1,quant:1,left_jj:1}""", None),
    (r"""(the _{subj:1}*){the:1,left_jj:1}""", None),

    (r"""_{subj:1}* _VBN|VBG{subj_last:1}""", None),

    (r"""(in* comparison to){ps:null,join_right:subj}""", None),

    (r"""(and* , _RB{leaf:1} ,)""", None),
    (r"""(or* , _RB{leaf:1} ,)""", None),

    (r"""(_RB and _RB*){and: 1}""", None),

    (r"""_RB _{gsp:VB}*""", None),
    (r"""^(_RB _VBG|VBN*)""", None),
    (r"""^(_RB _JJ|JJR|RB|RBR*)""", None),

    # (r"""(!and|or [_NN|NNS|NNP|NNPS{and:null,or:null} , _NN|NNS|NNP|NNPS*]){comma: 1}""", None),
    (r"""(_NN|NNS|NNP|NNPS{comma:1,and:null,or:null} ,{arg:=comma} and _NN|NNS|NNP|NNPS*){and: 1, many: 1}""", None, multi_arg_enumeration),
    (r"""(_NN|NNS|NNP|NNPS{comma:1,and:null,or:null} ,{arg:=comma} or _NN|NNS|NNP|NNPS*){or: 1}""", None, multi_arg_enumeration),
    (r"""^(!and|or{arg:=_} [_NN|NNS|NNP|NNPS{the:null} and _NN|NNS|NNP|NNPS{the:null}*]){and: 1, many: 1}""", None, multi_arg_enumeration),
    (r"""^(!and|or{arg:=_} [_NN|NNS|NNP|NNPS{the:{0,1}} and _NN|NNS|NNP|NNPS{the:{0,1}}*]){and: 1, many: 1}""", None, multi_arg_enumeration),
    (r"""^(!and|or{arg:=_} [_NN|NNS|NNP|NNPS{the:null} or _NN|NNS|NNP|NNPS{the:null}*]){or: 1}""", None, multi_arg_enumeration),
    (r"""^(!and|or{arg:=_} [_NN|NNS|NNP|NNPS{the:{0,1}} or _NN|NNS|NNP|NNPS{the:{0,1}}*]){or: 1}""", None, multi_arg_enumeration),
    (r"""^([_NN|NNS|NNP|NNPS{the:null,and:null,or:null}* and _NN|NNS|NNP|NNPS{the:null}] !_VB|VBD|VBZ|VBP){and: 1, many: 1}""", None, multi_arg_enumeration),
    (r"""^([_NN|NNS|NNP|NNPS{the:{0,1},and:null,or:null}* and _NN|NNS|NNP|NNPS{the:{0,1}}]  !_VB|VBD|VBZ|VBP){and: 1, many: 1}""", None, multi_arg_enumeration),
    (r"""^([_NN|NNS|NNP|NNPS{the:null,and:null,or:null}* or _NN|NNS|NNP|NNPS{the:null}]  !_VB|VBD|VBZ|VBP){or: 1}""", None, multi_arg_enumeration),
    (r"""^([_NN|NNS|NNP|NNPS{the:{0,1},and:null,or:null}* or _NN|NNS|NNP|NNPS{the:{0,1}}]  !_VB|VBD|VBZ|VBP){or: 1}""", None, multi_arg_enumeration),

    (r"""(!and|or{arg:=_} [_VBN{and:null,or:null} , _VBN*]){comma: 1}""", None),
    (r"""(_VBN{comma:1,and:null,or:null} ,{arg:=comma} and _VBN*){and: 1}""", None, multi_arg_enumeration),
    (r"""(_VBN{comma:1,and:null,or:null} ,{arg:=comma} or _VBN*){or: 1}""", None, multi_arg_enumeration),
    (r"""(_VBN{and:null,or:null} and _VBN*){and: 1}""", None, multi_arg_enumeration),
    (r"""^(_VBN{and:null,or:null}* and _VBN){and: 1}""", None, multi_arg_enumeration),
    (r"""(_VBN or _VBN*){or: 1}""", None, multi_arg_enumeration),
    (r"""^(_VBN* or _VBN){or: 1}""", None, multi_arg_enumeration),

    (r"""^(!and|or [_EQN{and:null,or:null} , _EQN*]){comma: 1, many: 1}""", None),
    (r"""^(_EQN{and:null,or:null}* , _EQN){comma: 1, many: 1}""", None),
    (r"""^(_EQN{comma:1,and:null,or:null} ,{arg:=comma} and _EQN*){and: 1, many: 1}""", None, multi_arg_enumeration),
    (r"""^(_EQN{comma:1,and:null,or:null} ,{arg:=comma} or _EQN*){or: 1}""", None, multi_arg_enumeration),
    #(r"""^(_EQN{comma:null} ,{arg:=comma} and _EQN*){and: 1, many: 1, err: "redundant ',' before and"}""", None, multi_arg_enumeration),
    #(r"""^(_EQN{comma:null} ,{arg:=comma} or _EQN*){or: 1, err: "redundant ',' before or"}""", None, multi_arg_enumeration),
    (r"""^(!and|or{arg:=_} [_EQN and _EQN*]){and: 1, many: 1}""", None, multi_arg_enumeration),
    (r"""^(!and|or{arg:=_} [_EQN or _EQN*]){or: 1}""", None, multi_arg_enumeration),
    (r"""^(_EQN* and _EQN){and: 1, many: 1}""", None, multi_arg_enumeration),
    (r"""^(_EQN* or _EQN){or: 1}""", None, multi_arg_enumeration),
    (r"""(either _{or:1}*)""", None),

    (r"""([_{subj:1}*] !_{of_type:1}){of_arg:1}""", None),
    (r"""(_{gsp:NN}* _{of_type:1} _{of_arg:1}){of_arg:1,has_right:1}""", None),

    (r"""^(_{and:1}* , respectively){resp:1}""", None),
    (r"""^(_{or:1}* , respectively){resp:1}""", None),
    (r"""^(_{and:1}* respectively){resp:1,err:"',' expected before 'respectively'"}""", None),
    (r"""^(_{or:1}* respectively){resp:1,err:"',' expected before 'respectively'"}""", None),

    (r"""^(_{gsp:VB,what:null}* _DT){what:1}""", None),
    (r"""^(_VBD{what:null}* _NN){what:1,canbe:NN}""", None),
    (r"""^(_VBD{what:null}* _NNS){what:1,canbe:NNS}""", None),
    (r"""^(_VBD{what:null}* _NNP){what:1,canbe:NNP}""", None),
    (r"""^(_VBD{what:null}* _NNPS){what:1,canbe:NNPS}""", None),
    (r"""^(_{gsp:VB,what:null}* _{subj:1}){what:1}""", None),
    (r"""^(_{gsp:VB,whole:0}* : _EQN){what:1}""", None),
    (r"""^(_{gsp:NN}* : _{subj:1,comma:1})""", None),
    (r"""^(_{gsp:NN}* : _{subj:1,and:1})""", None),
    (r"""(_{subj:1}*){can_be_what:1}""", None),
    (r"""(us|me|him|her|it|them*){can_be_what:1}""", None),
    (r"""^([_VBG{what:null}* _RB] _{can_be_what:1})""", None),
    (r"""^(_VBG{what:null}* _{subj:1}){what:1}""", None),
    (r"""^(_VBG{what:null}* _JJ|JJR|JJS){what_jj:1}""", None),
    (r"""^(_{gsp:VB,what:null}* us|me|him|her|it|them|itself|himself|myself|herself|themselves){whom:1}""", None),
    (r"""^(_VBG{what:null}* us|me|him|her|it|them|itself|himself|myself|herself|themselves){whom:1}""", None),

    (r"""_{gsp:VB}* _VBG""", None),
    (r"""(_MD _{gsp:VB,whole:0}*){vbz:_}""", None, check_modal_verb),
    (r"""has|have{leaf:1}* _VBN|VBD""", None),
    (r"""(has|have{leaf:1}* _{gsp:VB,whole:0}){err: "possible wrong verb form after 'has/have'"}""", None),

    (r"""^(is|are|am|be|been|was|were{leaf:1}* _JJ|JJR|JJS|VBN){jprop:1}""", None),
    (r"""^(is|are|am|be|been|was|were{leaf:1}* _VB|VBZ|VBP){jprop:1,err: verb in wrong form after (is|are|am|be|been|was|were)}""", None),
    (r"""^(do|does|did|was|were{leaf:1}* _VB|VBP|VBZ|VBD)""", None),

    (r"""(^_{accept_to:1}* _TO _{subj:1}){with_to:1}""", None),
    (r"""(^_{accept_to:1}* _TO us|me|him|her|it|them){with_to:1}""", None),
    (r"""(_SOL [_{gsp:NN}* _TO _{subj:1}])""", None),
    (r"""(_SOL [_{gsp:NN}* _TO us|me|him|her|it|them])""", None),
    (r"""(_CD _{subj:1,the:null}*){count:1}""", None),
    (r"""(_IN [_EQN{leaf:1} _{subj:1,many:1,the:null}*]){count:1}""", None),
    (r"""(_IN [_EQN{leaf:1} _{subj:1,many:0,the:null}*]){count:1,err:"'s' expected after unknown count specified by formula"}""", None),
    (r"""(one of _{subj:1}*){many:0}""", None),

    (r"""^_{gsp:{VB,NN},that:null}* from _{subj:1} to _{subj:1}""", None),
    (r"""^_{gsp:{VB,NN},that:null}* from _JJ to _JJ""", None),
    (r"""^_{gsp:{VB},that:null}* _RB _IN _{subj:1}""", None),
    (r"""^_{gsp:{VB,NN},that:null}* _IN _{subj:1}""", None),
    (r"""^_{gsp:{VB}}* _RB _RP _{subj:1}""", None),
    (r"""^_{gsp:{VB}}* _RP _{subj:1}""", None),
    (r"""^_{gsp:{VB},that:null}* , like _{subj:1}""", None),
    (r"""^_{gsp:{NN},that:null}* _OF _{subj:1}""", None),
    (r"""^_{gsp:{VB,NN},that:null}* _IN|OF _DT of _{subj:1}""", None),
    (r"""^_{gsp:VB}* _STD{gsp:RB,has_prep:1}""", None),
    (r"""^_{gsp:{VB,JJ}}* _{join_right:{nn,subj}} _{subj:1}""", None),
    (r"""^_{gsp:{VB,NN},that:null}* _IN|OF _VBG""", None),
    (r"""^_{gsp:NN}* _VBG|VBN _IN|OF|TO _{subj:1}""", None),
    (r"""^_{gsp:NN}* , _VBG|VBN _IN|OF|TO _{subj:1}""", None),
    (r"""^_{gsp:NN}* _VBG|VBN{what:1}""", None),
    (r"""^_{gsp:NN}* _VBG|VBN{with_to:1}""", None),
    (r"""^_VBG* _TO|IN|OF _{subj:1}""", None),
    (r"""^(_{gsp:VB}* _TO _VB)""", None),

    (r"""^(_{gsp:VB,what:null}* _{subj:1}){what:1}""", None),
    (r"""^(_{gsp:VBG,what:null}* _{subj:1}){what:1}""", None),
    (r"""^(_{gsp:VB,what:null}* _JJ){what:1}""", None),
    (r"""^(_{gsp:VB,post_prep:null}* _IN{nontran:1}){post_prep:1}""", None),

    (r"""^(_RB _IN|OF*)""", None),

    (r"""(_JJR{than:null}* than _{subj:1}){than:1}""", None),
    (r"""(more|less{than:null}* _JJ than _{subj:1}){ps:JJR, than:1}""", None),
    (r"""(more|less{than:null}* _RB than _{subj:1}){than:1}""", None),
    (r"""(_JJR{than:null}* then _{subj:1}){than:1,err:"'than' expected instead of 'then'"}""", None),
    (r"""^(_{subj:1}* _JJR{than:1})""", None),
    (r"""^(_RBR _JJ|JJR|RB*)""", None),

    (r"""^(_NN|NNS|NNP|NNPS{with_formula:null,with_number:null}* _CD){with_number:1}""", None),
    (r"""^(_{subj:1}* but _{subj:1}){butsubj:1}""", None),

    (r"""(_JJ _EQN*){subj:1,gsp:NN}""", None),

    (r"""(!_IN* _CITE){cite:1}""", None),
    #
    (r"""(_VB , _VB*){comma: 1}""", None),
    (r"""(_VB{comma:1} ,{arg:comma} and _VB*){and: 1}""", None, multi_arg_enumeration),
    (r"""(_VB{comma:1} ,{arg:comma} or _VB*){or: 1}""", None, multi_arg_enumeration),
    (r"""(_VB and _VB*){and: 1}""", None, multi_arg_enumeration),
    (r"""(_VB or _VB*){or: 1}""", None, multi_arg_enumeration),
    (r"""(_VB or _VB*){or: 1}""", None, multi_arg_enumeration),
    (r"""^(_VBG{gsp:VB}* and _VBG{gsp:=VB,accept_to:=1}){and:1}""", None, multi_arg_enumeration),
    (r"""^(_VBG{gsp:VB}* or _VBG{gsp:=VB,accept_to:=1}){or:1}""", None, multi_arg_enumeration),
    (r"""(_VBG{gsp:null}* and _VBG{gsp:null}){and:1}""", None, multi_arg_enumeration),
    (r"""(_VBG{gsp:null}* or _VBG{gsp:null}){or:1}""", None, multi_arg_enumeration),

    #
    (r"""let{has_verb:null->1}* _{gsp:VB,whole:0}""", None, check_let_clause),
    (r"""let{has_verb:null->1}* _{gsp:NN,arg:subj} _{gsp:VB,whole:0}""", None, check_let_clause),
    #
    (r"""([_{canbe:NN}*] _{gsp:VB,whole:0}){ps:NN,subj:1,many:0}""", None),
    (r"""([_{canbe:NNS}*] _{gsp:VB,whole:0}){ps:NNS,subj:1,many:1}""", None),
    (r"""([_{canbe:NNP}*] _{gsp:VB,whole:0}){ps:NNP,subj:1,many:0}""", None),
    (r"""([_{canbe:NNPS}*] _{gsp:VB,whole:0}){ps:NNPS,subj:1,many:1}""", None),

    (r"""([_{subj:1}* _VBD{whole:0,ps:=VBN}] _VB|VBZ|VBP{whole:0})""", None),
    (r"""(no{arg:_} _{subj:1} _{gsp:VB,whole:0}*){whole:1,neg:1}""", None, check_many),
    (r"""(!_IN{left_vb:1,right_nn:1,before:null} [_{subj:1} _{gsp:VB,whole:0}*]){whole:1}""", None, check_many),
    (r"""(!_IN{left_vb:1,right_nn:1,before:null} [_{can_p:1} _{gsp:VB,whole:0}*]){whole:1}""", None, check_many),
    (r"""([_{can_p:1,no_prep:1} _{gsp:VB,whole:0}*]){whole:1}""", None, check_many),
    (r"""(_JJ{many:=0} is|are{whole:0}*){whole:1,def:1}""", None, check_many),
    (r"""(_SOL{arg:_} [_VB|VBP|VBD|VBZ{gsp:VB,whole:0}*]){whole:1}""", None, None, check_complete),

    (r"""^(_{gsp:VB,whole:1}* ,|and therefore|hence|thus|so _{at_end:1}){dep:hence,at_end:1}""", None),
    (r"""(therefore|thus|hence _{gsp:VB,whole:1}*){hence:1}""", None),
    (r"""(therefore|thus|hence , _{gsp:VB,whole:1}*){hence:1}""", None),
    (r"""(_SOL [then _{gsp:VB,whole:1}*]){hence:1}""", None),
    #
    (r"""(_SOL [_RB , _{gsp:VB}*])""", None),
    (r"""(, _RB , _{gsp:VB}*)""", None),
    (r"""_{vvod:1} , _{gsp:VB}*""", None),
    (r"""(_{vvod:1} _{gsp:VB}*){err:"',' expected"}""", None),
    #
    (r"""(_JJRT{whole:0}* _{jjrt_arg:1,arg:=nn1} _JJRT{whole:0} _{jjrt_arg:1,arg:=nn2}){whole:1}""", None, check_the_the_nn),
    (r"""(_RBRT|JJRT{whole:0}* _{gsp:VB,whole:1} _RBRT|JJRT{whole:0} _{gsp:VB,whole:1}){whole:1}""", None),
    (r"""(_JJRT{whole:0}* _{jjrt_arg:1,arg:=nn1} ,{arg:=_} _JJRT{whole:0} _{jjrt_arg:1,arg:=nn2}){whole:1}""", None, check_the_the_nn),
    (r"""(_RBRT|JJRT{whole:0}* _{gsp:VB,whole:1} , _RBRT|JJRT{whole:0} _{gsp:VB,whole:1}){whole:1}""", None),
    (r"""(_JJRT|RBRT{whole:1}* ,{arg:=_} _JJRT|RBRT{whole:0} _{jjrt_arg:1,arg:=nn1}){whole:1}""", None, check_the_the_nn),
    (r"""(_JJRT|RBRT{whole:1}* ,{arg:=_} and _JJRT|RBRT{whole:0} _{jjrt_arg:1,arg:=nn1}){whole:1}""", None, check_the_the_nn),

    #
    (r"""_IN{right_nn:1,left_vb:1} _{subj:1} _VB|VBP|VBZ|VBD{whole:1}*""", None),
    (r"""(_SOL [_IN _{subj:1} , _VB|VBP|VBZ|VBD{whole:1}*])""", None),
    (r"""(_{subj:1} [, _IN _{subj:1} , _VB|VBP|VBZ|VBD{whole:0}*])""", None),
    (r"""(which [, _IN _{subj:1} , _VB|VBP|VBZ|VBD{whole:0}*])""", None),

    (r"""(there _{gsp:VB}*){whole:1}""", None),
    (r"""(_{gsp:NN}*){accept_to:1}""", None),
    (r"""(_{gsp:VB}* _RB)""", None),
    #
    (r"""(_VB , _VB*){comma: 1}""", None),
    (r"""(_VB{comma:1} , and _VB*){and: 1}""", None),
    (r"""(_VB{comma:1} , or _VB*){or: 1}""", None),
    (r"""^(_{gsp:VB,whole:0}* and _{gsp:VB,whole:0}){and: 1}""", None, multi_arg_enumeration),
    (r"""^(_{gsp:VB,whole:0,comma:1}* ,{arg:comma} and _{gsp:VB,whole:0}){and: 1}""", None, multi_arg_enumeration),
    (r"""(_{gsp:VB,whole:1} and _{gsp:VB,whole:1}*){and: 1}""", None),
    (r"""^(_{gsp:VB,whole:0}* or _{gsp:VB,whole:0}){or: 1}""", None, multi_arg_enumeration),
    (r"""^(_{gsp:VB,whole:0,comma:1}* ,{arg:comma} or _{gsp:VB,whole:0}){or: 1}""", None, multi_arg_enumeration),
    (r"""(_{gsp:VB,whole:1} or _{gsp:VB,whole:1}*){or: 1}""", None),
    (r"""^(_NN|NNS|NNP|NNPS{and:null,or:null}* ,{arg:comma} _NN|NNS|NNP|NNPS){comma: 1, many: 1}""", None),
    (r"""^(_NN|NNS|NNP|NNPS* and _NN|NNS|NNP|NNPS){and: 1, many: 1}""", None, multi_arg_enumeration),
    (r"""^(_NN|NNS|NNP|NNPS* or _NN|NNS|NNP|NNPS){or: 1}""", None, multi_arg_enumeration),
    (r"""^(_NN|NNS|NNP|NNPS{comma:1}* ,{arg:comma} and _NN|NNS|NNP|NNPS){and: 1, many: 1}""", None, multi_arg_enumeration),
    (r"""^(_NN|NNS|NNP|NNPS{comma:1}* ,{arg:comma} or _NN|NNS|NNP|NNPS){or: 1}""", None, multi_arg_enumeration),

    (r"""^(_NN|NNS{which:null}* which _{gsp:VB}){which:1}""", None),
    (r"""^(_NN|NNS{which:null}* _IN which _{gsp:VB}){which:1}""", None),
    (r"""^(_NNP|NNPS{which:null}* who _{gsp:VB,whole:0}){which:1}""", None),
    (r"""^(_NNP|NNPS{which:null}* _IN whom _{gsp:VB,whole:0}){which:1}""", None),
    (r"""^(_{subj:1,which:null}* which _{gsp:VB}){which:1}""", None),
    (r"""^(_{subj:1,which:null}* _IN which _{gsp:VB}){which:1}""", None),
    (r"""^(_NN|NNS{which:null}* , which _{gsp:VB} ,){which:1}""", None),
    (r"""^(_NN|NNS{which:null}* , which _{gsp:VB}){which:1}""", None),
    (r"""^(_NN|NNS{which:null}* , _IN which _{gsp:VB} ,){which:1}""", None),
    (r"""^(_NN|NNS{which:null}* , _IN which _{gsp:VB}){which:1}""", None),
    (r"""^(_NNP|NNPS{which:null}* , who _{gsp:VB,whole:0} ,){which:1}""", None),
    (r"""^(_NNP|NNPS{which:null}* , who _{gsp:VB,whole:0}){which:1}""", None),
    (r"""^(_NNP|NNPS{which:null}* , _IN whom _{gsp:VB,whole:0}){which:1}""", None),
    (r"""^(_{subj:1,which:null}* , which _{gsp:VB} ,){which:1}""", None),
    (r"""^(_{subj:1,which:null}* , which _{gsp:VB}){which:1}""", None),
    (r"""^(_{subj:1,which:null}* , _IN which _{gsp:VB} ,){which:1}""", None),
    (r"""^(_{subj:1,which:null}* , _IN which _{gsp:VB}){which:1}""", None),

    (r"""([_{whole:1}*] _EOL){at_end:1}""", None),
    (r"""(_{gsp:VB,whole:0}* how to _{gsp:VB}){dep:howto}""", None),
    (r"""(_{gsp:VB,whole:0}* how _{whole:1}){dep:how}""", None),
    (r"""(_{gsp:VB,whole:0}* , how _{whole:1}){dep:how}""", None),
    (r"""^(_{that:can_join}* _{whole:1}){that:none,dep:that}""", None),
    (r"""^(_{gsp:VB,whole:0}* that _{whole:1}){dep:that}""", None),
    (r"""^(_{gsp:NN}* that _{gsp:VB})""", None),
    (r"""^(_{gsp:VB,whole:0}* when|whenever _{whole:1}){dep:when}""", None),
    (r"""(_{gsp:VB,whole:1}* , so that _{whole:1}){dep:sothat}""", None),
    (r"""(_{gsp:VB,whole:1}* so that _{whole:1}){dep:sothat}""", None),
    (r"""^(_{gsp:NN}* , _SUCHTHAT _{at_end:1}){at_end:1,dep:suchthat}""", None),
    (r"""^(_{gsp:NN}* _SUCHTHAT _{at_end:1}){at_end:1,dep:suchthat}""", None),
    (r"""^(_{gsp:VB,whole:0}* whether _{at_end:1}){dep:whether}""", None),

    (r"""^(that _VBG|NN|NNS|NNP|NNPS{the:null}*){gsp:NN,subj:1,that_arg:1}""", None),

    (r"""(if|when _{whole:1}* then _{whole:1}){dep:if_then}""", None),
    (r"""(if|when _{whole:1}* , then _{whole:1}){dep:if_then}""", None),
    (r"""(when|whenever _{whole:1}* , _{whole:1}){dep:if_then}""", None),
    (r"""(when|whenever _{whole:1}* _{whole:1}){dep:if_then, warn: "probably ',' expected between parts"}""", None),
    (r"""(if _{whole:1}* _{whole:1}){dep:if_then,warn:"probably ',', 'or', 'and' or 'then' expected"}""", None),
    (r"""(if _{whole:1}* , _{whole:1}){dep:if_then,warn:"maybe, 'then' expected"}""", None),
    (r"""([_{gsp:VB,whole:1}* _IF _{whole:1}] !then){dep:if}""", None),
    (r"""([_{gsp:VB,whole:1}* , _IF _{whole:1}] !then){dep:if}""", None),

    (r"""(_{gsp:VB,whole:1}* though|before|after|as _{gsp:VB,whole:1,at_end:1}){dep:infix,at_end:1}""", None),
    (r"""([_{gsp:VB,whole:1}* as _EQN] _EOL){dep:as_limit,at_end:1}""", None),
    (r"""(though|before|after|as _{whole:1} _{whole:1}*){dep:prefix}""", None),
    (r"""(though|before|after|as _{whole:1} , _{whole:1}*){dep:prefix}""", None),

    (r"""(_{whole:1}* where _{whole:1}){dep:where}""", None),
    (r"""(_{whole:1}* , where _{whole:1}){dep:where}""", None),
    (r"""(_{whole:1}* since _{whole:1,at_end:1}){dep:since,at_end:1}""", None),
    (r"""(_{whole:1}* , since _{whole:1,at_end:1}){dep:since,at_end:1}""", None),
    (r"""(_SOL [since _{whole:1} _{whole:1}*]){dep:since}""", None),
    (r"""(_SOL [since _{whole:1} , _{whole:1}*]){dep:since}""", None),
    (r"""(_SOL [since _{whole:1} then _{whole:1}*]){dep:since}""", None),
    (r"""(_SOL [since _{whole:1} , then _{whole:1}*]){dep:since}""", None),
    (r"""(_{gsp:VB,whole:1}* _{due_to:1} _{gsp:VB,whole:1,at_end:1}){dep:dueto,at_end:1}""", None),
    (r"""(_{gsp:VB,whole:1}* , _{due_to:1} _{gsp:VB,whole:1,at_end:1}){dep:dueto,at_end:1}""", None),
    (r"""(_{gsp:VB,whole:1}* because _{gsp:VB,whole:1,at_end:1}){dep:because,at_end:1}""", None),
    (r"""(_{gsp:VB,whole:1}* , because _{gsp:VB,whole:1,at_end:1}){dep:because,at_end:1}""", None),
    (r"""(_{gsp:VB,whole:1}* in order to _{gsp:VB}){dep:goal}""", None),
    (r"""(_{gsp:VB,whole:1}* , in order to _{gsp:VB}){dep:goal}""", None),

    (r"""(is* why _{whole:1}){dep:reason}""", None),

    (r"""(for _{subj:1} _{gsp:VB,whole:1}*){for:1}""", None),
    (r"""(for _{subj:1} , _{gsp:VB,whole:1}*){for:1}""", None),
    (r"""(to _VB _{gsp:VB,whole:1}*){to:1}""", None),
    (r"""(to _VB , _{gsp:VB,whole:1}*){to:1}""", None),
    (r"""(_{whole:1} but* _{whole:1}){whole:1,dep:but}""", None),
    (r"""(_{whole:1} , but* _{whole:1}){whole:1,dep:but}""", None),
    (r"""(_{whole:1} , while* _{whole:1}){whole:1,dep:while}""", None),
    (r"""(_{whole:1} while* _{whole:1}){whole:1,dep:while}""", None),
    (r"""([_{whole:1}* , _VBG] _EOL){comment:1}""", None),
    (r"""(_VBG , _{whole:1}*){how:1}""", None),
    (r"""(_VBN{whole:0} _{whole:1}*){dep:cond}""", None),
    (r"""(_VBN{whole:0} , _{whole:1}*){dep:cond}""", None),
    (r"""([_{whole:1}* , _IE _{whole:1}] _EOL){ie:1}""", None),
    (r"""(_{whole:1}* , _IE _{whole:1} ,){ie:1}""", None),
    (r"""([_{whole:1}* _IE _{whole:1}] _EOL){ie:1,err:"',' before 'that is' or 'i.e.' expected"}""", None),

    (r"""(_{whole:1} , _{whole:1}*){comma:1}""", None),
    (r"""(_{whole:1}* : _{whole:1}){compound:colon}""", None),
    (r"""(_{whole:1}* ; _{whole:1}){compound:semicolon}""", None),
    (r"""(_{whole:1} , and _{whole:1}*){and:1}""", None),
    (r"""(_{whole:1} , or _{whole:1}*){or:1}""", None),
    (r"""(_{whole:1} and _{whole:1}*){and:1}""", None),
    (r"""(_{whole:1} or _{whole:1}*){or:1}""", None),
    (r"""(let*){whole:1,incomplete:0}""", None),

    (r"""^(_{gsp:NN}* _SUCHTHAT _{whole:1}){dep:suchthat}""", None),
    (r"""^(_{gsp:NN}* _SUCHTHAT _EQN){dep:suchthat}""", None),
    (r"""(_VBG _{gsp:VB,whole:1}*){dep:how}""", None),
    (r"""(_EQN*){whole:1}""", None),

    (r"""(_{vvod:1} _{whole:1}*)""", None),
    (r"""(_{vvod:1} , _{whole:1}*)""", None),
    (r"""(_IN _{subj:1} , _VB|VBP|VBZ|VBD{whole:1}*)""", None),

    (r"""(_CD){subj:1}""", None),
    (r"""^(_{gsp:VB,what:null}* that){what:1,that:1}""", None),
    (r"""(which*){ps:DT}""", None),
    (r"""(of){ps:IN}""", None),
    # (r"""(to){without_accept:1}""",None),

    (r"""(_{}* \( _{} \)){comment:1,warn:cannot determine comment type}""", None),
    (r"""(_{}* \( _{} _{} \)){comment:1,warn:comment not fully parsed}""", None),

    (r"""^(_{gsp:NN}* , namely|specifically|particularly _{gsp:NN})""", None),
    (r"""^(_{gsp:VB}* , namely|specifically|particularly _{gsp:VB})""", None),

    (r"""^(_EQN* _CD)""", None),
    (r"""^(_EQN _CD*)""", None),
    (r"""^(_CD _EQN*)""", None),
    (r"""^(_CD* _EQN)""", None),
    (r"""(: _EQN*){eqcolon:1}""", None),
    (r"""(_VBD{whole:0}){ps:VBN}""", None),
    (r"""(_VBN{whole:0}){ps:VBD}""", None),
    (r"""(_VB|VBP{whole:0}){whole:1}""", None, None, check_complete),
    #(r"""^(_JJRT _NN|NNS|NNP|NNPS*){the:1}""", None),

    (r"""#""", None),
    (r"""([_{gsp:VB,whole:1}* _CITE] _EOL){cite:1}""", None),
    (r"""(_SOL _{whole:1,incomplete:1}* _EOL){sentense:1,err:sentense is incomplete}""", None),
    (r"""(_SOL _{whole:1}* _EOL){sentense:1}""", None),
    (r"""(_SOL _{}* _EOL){sentense:1,err:sentense is incomplete}""", None),
]
