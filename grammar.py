grammar_str = r"""
(_SOL [note*] that){ps:VB}
(induces|implies|coincides|corresponds|varies|decreases|increases|belongs){ps:VBZ}
(commute|denote|recall|decrease|increase){ps:VB}
(explicit|quantum|interesting){ps:JJ}
(code|fiber|complex|soundness){ps:NN}
(codes){ps:NNS}
(random){ps:JJ}
(at random*){leaf:1}
(in turn){ps:RB}

(whether){ps:WHETHER}
(if|iff){ps:IF}
(if and only if){ps:IF}
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
(cf){ps:VB}
(almost all*){ps:DT}

(@ [cite|ref|eqref|cref]){ps:null}
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
(_VB|VBP|VBD|VBZ|VBN|VBG*){accept_to:1}
(be|been|is|are|am|was|were* _VBD|VBN){passive: 1} 
(is|are|am|was|were|can|could|will|would|have|has|had|did|does|do* not){neg:1, leaf:1} 
(of|in|on|over*){of_type: 1}
(_{gsp:NN}* _POS){nnpos: 1}
(he|she|it|they|we|I|you|this*){can_p:1}
(he|she|it|they|we|I|you [_VBN*]){ps:VBD}
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
(in order to*){leaf:1}

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
(the _NN|NNS|NNP|NNPS|EQN*){the:1}
(a|an _NN|NNP|EQN*){the:0}
(a|an _NNS|NNPS*){the:0, err:'a|an before plural'}
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

(_JJ|DT|ART [_VBG _{subj:1}*])

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

(^_{accept_to:1}* _TO _{subj:1}){with_to:1}
(^_{accept_to:1}* _TO us|me|him|her|it|them){with_to:1}
(_SOL [_{gsp:NN}* _TO _{subj:1}])
(_SOL [_{gsp:NN}* _TO us|me|him|her|it|them])
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
^_{gsp:NN}* _VBG|VBN _IN|OF|TO _{subj:1}
^_{gsp:NN}* , _VBG|VBN _IN|OF|TO _{subj:1}
^_{gsp:NN}* _VBG|VBN{what:1}
^_{gsp:NN}* _VBG|VBN{with_to:1}
^_VBG* _TO|IN|OF _{subj:1}
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
^(_NN|NNS|NNP|NNPS{and:null,or:null}* , _NN|NNS|NNP|NNPS){comma: 1}
^(_NN|NNS|NNP|NNPS* and _NN|NNS|NNP|NNPS){and: 1, many: 1}
^(_NN|NNS|NNP|NNPS* or _NN|NNS|NNP|NNPS){or: 1}
^(_NN|NNS|NNP|NNPS* , and _NN|NNS|NNP|NNPS){and: 1, many: 1}
^(_NN|NNS|NNP|NNPS* , or _NN|NNS|NNP|NNPS){or: 1}


^(_NN|NNS{which:null}* which _{gsp:VB}){which:1}
^(_NN|NNS{which:null}* _IN which _{gsp:VB}){which:1}
^(_NNP|NNPS{which:null}* who _{gsp:VB,whole:0}){which:1}
^(_NNP|NNPS{which:null}* _IN whom _{gsp:VB,whole:0}){which:1}
^(_{subj:1,which:null}* which _{gsp:VB}){which:1}
^(_{subj:1,which:null}* _IN which _{gsp:VB}){which:1}
^(_NN|NNS{which:null}* , which _{gsp:VB} ,){which:1}
^(_NN|NNS{which:null}* , which _{gsp:VB}){which:1}
^(_NN|NNS{which:null}* , _IN which _{gsp:VB} ,){which:1}
^(_NN|NNS{which:null}* , _IN which _{gsp:VB}){which:1}
^(_NNP|NNPS{which:null}* , who _{gsp:VB,whole:0} ,){which:1}
^(_NNP|NNPS{which:null}* , who _{gsp:VB,whole:0}){which:1}
^(_NNP|NNPS{which:null}* , _IN whom _{gsp:VB,whole:0}){which:1}
^(_{subj:1,which:null}* , which _{gsp:VB} ,){which:1}
^(_{subj:1,which:null}* , which _{gsp:VB}){which:1}
^(_{subj:1,which:null}* , _IN which _{gsp:VB} ,){which:1}
^(_{subj:1,which:null}* , _IN which _{gsp:VB}){which:1}

([_{whole:1}*] _EOL){at_end:1}
(_{gsp:VB,whole:0}* how to _{gsp:VB}){dep:howto}
(_{gsp:VB,whole:0}* how _{whole:1}){dep:how}
(_{gsp:VB,whole:0}* , how _{whole:1}){dep:how}
^(_{that:can_join}* _{whole:1}){that:none,dep:that}
^(_{gsp:VB,whole:0}* that _{whole:1}){dep:that}
^(_{gsp:NN}* that _{gsp:VB})
^(_{gsp:VB,whole:0}* when|whenever _{whole:1}){dep:when}
(_{gsp:VB,whole:1}* , so that _{whole:1}){dep:sothat}
(_{gsp:VB,whole:1}* so that _{whole:1}){dep:sothat}
^(_{gsp:NN}* , _SUCHTHAT _{at_end:1}){at_end:1,dep:suchthat}
^(_{gsp:NN}* _SUCHTHAT _{at_end:1}){at_end:1,dep:suchthat}
^(_{gsp:VB,whole:0}* whether _{at_end:1}){dep:whether}

^(that _VBG|NN|NNS|NNP|NNPS{the:null}*){gsp:NN,subj:1,that_arg:1}

(if|when _{whole:1}* then _{whole:1}){dep:if_then}
(if|when _{whole:1}* , then _{whole:1}){dep:if_then}
(when|whenever _{whole:1}* , _{whole:1}){dep:if_then}
(when|whenever _{whole:1}* _{whole:1}){dep:if_then, warn: "',' expected between parts"}
(if _{whole:1}* _{whole:1}){dep:if_then,err:"',', 'or', 'and' or 'then' expected"}
(if _{whole:1}* , _{whole:1}){dep:if_then,warn:"maybe, 'then' expected"}
([_{gsp:VB,whole:1}* _IF _{whole:1}] !then){dep:if}
([_{gsp:VB,whole:1}* , _IF _{whole:1}] !then){dep:if}

(_{gsp:VB,whole:1}* though|before|after|as _{gsp:VB,whole:1,at_end:1}){dep:infix,at_end:1}
([_{gsp:VB,whole:1}* as _EQN] _EOL){dep:as_limit,at_end:1}
(though|before|after|as _{whole:1} _{whole:1}*){dep:prefix}
(though|before|after|as _{whole:1} , _{whole:1}*){dep:prefix}

(_{whole:1}* where _{whole:1}){dep:where}
(_{whole:1}* , where _{whole:1}){dep:where}
(_{whole:1}* since _{whole:1,at_end:1}){dep:since,at_end:1}
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
%(to){without_accept:1}

(_{}* \( _{} \)){comment:1,warn:cannot determine comment type}
(_{}* \( _{} _{} \)){comment:1,warn:comment not fully parsed}

^(_EQN* _CD)
^(_EQN _CD*)
^(_CD _EQN*)
^(_CD* _EQN)
(: _EQN*){eqcolon:1}
(_VBD{whole:0}){ps:VBN}
(_VBN{whole:0}){ps:VBD}
(_VB|VBP{whole:0}){whole:1,incomplete:1}
#
([_{gsp:VB,whole:1}* _CITE] _EOL){cite:1}
(_SOL _{whole:1,incomplete:1}* _EOL){sentense:1,err:sentense is incomplete}
(_SOL _{whole:1}* _EOL){sentense:1}
(_SOL _{}* _EOL){sentense:1,err:sentense is incomplete}
"""
