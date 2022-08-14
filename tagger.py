import gzip
import json
import pickle

import yaml
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) | {'ref', 'cref', 'eqref', 'cite', 'hence',
                                                'fix', 'put', 'let', 'consider', 'denote', 'define', 'introduce',
                                                'find', 'imagine', 'note', 'observe', 'suppose'}


modal_verbs = {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would'}


pos_to_mobipos = {
    'NN': 'N',
    'NNP': 'N',
    'NNS': 'p',
    'NNPS': 'p',
    'JJ': 'A',
    'JJR': 'A',
    'JJS': 'A',
    'RB': 'v',
    'RBR': 'v',
    'VB': 'Vti',
    'DT': 'D',
    'CD': 'M',
}


class Dictionaries:
    def __init__(self):
        self.prep_types = {}
        self.dict_pos_set = set()
        self.pos_dict = {}
        self.vbp_set = set()
        self.vbd_set = set()
        self.vbn_set = set()
        self.vb_irreg_set = set()
        # vb_irreg_set_ext = {}
        self.mobypos_dict = {}
        self.mobypos_comb_dict = {}
        self.pos_exceptions = {}

    def to_json(self):
        fields = {
            'dict_pos_set': list(self.dict_pos_set),
            'pos_dict': self.pos_dict,
            'vbp_set': list(self.vbp_set),
            'vbd_set': list(self.vbd_set),
            'vbn_set': list(self.vbn_set),
            'vb_irreg_set': list(self.vb_irreg_set),
            'mobypos_dict': self.mobypos_dict,
            'mobypos_comb_dict': self.mobypos_comb_dict,
            'pos_exceptions': self.pos_exceptions,
            'prep_types': self.prep_types,
        }
        return json.dumps(fields)

    def from_json(self, json_string):
        fields = json.loads(json_string)
        self.dict_pos_set = set(fields['dict_pos_set'])
        self.pos_dict = fields['pos_dict']
        self.vbp_set = set(fields['vbp_set'])
        self.vbd_set = set(fields['vbd_set'])
        self.vbn_set = set(fields['vbn_set'])
        self.vb_irreg_set = set(fields['vb_irreg_set'])
        self.mobypos_dict = fields['mobypos_dict']
        self.mobypos_comb_dict = fields['mobypos_comb_dict']
        self.pos_exceptions = fields['pos_exceptions']
        self.prep_types = fields['prep_types']

    def save_json(self, filename):
        with open(filename, 'w') as f:
            f.write(self.to_json())


    def load_json(self, filename):
        with open(filename, 'r') as f:
            self.from_json(f.read())

    # json+gzip save/load functions (can be used for safety instead of pickle)
    def save_json_compressed(self, filename):
        with gzip.open(filename, 'wb') as f:
            f.write(self.to_json().encode('utf-8'))

    def load_json_compressed(self, filename):
        with gzip.open(filename, 'rb') as f:
            self.from_json(f.read().decode('utf-8'))

    def load_from_src(self, dict_dir):
        with open(f'{dict_dir}/pos_other_exceptions.yml') as f:
            data = yaml.load(f, Loader=yaml.CSafeLoader)
            self.pos_exceptions = {x: v for k, v in data.items() for x in k.split(',')}

        with open(f'{dict_dir}/mobypos.pickle', 'rb') as f:
            self.mobypos_dict, self.mobypos_comb_dict = pickle.load(f)

        with open(f'{dict_dir}/irregular_verbs.txt') as f:
            for line in f:
                vbp, vbd, vbn, *pref = line.split()
                self.vbp_set.update(vbp.split('/'))
                self.vbd_set.update(vbd.split('/'))
                self.vbn_set.update(vbn.split('/'))
                if pref:
                    for p in pref[0].split(','):
                        self.vbp_set.update([p+x for x in vbp.split('/')])
                        self.vbd_set.update([p+x for x in vbd.split('/')])
                        self.vbn_set.update([p+x for x in vbn.split('/')])
            self.vb_irreg_set = self.vbp_set|self.vbn_set|self.vbd_set

        with open(f'{dict_dir}/dict_2-4gram_10000-f.yml', 'r') as pos_dict_file:
            pos_dict = yaml.load(pos_dict_file, Loader=yaml.CSafeLoader)
            for key in pos_dict:
                self.pos_dict[key] = {k: v for k, v in pos_dict[key].items() if v > 5}

        self.dict_pos_set = set(sum((list(v.keys()) for v in self.pos_dict.values()), []))

        with open(f'{dict_dir}/prepositions.txt', 'r') as f:
            prep_types = yaml.load(f, Loader=yaml.CSafeLoader)
            self.prep_types = {prep: v for k, v in prep_types.items() for prep in k.split(',')}

    def can_be_pos_start(self, word, pos):
        if pos in pos_to_mobipos and word in self.mobypos_dict:
            return bool(set(pos_to_mobipos[pos]) & set(self.mobypos_dict[word]))
        return any(key.startswith(pos) for key in self.pos_dict.get(word, {}))

    def can_be_pos(self, word, pos):
        if pos in pos_to_mobipos and word in self.mobypos_dict:
            return bool(set(pos_to_mobipos[pos]) & set(self.mobypos_dict[word]))
        return pos in self.pos_dict.get(word, {})

    # detect possible part of speech of a word by its suffix
    def detect_pos_recommendation(self, word: str):
        if '-' in word:
            end = word.split('-')[-1]
            if end:
                return self.detect_pos_recommendation(end)
        if word in self.pos_exceptions:
            return [self.pos_exceptions[word]], {}
        pos = []
        neg = {}
        if len(word) < 3:
            return pos, neg

        if word.endswith('ing'):
            if self.can_be_pos_start(word[:-3], 'VB') or self.can_be_pos_start(word[:-3]+'e', 'VB') or \
               len(word) > 5 and word[-5] == word[-4] and word[-4] in 'bcdfghjklmnpqrstvwxz' and self.can_be_pos_start(word[:-4], 'VB'):
                #any(key.startswith('VB') for key in pos_dict.get(word[:-3], {})) or \
                    #any(key.startswith('VB') for key in pos_dict.get(word[:-3]+'e', {})):
                pos += ['VBG']
        else:
            neg['VBG'] = 'VBD' if word.endswith('ed') else 'VBN' if word.endswith('en') else 'VBZ' if word.endswith('es') else 'VB'

        if word.endswith('ly'):
            if self.can_be_pos(word[:-2], 'JJ'): #any(key.startswith('JJ') for key in pos_dict.get(word[:-2], {})):
                pos += ['RB']
            neg['JJ'] = 'RB'

        if word in modal_verbs:
            return ['MD'], neg

        if word in self.vb_irreg_set:
            w = word  # if word in vb_irreg_set else vb_irreg_set_ext[word]
            if w in self.vbp_set:
                pos += ['VB', 'VBP']
            if w in self.vbd_set:
                pos += ['VBD']
            if w in self.vbn_set:
                pos += ['VBN']
            for p in ['VB', 'VBP', 'VBD', 'VBN']:
                if p not in pos:
                    neg[p] = pos[0]
        else:
            if word.endswith('ed'):
                if self.can_be_pos_start(word[:-2], 'VB') or self.can_be_pos_start(word[:-2]+'e', 'VB') or \
                   word[-3:] == 'ied' and self.can_be_pos_start(word[:-3]+'y', 'VB') or \
                   len(word) > 4 and word[-4] == word[-3] and word[-3] in 'bcdfghjklmnpqrstvwxz' and self.can_be_pos_start(word[:-3], 'VB'):
                    pos += ['VBD', 'VBN', 'JJ']
            else:
                neg['VBD'] = 'VB'
                neg['VBN'] = 'VB'

            # if word.endswith('en'):
            #     if any(key.startswith('VB') for key in pos_dict.get(word[:-2], {})) or \
            #             any(key.startswith('VB') for key in pos_dict.get(word[:-2]+'e', {})):
            #         pos = ['VBN', 'JJ']
        if word.endswith('er'):
            if self.can_be_pos(word[:-2], 'JJ') or self.can_be_pos(word[:-2]+'e', 'JJ'):
                pos += ['JJR']
            if self.can_be_pos(word[:-2], 'RB') or self.can_be_pos(word[:-2]+'y', 'RB'):
                pos += ['RBR']
            if self.can_be_pos(word[:-2], 'VB') or self.can_be_pos(word[:-2]+'e', 'VB'):
                pos += ['NN']
        else:
            if word not in ('worse', 'less', 'more'):
                neg['JJR'] = 'JJ'
                neg['RBR'] = 'RB'

        if word.endswith('est'):
            if self.can_be_pos(word[:-3], 'JJ') or self.can_be_pos(word[:-3]+'e', 'JJ'):
                pos += ['JJS']
        elif not word.endswith('st'):
            neg['JJS'] = 'JJ'

        if len(word) > 2 and word.endswith('s'):
            if self.can_be_pos(word[:-1], 'NN') or \
                    word[-3:] == 'ies' and self.can_be_pos_start(word[:-3]+'y', 'NN') or\
                    word[-3:] in ('ses', 'xes', 'zes') and self.can_be_pos_start(word[:-2]+'s', 'NN') or\
                    word[-4:] == 'ices' and self.can_be_pos_start(word[:-4]+'ex', 'NN'):
                if any(letter.isupper() for letter in word):
                    pos.append('NNPS')
                else:
                    pos.append('NNS')
            if self.can_be_pos_start(word[:-1], 'VB') or \
                    word[-3:] == 'ies' and self.can_be_pos_start(word[:-3] + 'y', 'VB') or \
                    word[-3:] in ('ses', 'xes', 'zes', 'oes') and self.can_be_pos_start(word[:-2], 'VB'):
                pos.append('VBZ')
        else:
            neg['VBZ'] = 'VB'
            neg['NNPS'] = 'NNP'
            if word[-1] == 'a':
                if self.can_be_pos(word[:-1], 'NN'):
                    pos += ['NNS']
            elif 'p' not in self.mobypos_dict.get(word, ''):
                neg['NNS'] = 'NN'

        if word.endswith('ness'):
            pos += ['NN']

        #if word.endswith('ate'):
        #    pos = ['VB', 'VBP', 'NN', 'JJ']

        if word.endswith('ful'):
            if self.can_be_pos(word[:-3], 'NN') or self.can_be_pos(word[:-3]+'e', 'NN'):
                pos += ['JJ']

        sym_map = {'N': 'NN', 'p': 'NNS', 't': 'VB', 'i': 'VB', 'V': 'VB', 'A': 'JJ', 'v': 'RB','D': 'DT', 'M': 'CD'}
        if word in self.mobypos_dict:
            for sym in self.mobypos_dict[word]:
                if sym in sym_map:
                    r = sym_map[sym]
                    r = neg.get(r, r)
                    if r not in pos:
                        pos.append(r)

        return pos, neg

    special_tokens = {'ie': 'IE', 'eg': 'EG', 'et_al': 'ET_AL', 'cf': 'CF'}

    def get_pos(self, word, tag):
        if not word:
            return word, tag
        if word[0].isupper() and not tag.startswith('NNP'):
            if word.lower() in stop_words:
                word = word.lower()
            else:
                tag = 'NNP'
        elif word in self.special_tokens:
            tag = self.special_tokens[word]
        else:
            pos, neg = self.detect_pos_recommendation(word)
            if tag in neg:
                tag = neg[tag]
            if tag in self.dict_pos_set and ((word in self.pos_dict and self.pos_dict[word].get(tag, 0) < 5) or (pos and tag not in pos)):
                items = [(neg.get(k, k), v) for k, v in self.pos_dict.get(word, {}).items()]
                if pos and set(pos)&set(items):
                    tag = max(((k,v) for k,v in items if k in pos), key=lambda x: x[1])[0]
                elif pos:
                    tag = pos[0]
                elif items:
                    tag = max(items, key=lambda x: x[1])[0]
        return word, tag


_dictionaries = None


def dictionaries():
    global _dictionaries
    if _dictionaries is None:
        _dictionaries = Dictionaries()
        _dictionaries.load_json_compressed('dicts/dicts.gz')
    return _dictionaries
