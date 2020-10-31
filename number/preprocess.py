#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import re
import regex
import jieba
from traceback import format_exc
from loguru import logger

class Preprocess:
    def __init__(self, ne_file):
        self.control_char_re = regex.compile('\p{C}')
        #self.chinese_blank_re = re.compile('([^u4E00-u9FA5]+) ')
        #self.blank_chinese_re = re.compile(' ([^u4E00-u9FA5]+)')
        self.ne_term_re = re.compile('\|<[A-Z]+>\|')
        self.ne_term_order_re = re.compile('\|(<[A-Z]+>)#\d+\|')

        jieba.initialize()

        self.NE_dict_ch = {}
        self.NE_dict_en = {}
        self.NE_types = {
                'PER':'PER',
                'LOC':'LOC',
                'ORG':'ORG',
                'TIT':'PER',
                'PERSON':'PER',
                'LOCATION':'LOC',
                'ORGANIZATION':'ORG',
                'TITLE':'PER'
                }
        self.load_NE_dict(ne_file)
        #self.save_NE_dict(ne_file + '.save')
        self.max_word_cnt = 6

    def load_NE_dict(self, filename):
        self.NE_dict_ch = {}
        self.NE_dict_en = {}
        with open(filename) as fp:
            for line in fp:
                items = line.strip().split('\t')
                if len(items) != 3:
                    continue
                (src, type, tgt) = items
                if type not in self.NE_types:
                    continue
                type = self.NE_types[type]
                src = src.replace(' ', '_')
                tgt = tgt.replace(' ', '_')
                self.NE_dict_ch[src] = (type, tgt)
                self.NE_dict_en[tgt] = (type, src)
        #print(len(self.NE_dict_ch), len(self.NE_dict_en))

    def parse_user_nedict(self, ne_list):
        ne_dict = {}
        for ne in ne_list:
            try:
                src = ne['zh']
                tgt = ne['en']
                type = ne['category']
                if type not in self.NE_types:
                    logger.debug('skip unimplemented category term: {}'.format(ne))
                    continue
                type = self.NE_types[type]
                src = src.replace(' ', '_')
                tgt = tgt.replace(' ', '_')
                ne_dict[src] = (type, tgt)
            except Exception as err:
                logger.warning('{}:{}\n{}'.format(err.__class__.__name__, err, format_exc()))
        return ne_dict

    def save_NE_dict(self, filename):
        with open(filename, 'w') as fp:
            for src in self.NE_dict_ch:
                (type, tgt) = self.NE_dict_ch[src]
                line = '{}\t{}\t{}\n'.format(src, type, tgt)
                fp.write(line)

    def clean_text(self, text):
        text = self.control_char_re.sub(' ', text)

        text = ' '.join(text.split())
        text = text.strip()
        return text

    def normalize_digit_letter(self, text):
        chars = list(text)
        for index, char in enumerate(chars):
            code = ord(char)
            if 65296 <= code and code <= 65305: # ０-９
                chars[index] = chr(code - (65296 - 48))
            elif (65313 <= code and code <= 65338) or (65345 <= code and code <= 65370): # Ａ-Ｚａ-ｚ
                chars[index] = chr(code - (65313 - 65))
        return ''.join(chars)

    def normalize_punctuation(self, text, lang='ch'):
        if lang == 'ch':
            text = re.sub('([^\.])\.$', r'\1。', text)
            text = re.sub('([^\d]),', r'\1，', text)
            # text = re.sub(r'\d+\s*，\s*\d+', '', text)
            text = text.replace('?', '？')
            text = text.replace('!', '！')
            text = text.replace('％', '%')
        text = text.replace('|', ' ')
        text = text.replace('_', ' ')
        text = text.replace(' ', ' ')
        return text

    def is_chinese_char(self, char):
        if char >= '\u4e00' and char <= '\u9fa5':
            return True
        return False

    def detokenize_ch(self, text):
        #while self.chinese_blank_re.search(text):
        #    text = self.chinese_blank_re.sub(r'\1', text)
        #while self.blank_chinese_re.search(text):
        #    text = self.blank_chinese_re.sub(r'\1', text)
        #return text
        tokens = text.split()
        tokens_new = []
        for index, token in enumerate(tokens):
            if len(tokens_new) > 0 and \
                (self.is_chinese_char(tokens[index-1][-1]) or \
                self.is_chinese_char(token[0])):
                tokens_new[-1] += token
            else:
                tokens_new.append(token)
        return ' '.join(tokens_new)

    def tokenize_ch(self, text):
        text = ' '.join(jieba.cut(text, HMM=True))
        text = ' '.join(text.split())
        text = text.strip()
        return text

    def str_to_lower(self, text):
        chars = list(text)
        for index, char in enumerate(chars):
            code = ord(char)
            if 65 <= code and code <= 90:
                chars[index] = chr(code + 32)
        return ''.join(chars)

    def search_NE_dict(self, text, base_nedict, user_nedict):
        if text in user_nedict:
            (type, tgt) = user_nedict[text]
            return text, type, tgt
        if text in base_nedict:
            (type, tgt) = base_nedict[text]
            return text, type, tgt
        text = self.str_to_lower(text)
        if text in user_nedict:
            (type, tgt) = user_nedict[text]
            return text, type, tgt
        if text in base_nedict:
            (type, tgt) = base_nedict[text]
            return text, type, tgt
        return None, None, None

    def match_NE_dict(self, text, user_nelist=[], lang='ch'):
        separator = '' if lang == 'ch' else '_'
        base_nedict = self.NE_dict_ch if lang == 'ch' else self.NE_dict_en
        user_nedict = self.parse_user_nedict(user_nelist)
        words = text.split()
        words_ne = []
        words_tag = []
        ne_order_dict = {}
        beg = 0
        while beg < len(words):
            max_end_index = min(len(words), beg + self.max_word_cnt)
            matched = False
            for end in range(max_end_index, beg, -1):
                cand = separator.join(words[beg:end])
                (src, type, tgt) = self.search_NE_dict(cand, base_nedict, user_nedict)
                if src != None:
                    ne_order_dict[type] = ne_order_dict.get(type, -1) + 1
                    ne_tag = '<{}>#{}'.format(type, ne_order_dict[type])
                    ne = '{}|{}|{}'.format(self.str_to_lower(cand), ne_tag, tgt)
                    words_ne.append(ne)
                    words_tag.append(ne_tag)
                    matched = True
                    beg = end
                    break
            if not matched:
                words_ne.append(self.str_to_lower(words[beg]))
                words_tag.append(self.str_to_lower(words[beg]))
                beg += 1
        return '{}\t{}'.format(' '.join(words_ne), ' '.join(words_tag))

    def identify_number(self, token):
        # 识别单个token是否是数字
        if token.isdigit():  # 20  整数
            return True
        elif '%' in token[-1] and token.replace('%', '').isdigit():   # 20%
            return True
        elif '%' in token[-1] and '.' in token[1: -2] and re.sub(r'[\.%]', '', token).isdigit():  # 20.34%
            return True
        else:
            for symbol in ['.', ',', '，', '/']:  # 有小数点的数、英文数字形式、分数
                if symbol in token[1: -1] and token.replace(symbol, '').isdigit():
                    return True
        return False

    def detokenize_num(self, text):
        chars = list(text)
        symbol = ['.', ',', '，', '/']
        for index, char in enumerate(chars):
            if (index + 2 < len(chars)) and char in symbol and chars[index - 1].isdigit() and chars[index + 1] is ' ' and chars[index + 2].isdigit():  # 2. 24
                chars[index - 1] = chars[index - 1] + char + chars[index + 2]
                del chars[index]
                del chars[index]
                del chars[index]
            elif (index + 1 < len(chars)) and char in symbol and chars[index - 1] is ' ' and chars[index - 2].isdigit() and chars[index + 1].isdigit():  # 2 .34
                chars[index - 2] = chars[index - 2] + char + chars[index + 1]
                del chars[index - 1]
                del chars[index - 1]
                del chars[index - 1]
            elif (index + 2 < len(chars)) and char in symbol and chars[index - 1] is ' ' and chars[index - 2].isdigit() and chars[index + 1] is ' ' and chars[index + 2].isdigit():
                # 2 . 34
                chars[index - 2] = chars[index - 2] + char + chars[index + 2]
                del chars[index - 1]
                del chars[index - 1]
                del chars[index - 1]
                del chars[index - 1]
            elif (index + 1 < len(chars)) and char is ' ' and chars[index - 1].isdigit() and chars[index + 1].isdigit():  # 10 000
                chars[index - 1] = chars[index - 1] + chars[index + 1]
                del chars[index]
                del chars[index]

        return ''.join(chars)

    def detokenize_percent(self, text):
        chars = list(text)
        for index, char in enumerate(chars):
            if (index + 2 < len(chars)) and char.isdigit() and chars[index + 1] is ' ' and chars[index + 2] is '%':
                chars[index] = char + chars[index + 2]
                del chars[index + 1]
                del chars[index + 1]

        return ''.join(chars)

    def match_number(self, text):
        inputs1, inputs2 = text.split('\t')
        inputs1, inputs2 = inputs1.split(), inputs2.split()

        type = 'NUM'
        ne_digit_order_dict1 = {}
        ne_digit_order_dict2 = {}

        for index1, input1 in enumerate(inputs1):
            if self.identify_number(input1):
                ne_digit_order_dict1[type] = ne_digit_order_dict1.get(type, -1) + 1
                inputs1[index1] = '{}|<{}>#{}|{}'.format(input1, type, ne_digit_order_dict1[type], input1)

        for index2, input2 in enumerate(inputs2):
            if self.identify_number(input2):
                ne_digit_order_dict2[type] = ne_digit_order_dict2.get(type, -1) + 1
                inputs2[index2] = '<{}>#{}'.format(type, ne_digit_order_dict2[type])

        return '{}\t{}'.format(' '.join(inputs1), ' '.join(inputs2))

    def remove_NE_order(self, words):
        for index, word in enumerate(words):
            search_res = self.ne_term_order_re.search(word)
            if search_res:
                type = search_res.group(1)
                (src, temp, tgt) = word.split('|')
                words[index] = '{}|{}|{}'.format(src, type, tgt)

    def collect_NE(self, words):
        ne_dict = {}
        for word in words:
            if self.ne_term_re.search(word):
                word = self.str_to_lower(word)
                ne_dict[word] = ne_dict.get(word, 0) + 1
        return ne_dict

    def align_NE(self, text):
        items = text.strip().split('\t')
        if len(items) != 2:
            return ''
        src_words = items[0].split()
        tgt_words = items[1].split()
        self.remove_NE_order(src_words)
        self.remove_NE_order(tgt_words)
        tgt_ne_dict = self.collect_NE(tgt_words)

        self.ne_order_dict = {}
        self.tgt_ne_dict_new = {}
        for index, word in enumerate(src_words):
            if self.ne_term_re.search(word):
                (src, type, tgt) = word.split('|')
                ne_tgt = '{}|{}|{}'.format(tgt, type, src)
                ne_tgt = self.str_to_lower(ne_tgt)
                if ne_tgt in tgt_ne_dict and tgt_ne_dict[ne_tgt] > 0:
                    tgt_ne_dict[ne_tgt] -= 1
                    self.ne_order_dict[type] = self.ne_order_dict.get(type, -1) + 1
                    #src_words[index] = '{}|{}#{}|{}'.format(src, type, ne_order_dict[type], tgt)
                    src_words[index] = '{}#{}'.format(type, self.ne_order_dict[type])
                    self.tgt_ne_dict_new.setdefault(ne_tgt, []).append(self.ne_order_dict[type])
                else:
                    src_words[index] = src.replace('_', ' ')
        for index, word in enumerate(tgt_words):
            if self.ne_term_re.search(word):
                (src, type, tgt) = word.split('|')
                word = self.str_to_lower(word)
                if word in self.tgt_ne_dict_new and len(self.tgt_ne_dict_new[word]) > 0:
                    #tgt_words[index] = '{}|{}#{}|{}'.format(src, type, tgt_ne_dict_new[word][0], tgt)
                    tgt_words[index] = '{}#{}'.format(type, self.tgt_ne_dict_new[word][0])
                    del self.tgt_ne_dict_new[word][0]
                else:
                    tgt_words[index] = src.replace('_', ' ')
        text = '{}\t{}'.format(' '.join(src_words), ' '.join(tgt_words))
        return text

    def do_preprocess_ch(self, text, ne_list=[]):
        text = self.clean_text(text)
        text = self.normalize_digit_letter(text)
        text = self.normalize_punctuation(text)
        text = self.detokenize_ch(text)
        text = self.tokenize_ch(text)
        #text = self.str_to_lower(text)
        text = self.match_NE_dict(text, user_nelist=ne_list)
        text = self.detokenize_num(text)
        text = self.detokenize_percent(text)
        text = self.match_number(text)
        return text

    def do_preprocess_en(self, text):
        #text = self.str_to_lower(text)
        text = self.match_NE_dict(text, lang='en')
        text = self.detokenize_num(text)
        text = self.detokenize_percent(text)
        text = self.match_number(text)
        return text

    def do_preprocess_ce(self, text):
        text = self.align_NE(text)
        return text


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('python preprocess.py <ne_file> <language(ch|en|ce)>')
        sys.exit(1)

    prep = Preprocess(sys.argv[1])
    ne_list = [
        {
            'zh': '莫凡',
            'en': 'Mo Fan',
            'category': 'PER'
        },
        {
            'zh': '莫家兴',
            'en': 'Mo Jiaxing',
            'category': 'PERSON'
        },
        {
            'zh': '穆贺',
            'en': 'Mu He',
            'category': 'PERSON'
        },
        {
            'zh': '穆宁',
            'en': 'Mu Ning',
            'category': 'PERSON'
        },
        {
            'zh': '天澜高中',
            'en': 'TianLan high school',
            'category': 'ORG'
        },
        {
            'zh': '心夏',
            'en': 'Xin Xia',
            'category': 'PERSON'
        }]
    for line in sys.stdin:
        line = line.strip('\n')
        if sys.argv[2] == 'ch':
            nline = prep.do_preprocess_ch(line)
        elif sys.argv[2] == 'en':
            nline = prep.do_preprocess_en(line)
        elif sys.argv[2] == 'ce':
            nline = prep.do_preprocess_ce(line)
        else:
            print('ERROR: unsupported language.')
            sys.exit(1)
        print(nline)
