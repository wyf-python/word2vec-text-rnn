import re


def detokenize_number(text):
    chars, chars_new, index, symbol = list(text), [], 0, ['.', ',', '，', '/']

    while index < len(chars):
        chars_new.append(chars[index])

        if index + 5 < len(chars):
            if chars[index].isdigit() and chars[index + 1] == ' ':  # 如果当前字符是数字且下一个字符是空格
                if chars[index + 2].isdigit() or chars[index + 2] == '%' or \
                        (chars[index + 2] in symbol and chars[index + 3].isdigit()):
                    # 2 ,00  2 ，00  2 /34   2 .34  2 %
                    chars_new.append(chars[index + 2])  # 把数字与特殊符号之间的空格去掉
                    index += 3
                    continue
                elif chars[index + 2] in symbol and chars[index + 3] == ' ' and chars[index + 4].isdigit():
                    # 特殊符号两边都有空格    2 . 2
                    chars_new.extend([chars[index + 2], chars[index + 4]])
                    index += 5
                    continue
            elif chars[index].isdigit() and chars[index + 1] in symbol and chars[index + 2] == ' ':
                # 2. 00   2, 00 2， 00   2/ 3
                if chars[index + 3].isdigit():
                    chars_new.extend([chars[index + 1], chars[index + 3]])
                    index += 4
                    continue
        index += 1

    for index, char in enumerate(chars_new):  # 把夹杂在数字与特殊符号中的多个空格(>=2)去掉
        if index + 3 < len(chars_new):
            if char == ' ' and chars_new[index - 1].isdigit():
                if chars_new[index + 1].isdigit() or chars_new[index + 1] == '%':
                    del chars_new[index]
                elif chars_new[index + 1] in symbol and chars_new[index + 2].isdigit():
                    del chars_new[index]
                elif chars_new[index + 1] in symbol and chars_new[index + 2] == ' ' and chars_new[index + 3].isdigit():
                    del chars_new[index]
                    del chars_new[index + 1]

    return ''.join(chars_new)


def identify_number(token):
    # 识别单个token是否是数字
    if token.isdigit():  # 20  整数
        return True
    elif '%' in token[-1] and token.replace('%', '').isdigit():  # 20%
        return True
    elif '%' in token[-1] and '.' in token[1: -2] and re.sub(r'[\.%]', '', token).isdigit():  # 20.34%
        return True
    else:
        for symbol in ['.', ',', '，', '/']:  # 有小数点的数、英文数字形式、分数
            if symbol in token[1: -1] and token.replace(symbol, '').isdigit():
                return True
    return False


def match_number(text, type='NUM'):
    inputs1, inputs2 = text.split('\t')
    inputs1, inputs2 = inputs1.split(), inputs2.split()
    ne_digit_order_dict1, ne_digit_order_dict2 = {}, {}

    for index1, input1 in enumerate(inputs1):
        if identify_number(input1):
            ne_digit_order_dict1[type] = ne_digit_order_dict1.get(type, -1) + 1
            inputs1[index1] = '{}|<{}>#{}|{}'.format(input1, type, ne_digit_order_dict1[type], input1)
    for index2, input2 in enumerate(inputs2):
        if identify_number(input2):
            ne_digit_order_dict2[type] = ne_digit_order_dict2.get(type, -1) + 1
            inputs2[index2] = '<{}>#{}'.format(type, ne_digit_order_dict2[type])

    return '{}\t{}'.format(' '.join(inputs1), ' '.join(inputs2))


text = '我 在 2 0 00 年 喝 了 2 0 .3 4 % 升 水 ，走 了 3 500 公里 。\t我 在 2 0 00 年 喝 了 2 0 .3 4 % 升 水 ，走 了 3 500 公里 。'
print(text)
text = detokenize_number(text)
print(text)
text = match_number(text)
print(text)
