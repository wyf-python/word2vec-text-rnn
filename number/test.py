def detoken(text):
    chars = list(text)
    symbol = ['.', ',', '，', '/']
    chars_new = ''
    for index, char in enumerate(chars):
        chars_new += char
        if index + 2 < len(chars) :
            if char.isdigit() and chars[index + 1] is ' ':
                if char in symbol + ['%'] and chars[index + 2].isdigit():
                    chars_new += chars[index + 2]
            elif chars_new[-1].isdigit() and char in symbol and chars[index + 1] is ' ': 
                if chars[index + 2].isdigit():
                    chars_new += chars[index + 2]