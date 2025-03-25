import re

curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

""" from https://github.com/keithito/tacotron """
def text_to_sequence(text, symbols):
    sequence = []
    while len(text):
        m = curly_re.match(text)
        if not m:
            sequence += list(text)
            break
        sequence += list(m.group(1))
        sequence += [f"@{s}" for s in m.group(2).split()]
        text = m.group(3)
    return [symbols.index(s) for s in sequence + [";"]]