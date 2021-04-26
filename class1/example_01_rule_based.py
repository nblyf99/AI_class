import random

rules = """
复合句子 = 句子 , 连词 复合句子 | 句子
连词 = 而且 | 但是 | 不过
句子 = 主语 谓语 宾语
主语 = 你| 我 | 他 
谓语 = 吃| 玩 
宾语 = 桃子| 皮球

"""

grammar = """
句子 = s_句子 , 连词 句子 | s_句子
连词 = 而且 | 但是 | 不过
s_句子 = 主语 谓语 宾语
主语 = 你| 我 | 他 
谓语 = 吃| 玩 
宾语 = 桃子| 皮球
"""

grammar2 = """
句子 = 打招呼 玩 活动 吗？
打招呼 = 你好 | 您好 | 好久不见
玩 = 需要玩 | 喜欢玩 | 想玩
活动 = 骑马 | 打球 | 喝茶 
"""


def get_grammar(grammar_string):
    grammar_gen = dict()

    for line in grammar_string.split('\n'):
        if not line.strip(): continue

        stmt, expr = line.split('=')

        expressions = expr.split('|')
        grammar_gen[stmt.strip()] = [e.strip() for e in expressions]

    return grammar_gen


choice = lambda t: random.choice(t)


def generate_sentence(gram, target='句子'):
    if target not in gram: return target

    return ''.join([generate_sentence(gram, e) for e in choice(gram[target]).split()])


print('generated: ')
for i in range(10):
    print(generate_sentence(get_grammar(grammar)))
    print(generate_sentence(get_grammar(grammar2)))
