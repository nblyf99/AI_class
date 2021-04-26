import time


def very_complxity_compute(n):
    time.sleep(2)
    return n ** 3


def get_number_1(n):
    for i in range(n):
        ans = very_complxity_compute(i)
        yield ans

def get_number_2(n):
    return (very_complxity_compute(i) for i in range(n))


begin = time.time()
for ans in get_number_2(3):
    now_ = time.time()
    print('used time = {}'.format(now_ - begin))
    print(ans)
