def flatten_list(nested_list: list):
    # 方法1
    #return [item for sublist in nested_list for item in sublist]

    # 方法2
    flattened_list = []
    for sublist in nested_list:
        for item in sublist:
            flattened_list.append(item)
    return flattened_list

    

def char_count(s: str):
    return {char: s.count(char) for char in set(s)}

import time
import random

def split_list(num):
    elements = list(range(num))
    random.shuffle(elements)
    split = random.sample(range(1, num), num // 10)
    split.sort()

    return [elements[i:j] for i,j in zip([0] + split, split + [None])]

#print(split_list(100))

for i in range(3):
    numOfElements = 10**(i+7)
    alist = split_list(numOfElements)

    begin = time.time()
    flatten_list(alist)
    end = time.time()

    print("The runnining time of 10^{} = {}.".format(i+3, end - begin))

