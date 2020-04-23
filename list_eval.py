import random
import copy
import sys
import csv


def super_shuffle(lst):
    new_lst = copy.copy(lst)
    random.shuffle(new_lst)
    # for old, new in zip(lst, new_lst):
    #     if old == new:
    #         return super_shuffle(lst)
    return new_lst

def duplicate(testList, n):
    return testList*n

directions = ['left','right','up','down','zoom in','zoom out']
new_labels_list = duplicate(directions,10)
shuffled_labels = super_shuffle(new_labels_list)
print(shuffled_labels)

import csv
import  pandas as pd
df = pd.DataFrame(list(zip(*[shuffled_labels]))).add_prefix('label')
df.to_csv('eval_list.csv', index=False)