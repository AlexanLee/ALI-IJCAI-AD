# __encoding__:utf-8

import pandas as pd


def merge():
    base = pd.read_csv("result/result0418_base.txt", sep="\s+")
    model = pd.read_csv("result/result0416.txt", sep="\s+")
    print(base['instance_id'][0], base['predicted_score'][0])
    i = 0
    j = 0
    for k in base['instance_id']:
        for mk in base['instance_id']:
            if k == mk:
                sub = abs(base['predicted_score'][i] - model['predicted_score'][j])
                print(i, j, base['predicted_score'][i], model['predicted_score'][j], sub)
                j = 0
                break
            j += 1
        i += 1


if __name__ == "__main__":
    print('main')
    merge()
