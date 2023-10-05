import pandas as pd
import numpy as np
import re

path = 'prov/Proposed/warsaw/1-fold/B_iris_proba.csv'
path_2 = 'prov/Proposed/warsaw/2-fold/A_iris_proba.csv'

path_list = [path, path_2]
full_apcer = 0
full_bpcer = 0
full_acer = 0

for i in path_list:
    original = pd.read_csv(path)
    a = pd.read_csv(path_2)

    original = pd.concat([original, a])
    print(original)

    label = np.array(original['2'].astype('float32'))
    proba_value = np.array(original.drop('2', axis=1).astype('float32'))

    predict_list = []

    for j in range(len(proba_value)):
        if proba_value[j][0] > proba_value[j][1]:
            predict_list.append([j, 0, label[j]])
        else:
            predict_list.append([j, 1, label[j]])

    print(len(predict_list))

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(predict_list)):
        if predict_list[i][1] == 1 and predict_list[i][2] == 0:
            fp = fp + 1
        elif predict_list[i][1] == 1 and predict_list[i][2] == 1:
            tp = tp + 1
        elif predict_list[i][1] == 0 and predict_list[i][2] == 1:
            fn = fn + 1
        elif predict_list[i][1] == 0 and predict_list[i][2] == 0:
            tn = tn + 1

    apcer = fn / (tp + fn)
    bpcer = fp / (fp + tn)

    # print(tp)
    # print(fp)
    # print(tn)
    # print(fn)

    # print(6 / 2509)
    print(f"APCER : {apcer * 100}")
    print(f"BPCER : {bpcer * 100}")
    print(f"ACER : {((apcer + bpcer) / 2) * 100}")

    full_apcer = full_apcer + apcer
    full_bpcer = full_bpcer + bpcer
    full_acer = full_acer + ((apcer + bpcer) / 2)

a = full_apcer / 2
b = full_bpcer / 2
c = (a + b) / 2
d = full_acer / 2
print(f"avg APCER : {a * 100}")
print(f"avg BPCER : {b * 100}")
print(f"avg ACER : {c * 100}")