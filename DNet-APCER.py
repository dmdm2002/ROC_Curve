import pandas as pd
import numpy as np
import re

path = 'Z:/Iris_dataset/scores/warsaw/DNetPad/1-fold_2/Scores.csv'
path_2 = 'Z:/Iris_dataset/scores/warsaw/DNetPad/2-fold_4/Scores_18.csv'

path_list = [path, path_2]
full_apcer = 0
full_bpcer = 0
for i in path_list:
    print('-----------------------------------------------------------------------')
    print(i)
    results_df = pd.read_csv(i, header=None)
    results_arr = np.array(results_df)

    label = [re.split('/', re.split("\\\\", sub_arr[0])[0])[-1] for sub_arr in results_arr]
    new_label = []
    for i in range(len(label)):
        if label[i] == 'live':
            new_label.append(1)
        elif label[i] == 'fake':
            new_label.append(0)

    predict_label = []
    for sub_arr in results_arr:
        if sub_arr[1] > 0.6:
            predict_label.append(1)
        else:
            predict_label.append(0)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(predict_label)):
        if new_label[i] == 1 and predict_label[i] == 1:
            tp = tp + 1

        elif new_label[i] == 0 and predict_label[i] == 0:
            tn = tn + 1

        elif new_label[i] == 1 and predict_label[i] == 0:
            fn = fn + 1

        elif new_label[i] == 0 and predict_label[i] == 1:
            fp = fp + 1

    apcer = fn / (tp + fn)
    bpcer = fp / (fp + tn)

    print(tp)
    print(fp)
    print(tn)
    print(fn)

    # print(6 / 2509)
    print(f"APCER : {apcer}")
    print(f"BPCER : {bpcer}")

    full_apcer = full_apcer + apcer
    full_bpcer = full_bpcer + bpcer
print('-----------------------------------------------------------------------')
a = full_apcer / 2
b = full_bpcer / 2
c = (a + b) / 2

print(f"avg APCER : {a * 100}")
print(f"avg BPCER : {b * 100}")
print(f"avg BPCER : {c * 100}")