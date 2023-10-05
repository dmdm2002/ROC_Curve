import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dclnet_predict_df = pd.read_csv('prov/DCLNet/nd/1-fold/original/nd_1-fold_prov_9.csv')
dclnet_predict_df_2 = pd.read_csv('prov/DCLNet/nd/1-fold/original/nd_1-fold_prov_9.csv')
dclnet_predict_df = pd.concat([dclnet_predict_df, dclnet_predict_df_2])

dclnet_predict_df.drop(['Unnamed: 0'], axis=1, inplace=True)
predict = np.array(dclnet_predict_df)

label_1 = pd.read_csv('prov/DCLNet/nd/1-fold/original/nd_1-fold_label_9.csv')
label_2 = pd.read_csv('prov/DCLNet/nd/1-fold/original/nd_1-fold_label_9.csv')
label_df = pd.concat([label_1, label_2])
label_df.drop(['Unnamed: 0'], axis=1, inplace=True)
label = np.array(label_df)

predict_list = []

for i in range(len(predict)):
    if predict[i] >= 0.5:
        predict_list.append([i, label[i], 1])
    else:
        predict_list.append([i, label[i], 0])

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

print(tp)
print(fp)
print(tn)
print(fn)

acer = (apcer + bpcer) / 2
print(f"APCER : {apcer * 100}")
print(f"BPCER : {bpcer * 100}")
print(f"BPCER : {acer * 100}")
# full_apcer = 0
# full_bpcer = 0
# count = 1
# print(path_list)
# for path in path_list:
#     print(path[0], path[1])
#     score = pd.read_csv(path[0])
#     label = pd.read_csv(path[1])
#
#     score = np.array(score.drop('Unnamed: 0', axis=1).astype('float32'))
#     label = np.array(label.drop('Unnamed: 0', axis=1).astype('float32'))
#
#     predict_list = []
#     for i in range(len(score)):
#         if score[i] >= 0.5:
#             predict_list.append([i, label[i], 1])
#         else:
#             predict_list.append([i, label[i], 0])
#
#     tp = 0
#     fp = 0
#     tn = 0
#     fn = 0
#
#     for i in range(len(predict_list)):
#         if predict_list[i][1] == 1 and predict_list[i][2] == 0:
#             fp = fp + 1
#         elif predict_list[i][1] == 1 and predict_list[i][2] == 1:
#             tp = tp + 1
#         elif predict_list[i][1] == 0 and predict_list[i][2] == 1:
#             fn = fn + 1
#         elif predict_list[i][1] == 0 and predict_list[i][2] == 0:
#             tn = tn + 1
#
#     apcer = fn / (tp + fn)
#     bpcer = fp / (fp + tn)
#
#     print(tp)
#     print(fp)
#     print(tn)
#     print(fn)
#
#     count = count + 1
#
#     print(f"APCER : {apcer}")
#     print(f"BPCER : {bpcer}")
#
#     full_apcer = full_apcer + apcer
#     full_bpcer = full_bpcer + bpcer
#
# a = full_apcer / 2
# b = full_bpcer / 2
# c = (a + b) / 2
#
# print(f"avg APCER : {a * 100}")
# print(f"avg BPCER : {b * 100}")
# print(f"avg BPCER : {c * 100}")