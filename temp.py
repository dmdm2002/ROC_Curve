import matplotlib.pyplot as plt
import numpy as np

from preprocessing import Preprocessing
from ROC_func import get_FPR_TPR
from sklearn.metrics import auc

path_1_fold = ['prov/Proposed/warsaw/1-fold/B_iris_proba.csv', 'prov/AGPAD/warsaw/1-fold/original.csv', 'Z:/Iris_dataset/scores/warsaw/DNetPad/1-fold/Scores.csv', 'prov/DCLNet/warsaw/1-fold/warsaw_1-fold_prov_9.csv']
path_2_fold = ['prov/Proposed/warsaw/1-fold_2/A_iris_proba.csv', 'prov/AGPAD/warsaw/1-fold_2/original.csv', 'Z:/Iris_dataset/scores/warsaw/DNetPad/1-fold_2/Scores.csv', 'prov/DCLNet/warsaw/1-fold_2/warsaw_1-fold_prov_9.csv']

PROPOSED_1, AG_1, DNET_1, DCLNET_1 = Preprocessing(path_1_fold[0], path_1_fold[1], path_1_fold[2], path_1_fold[3])()

PROPOSED_FPR_1, PROPOSED_TPR_1, _ = get_FPR_TPR(PROPOSED_1)
AG_FPR_1, AG_TPR_1, _ = get_FPR_TPR(AG_1)
DNET_FPR_1, DNET_TPR_1, _ = get_FPR_TPR(DNET_1)
DCLNET_FPR_1, DCLNET_TPR_1, _ = get_FPR_TPR(DCLNET_1)

PROPOSED_2, AG_2, DNET_2, DCLNET_2 = Preprocessing(path_2_fold[0], path_2_fold[1], path_2_fold[2], path_2_fold[3])()

PROPOSED_FPR_2, PROPOSED_TPR_2, _ = get_FPR_TPR(PROPOSED_2)
AG_FPR_2, AG_TPR_2, _ = get_FPR_TPR(AG_2)
DNET_FPR_2, DNET_TPR_2, _ = get_FPR_TPR(DNET_2)
DCLNET_FPR_2, DCLNET_TPR_2, _ = get_FPR_TPR(DCLNET_2)

plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

print(len(PROPOSED_FPR_1), len(PROPOSED_TPR_1))
print(len(AG_FPR_1), len(AG_TPR_1))
print(len(DNET_FPR_1), len(DNET_TPR_1))
print(len(DCLNET_FPR_1), len(DCLNET_TPR_1))

print(len(PROPOSED_FPR_2), len(PROPOSED_TPR_2))
print(len(AG_FPR_2), len(AG_TPR_2))
print(len(DNET_FPR_2), len(DNET_TPR_2))
print(len(DCLNET_FPR_2), len(DCLNET_TPR_2))


# PROPOSED_ROC_AUC = auc(PROPOSED_FPR, PROPOSED_TPR)
# AG_ROC_AUC = auc(AG_FPR, AG_TPR)
# DNET_FPR_ROC_AUC = auc(DNET_FPR, DNET_TPR)
# DCLNET_FPR_ROC_AUC = auc(DCLNET_FPR, DCLNET_TPR)
