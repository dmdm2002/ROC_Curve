import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def get_FPR_TPR(label_pr_list):
    label = [label[0] for label in label_pr_list]
    score = [score[1] for score in label_pr_list]

    return roc_curve(label, score, pos_label=1)