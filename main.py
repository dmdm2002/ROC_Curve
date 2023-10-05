import matplotlib.pyplot as plt

from preprocessing import Preprocessing
from ROC_func import get_FPR_TPR
from sklearn.metrics import auc

AG, DNET, PROPOSED, DCLNET, convnext, vit, maxvit = Preprocessing()()

AG_FPR, AG_TPR, _ = get_FPR_TPR(AG)
DNET_FPR, DNET_TPR, _ = get_FPR_TPR(DNET)
PROPOSED_FPR, PROPOSED_TPR, _ = get_FPR_TPR(PROPOSED)
DCLNET_FPR, DCLNET_TPR, _ = get_FPR_TPR(DCLNET)
CONVNEXT_FPR, CONVNEXT_TPR, _ = get_FPR_TPR(convnext)
ViT_FPR, ViT_TPR, _ = get_FPR_TPR(vit)
MaxViT_FPR, MaxViT_TPR, _ = get_FPR_TPR(maxvit)

plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

PROPOSED_ROC_AUC = auc(PROPOSED_FPR, PROPOSED_TPR)
AG_ROC_AUC = auc(AG_FPR, AG_TPR)
DNET_FPR_ROC_AUC = auc(DNET_FPR, DNET_TPR)
DCLNET_FPR_ROC_AUC = auc(DCLNET_FPR, DCLNET_TPR)
CONVNEXT_FPR_ROC_AUC = auc(CONVNEXT_FPR, CONVNEXT_TPR)
ViT_FPR_ROC_AUC = auc(ViT_FPR, ViT_TPR)
MaxViT_FPR_ROC_AUC = auc(MaxViT_FPR, MaxViT_TPR)

# print(1 - AG_ROC_AUC)
print(f'Proposed:{(1-PROPOSED_ROC_AUC) * 100}')
print(f'AG : {(1-AG_ROC_AUC) * 100}')
print(f'Dnet : {(1 - DNET_FPR_ROC_AUC) * 100}')
print(f'DCL : {(1-DCLNET_FPR_ROC_AUC) * 100}')
print(f'ConvNeXt : {(1-CONVNEXT_FPR_ROC_AUC) * 100}')
print(f'ViT : {(1-ViT_FPR_ROC_AUC) * 100}')
print(f'MaxViT : {(1-MaxViT_FPR_ROC_AUC) * 100}')
print('----------------------------------------------------------------------')
print(f'Proposed:{PROPOSED_ROC_AUC * 100}')
print(f'AG : {AG_ROC_AUC * 100}')
print(f'Dnet : {DNET_FPR_ROC_AUC * 100}')
print(f'DCL : {DCLNET_FPR_ROC_AUC * 100}')
print(f'ConvNeXt : {CONVNEXT_FPR_ROC_AUC * 100}')
print(f'ViT : {ViT_FPR_ROC_AUC * 100}')
print(f'MaxViT : {MaxViT_FPR_ROC_AUC * 100}')

plt.rc('axes', labelsize=15)
plt.rc('legend', fontsize=15)

# print(PROPOSED_FPR, PROPOSED_TPR)
plt.plot(PROPOSED_FPR, PROPOSED_TPR, color='blue', lw=2)
plt.plot(AG_FPR, AG_TPR, color='red', lw=2)
plt.plot(DNET_FPR, DNET_TPR, color='green', lw=2)
plt.plot(DCLNET_FPR, DCLNET_TPR, color='orange', lw=2)
plt.plot(CONVNEXT_FPR, CONVNEXT_TPR, color='gray', lw=2)
plt.plot(ViT_FPR, ViT_TPR, color='yellow', lw=2)
plt.plot(MaxViT_FPR, MaxViT_TPR, color='pink', lw=2)

# plt.plot(PROPOSED_FPR, PROPOSED_TPR)
# plt.plot(AG_FPR, AG_TPR)
# plt.plot(DNET_FPR, DNET_TPR)
plt.plot([1, 0], [0, 1], color='black', linestyle='--')
plt.ylim([0.975, 1])
plt.xlim([0.0, 0.025])

plt.xlabel('False Detection Rate')
plt.ylabel('True Detection Rate')

plt.legend([f'LRFID-Net', f'AG-PAD', f'D-NetPAD', f'DCLNet', 'ConvNeXt', 'ViT', 'MaxViT'],
           loc='upper right', fontsize=10)
plt.show()