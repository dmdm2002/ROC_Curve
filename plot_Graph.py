import matplotlib.pyplot as plt
import numpy as np
from Get_FAR_FRR import FAR_FRR
from preprocessing import Preprocessing

"""Load Scores"""
AG, DNET, PROPOSED, DCLNET = Preprocessing()()

"""GET FAR FRR"""
far_frr = FAR_FRR()
PROPOSED_FRR_rate, PROPOSED_FAR_rate = far_frr.cal_frr_far(PROPOSED)
AG_FRR_rate, AG_FAR_rate = far_frr.cal_frr_far(AG)
DNET_FRR_rate, DNET_FAR_rate = far_frr.cal_frr_far(DNET)
DCL_FRR_rate, DCL_FAR_rate = far_frr.cal_frr_far(DCLNET)

"""PRINT FAR FRR INFO"""
print('-------------------------PROPOSED-------------------------')
far_frr.find_EER(PROPOSED_FRR_rate, PROPOSED_FAR_rate)
print('-------------------------AG-------------------------')
far_frr.find_EER(AG_FRR_rate, AG_FAR_rate)
print('-------------------------DNET-------------------------')
far_frr.find_EER(DNET_FRR_rate, DNET_FAR_rate)
print('-------------------------DCL-------------------------')
far_frr.find_EER(DCL_FRR_rate, DCL_FAR_rate)

"""PLOT GRAPH"""
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rc('axes', labelsize=15)
plt.rc('legend', fontsize=15)

plt.step(1 - np.array(PROPOSED_FAR_rate), PROPOSED_FRR_rate, color='blue', lw=2)
plt.step(1 - np.array(AG_FAR_rate), AG_FRR_rate, color='red', lw=2)
plt.step(1 - np.array(DNET_FAR_rate), DNET_FRR_rate, color='green', lw=2)
plt.step(1 - np.array(DCL_FAR_rate), DCL_FRR_rate, color='orange', lw=2)

plt.plot([1, 0], [0, 1], color='black', linestyle='--')
plt.ylim([0.975, 1])
plt.xlim([0.0, 0.025])

plt.xlabel('False Detection Rate')
plt.ylabel('True Detection Rate')
# plt.legend([f'Proposed AUC : {PROPOSED_ROC_AUC:.5f}', f'AG-PAD AUC : {AG_ROC_AUC:.5f}',
#             f'D-NetPAD AUC : {DNET_FPR_ROC_AUC:.5f}', f'DCLNet AUC : {DCLNET_FPR_ROC_AUC:.5f}', 'EER line'],
#            loc='lower left')
plt.legend([f'LRFID-Net', f'AG-PAD', f'D-NetPAD', f'DCLNet'],
           loc='upper right')
plt.show()