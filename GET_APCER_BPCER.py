import pandas as pd
import numpy as np

from preprocessing import Preprocessing

_TH_MIN = 0
_TH_MAX = 1
_STEP_SZ = 0.001

_THRES = [0.5]


class FAR_FRR(object):
    def __init__(self):
        super(FAR_FRR, self).__init__()

        # """Load label and score"""
        # AG, DNET, PROPOSED, DCLNET = Preprocessing()()
        # a, b = self.cal_frr_far(AG)
        # self.find_EER(a, b)
        # self.F_data, self.L_data = self.split_F_L(AG)
        # c, d = self.count_frr_far(a, b)

    def split_F_L(self, scores):
        F_data = [score[1] for score in scores if score[0] == 0]
        L_data = [score[1] for score in scores if score[0] == 1]

        return F_data, L_data

    def count_frr_far(self, F_data, L_data):
        # F_data, L_data = self.split_F_L(data)
        FRR_COUNT = []
        FAR_COUNT = []

        for th in _THRES:
            false_reject = len([data for data in L_data if data > 0.5])
            false_accept = len([data for data in F_data if data < 0.5])

            FRR_COUNT.append(false_reject)
            FAR_COUNT.append(false_accept)

        return FRR_COUNT, FAR_COUNT

    def cal_frr_far(self, scores):
        F_data, L_data = self.split_F_L(scores)
        FRR_COUNT, FAR_COUNT = self.count_frr_far(F_data, L_data)
        FRR_rate = []
        FAR_rate = []

        for count in FRR_COUNT:
            # print(count)
            frr = count/int(len(L_data))
            FRR_rate.append(frr)

        for count in FAR_COUNT:
            far = count/int(len(F_data))
            FAR_rate.append(far)

        return FRR_rate, FAR_rate

    def find_EER(self, false_reject_rate, false_accept_rate):
        APCER = false_reject_rate[0]
        BPCER = false_accept_rate[0]

        ACER = (APCER + BPCER) / 2

        print(f'|| APCER : {(1 - APCER) * 100} | BPCER : {(1- BPCER) * 100} | ACER : {(1 - ACER) * 100} ||')


    # def __call__(self, *args, **kwargs):
    #     return

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
