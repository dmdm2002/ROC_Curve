import pandas as pd
import numpy as np

from preprocessing import Preprocessing

_TH_MIN = 0
_TH_MAX = 1
_STEP_SZ = 0.001

_THRES = np.arange(_TH_MIN, _TH_MAX, _STEP_SZ)


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
            false_reject = len([data for data in L_data if data > th])
            false_accept = len([data for data in F_data if data < th])

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
        # find_min_point
        min_frr = ''
        min_far = ''
        min_abs_value = 1000
        for frr, far in zip(false_reject_rate, false_accept_rate):
            frr_far_diff = abs(frr - far)
            if frr_far_diff < min_abs_value:
                min_abs_value = frr_far_diff
                min_frr = frr
                min_far = far
                # print("|frr : %.3f| |far : %.3f| |abs_diff : %.3f| |min_frr : %.3f| |min_far : %.3f| |min_abs_value %.3f: |" %(frr, far, frr_far_diff, min_frr, min_far, min_abs_value))
            else:
                continue  # print("|frr : %.3f| |far : %.3f| |abs_diff : %.3f| |min_frr : %.3f| |min_far : %.3f| |min_abs_value %.3f: |" %(frr, far, frr_far_diff, min_frr, min_far, min_abs_value))
        print("MINMUM_DIFF : %.5f FINAL FRR : %.5f FINAL FAR : %.5f" % (min_abs_value, min_frr, min_far))
        EER = (min_frr + min_far) / 2 * 100

        print("FINAL EER((FRR+FAR)/2 at MINMUM_DIFF point) : %.5f%%" % EER)

        APCER = min_frr
        BPCER = min_far

        ACER = (APCER + BPCER) / 2

        print(f'|| APCER : {(1 - APCER) * 100} | BPCER : {(1- BPCER) * 100} | ACER : {(1 - ACER) * 100} ||')