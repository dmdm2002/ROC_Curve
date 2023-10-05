import pandas as pd
import numpy as np
import re


class Preprocessing(object):
    def __init__(self):
        super(Preprocessing, self).__init__()
        # self.AGPAD_PATH = 'prov/AGPAD/warsaw/1-fold/blur.csv'
        self.AGPAD_PATH = 'Z:/1st/Iris_dataset/scores/ND/AGPAD/1-fold/original.csv'
        self.DNET_PATH = 'Z:/1st/Iris_dataset/scores/nd/DNetPad/1-fold_4/Scores_17.csv'
        # self.DNET_PATH = 'Z:/Iris_dataset/scores/warsaw/DNetPad/1-fold_4/Scores_16.csv'
        # self.DNET_PATH = 'Z:/Iris_dataset/scores/warsaw/DNetPad/1-fold_6/Scores_blur_22.csv'
        self.PROPOSED_PATH = 'prov/Proposed/nd/1-fold/B_46.csv'
        self.DCLNET_PATH = 'prov/DCLNet/nd/1-fold/original/nd_1-fold_prov_9.csv'

        self.convnext = 'Z:/1st/Iris_dataset/scores/nd/convnext/1-fold/convnext_base.csv'
        self.vit = 'Z:/1st/Iris_dataset/scores/nd/ViT/1-fold/vit_base_patch16_224.csv'
        self.max_vit = 'Z:/1st/Iris_dataset/scores/nd/maxViT/1-fold/maxvit_base.csv'

        # self.AGPAD_PATH_2 = 'prov/AGPAD/warsaw/2-fold/blur.csv'
        self.AGPAD_PATH_2 = 'Z:/1st/Iris_dataset/scores/ND/AGPAD/2-fold/original.csv'
        # self.DNET_PATH_2 = 'Z:/1st/Iris_dataset/scores/warsaw/DNetPad/1-fold_2/Scores_blur.csv'
        # self.DNET_PATH_2 = 'Z:/1st/Iris_dataset/scores/nd/DNetPad/2-fold/Scores_blur.csv'
        self.DNET_PATH_2 = 'Z:/1st/Iris_dataset/scores/nd/DNetPad/2-fold_4/Scores_15.csv'
        # Warsaw에서 Dnet은 22가 best
        # self.DNET_PATH_2 = 'Z:/Iris_dataset/scores/nd/DNetPad/2-fold_4/Scores_blur_18.csv'
        self.PROPOSED_PATH_2 = 'prov/Proposed/nd/2-fold/A_48.csv'
        self.DCLNET_PATH_2 = 'prov/DCLNet/nd/2-fold/original/nd_2-fold_prov_9.csv'

        self.convnext_2 = 'Z:/1st/Iris_dataset/scores/nd/convnext/2-fold/convnext_base.csv'
        self.vit_2 = 'Z:/1st/Iris_dataset/scores/nd/ViT/1-fold/vit_base_patch16_224.csv'
        self.max_vit_2 = 'Z:/1st/Iris_dataset/scores/nd/maxViT/2-fold/maxvit_base.csv'

    def add_model_prepro(self, one, two):
        predict_df = pd.read_csv(one)
        predict_df_2 = pd.read_csv(two)

        concat_predict_df = pd.concat([predict_df, predict_df_2])
        concat_predict_df.drop(['0'], axis=1, inplace=True)
        predict_arr = np.array(concat_predict_df)

        predict_cls = []
        for score in predict_arr:
            predict_cls.append([score[1], score[0]])

        return np.array(predict_cls)


    def DCLNet_prepro(self):
        dclnet_predict_df = pd.read_csv(self.DCLNET_PATH)
        dclnet_predict_df_2 = pd.read_csv(self.DCLNET_PATH_2)
        dclnet_predict_df = pd.concat([dclnet_predict_df, dclnet_predict_df_2])

        dclnet_predict_df.drop(['Unnamed: 0'], axis=1, inplace=True)
        predict = np.array(dclnet_predict_df)

        label_1 = pd.read_csv('prov/DCLNet/nd/1-fold/original/nd_1-fold_label_9.csv')
        label_2 = pd.read_csv('prov/DCLNet/nd/2-fold/original/nd_2-fold_label_9.csv')
        label_df = pd.concat([label_1, label_2])
        label_df.drop(['Unnamed: 0'], axis=1, inplace=True)

        label = np.array(label_df)

        predict_cls = []
        for i in range(len(predict)):
            predict_cls.append([label[i][0], predict[i][0]])

        return np.array(predict_cls)

    def Proposed_prepro(self):
        proposed_predict_df = pd.read_csv(self.PROPOSED_PATH)
        proposed_predict_df_2 = pd.read_csv(self.PROPOSED_PATH_2)
        proposed_predict_df = pd.concat([proposed_predict_df, proposed_predict_df_2])
        predict_arr = np.array(proposed_predict_df)

        predict_cls = []
        for score in predict_arr:
            predict_cls.append([score[2], score[1]])

        return np.array(predict_cls)

    def AGPAD_Prepro(self):
        path_list = [self.AGPAD_PATH, self.AGPAD_PATH_2]

        full_list = []

        count = 1
        for path in path_list:
            agpad_predict_df = pd.read_csv(path)
            proba_value = np.array(agpad_predict_df.drop('Unnamed: 0', axis=1).astype('float32'))

            predict_list = []

            # hold = 2592
            # if count == 2:
            #     hold = 2576
            hold = 2508
            if count == 2:
                hold = 2276

            print(hold)

            for i in range(len(proba_value)):
                if proba_value[i][0] > proba_value[i][1]:
                    if i <= hold:
                        predict_list.append([0, proba_value[i][1]])
                    else:
                        predict_list.append([1, proba_value[i][1]])
                else:
                    if i <= hold:
                        predict_list.append([0, proba_value[i][1]])
                    else:
                        predict_list.append([1, proba_value[i][1]])

            full_list = full_list + predict_list
            count = count + 1

        # np.array(full_list)
        return np.array(full_list)

    def DNet_prepro(self):
        dnet_predict_df = pd.read_csv(self.DNET_PATH, header=None)
        dnet_predict_df_2 = pd.read_csv(self.DNET_PATH_2, header=None)
        dnet_predict_df = pd.concat([dnet_predict_df, dnet_predict_df_2])
        results_arr = np.array(dnet_predict_df)
        # print(results_arr)

        label = [re.split('/', re.split("\\\\", sub_arr[0])[0])[-1] for sub_arr in results_arr]
        new_label = []

        for i in range(len(label)):
            if label[i] == 'live':
                new_label.append(1)
            elif label[i] == 'fake':
                new_label.append(0)

        predict_label = []
        for i in range(len(results_arr)):
            predict_label.append([new_label[i], results_arr[i][1]])

        # print(predict_label)

        return np.array(predict_label)

    def __call__(self, *args, **kwargs):
        return self.AGPAD_Prepro(), self.DNet_prepro(), self.Proposed_prepro(), self.DCLNet_prepro(), \
               self.add_model_prepro(self.convnext, self.convnext_2), self.add_model_prepro(self.vit, self.vit_2), \
               self.add_model_prepro(self.max_vit, self.max_vit_2)
