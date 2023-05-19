import json
import numpy as np
from scipy.stats import pearsonr

predict_json_path = "/apdcephfs/share_1316500/qiushizhu/eegdata/eeg_pytorch/models_unet/predictions/"
target_json_path = "/apdcephfs/share_1316500/qiushizhu/eegdata/TEST_task2_regression/labels/"


def compute_score(predict_json_path, target_json_path):

    testset1_score = []
    testset2_score = []
    for i in range(2, 86):
        if 2 <= i <= 71:
            k = '%03d' % i
            # print(k)
            with open(predict_json_path + "sub-{}.json".format(k), "r") as f1, \
            open(target_json_path + "sub-{}.json".format(k), "r") as f2:
                predict_dict = json.load(f1)
                target_dict = json.load(f2)
                for key in target_dict.keys():
                    score1_list = []
                    predict_data = predict_dict[key]
                    target_data = target_dict[key]
                    target_data_convert = [i for arr in target_data[0] for i in arr]

                    pccs = pearsonr(predict_data, target_data_convert)

                    score1_list.append(pccs[0])
                average_spk1 = np.average(score1_list)  # average pearson correlation per subject
            testset1_score.append(average_spk1)

        if 72 <= i <= 85:
            k = '%03d' % i
            # print(k)
            with open(predict_json_path + "sub-{}.json".format(k), "r") as f1, \
            open(target_json_path + "sub-{}.json".format(k), "r") as f2:
                predict_dict = json.load(f1)
                target_dict = json.load(f2)
                for key in target_dict.keys():

                    score2_list = []
                    predict_data = np.array(predict_dict[key])

                    target_data = np.array(target_dict[key])
                    target_data_convert = [i for arr in target_data[0] for i in arr]
                    pccs2 = pearsonr(predict_data, target_data_convert)

                    score2_list.append(pccs2[0])
                average_spk2 = np.average(score2_list)  # average pearson correlation per subject
            testset2_score.append(average_spk2)
    assert len(testset1_score) == 70
    assert len(testset2_score) == 14

    return testset1_score, testset2_score


testset1_score, testset2_score = compute_score(predict_json_path, target_json_path)



score1 = np.average(testset1_score)
score2 = np.average(testset2_score)
total = 2.0/3.0 * score1 + 1.0/3.0 * score2
print("score1: ", score1)
print("score2: ", score2)
print("total: ", total)




