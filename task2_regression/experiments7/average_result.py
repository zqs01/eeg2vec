import json 

total = []
with open("/data/zhu/eeg/auditory-eeg-challenge-2023-code/task2_regression_upload/experiments7bc/results_vlaai15/eval.json", "r") as f1:
    json_file = json.load(f1)
    for key in json_file.keys():
        total.append(float(json_file[key]["pearson_metric"]))
    average = sum(total)/len(json_file.keys())
    print(average)