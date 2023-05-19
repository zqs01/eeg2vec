import torch
import torch.nn  as nn
import numpy as np 

from unet import Unet
from load_data import EEGdataset
import metric
import tqdm
import json


eegdata_val = EEGdataset(filepaths="/apdcephfs/share_1316500/qiushizhu/eegdata/split_data", mode="val")
dataloader_val = torch.utils.data.DataLoader(eegdata_val, batch_size=1, shuffle=False)

unetmodel = Unet().cuda()

checkpoint = torch.load("/apdcephfs/share_1316500/qiushizhu/eegdata/eeg_pytorch/models_unet/model_8.pt", map_location="cuda:0")
# print(checkpoint)
unetmodel.load_state_dict(checkpoint, strict=True)
unetmodel.eval()

total_valild_loss = []
with torch.no_grad():
    for i in range(2, 86):
        k = '%03d' % i
        out_json = {}
        
        with open("/apdcephfs/share_1316500/qiushizhu/eegdata/TEST_task2_regression/preprocessed_eeg/" + "sub-{}.json".format(k), "r") as f1, \
            open("/apdcephfs/share_1316500/qiushizhu/eegdata/eeg_pytorch/models_unet/predictions/" + "sub-{}.json".format(k), "w") as f2:
            input_json = json.load(f1)
            for key in input_json.keys():
                input_data = input_json[key]
                
                data_mean = np.expand_dims(np.mean(input_data, axis=1), axis=1)
                data_std = np.expand_dims(np.std(input_data, axis=1), axis=1)
                input_data = (input_data - data_mean) / data_std
                
                input_data = torch.tensor(input_data, dtype=torch.float32)
                input_data = input_data.unfold(0, 640, 640)
                

                print("input_data", input_data.size())
                predictions = unetmodel(input_data.cuda())   
                predictions = predictions.squeeze(1).reshape(-1, 1) 
                # print("predictions", predictions)
                predictions = [np.array(value).tolist() for value in np.squeeze(predictions.cpu())]
                out_json[key] = predictions
            
            json.dump(out_json, f2)




