import torch
import torch.nn  as nn
import numpy as np 

from model import CNNnet
from load_data import EEGdataset
import metric
import tqdm

n_epochs=100

log_file = "train1.log"

cnnmodel = CNNnet()
cnnmodel.cuda()

eegdata_train = EEGdataset(filepaths="./split_data", mode="train")
eegdata_val = EEGdataset(filepaths="./split_data", mode="val")
eegdata_test = EEGdataset(filepaths="./split_data", mode="test")


optimizer=torch.optim.Adam(cnnmodel.parameters(), lr=5e-4, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5)

for epoch in range(0, n_epochs):
    
    print("epoch:{}".format(epoch))

    dataloader_train = torch.utils.data.DataLoader(eegdata_train, batch_size=1, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(eegdata_val, batch_size=1, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(eegdata_test, batch_size=1, shuffle=False)
    
    total_valild_loss = []
    cnnmodel.eval()
    with torch.no_grad():
        for batch_id, (feat, label) in enumerate(tqdm.tqdm(dataloader_test)):
            feat = feat.squeeze(0)
            label = label.squeeze(0).transpose(1,2)
            out = cnnmodel(feat.squeeze(0))
            out = out.transpose(1,2)            
            val_loss = -metric.pearson_torch(label, out)
        total_valild_loss.append(val_loss.detach().item())
    print("average valid loss", sum(total_valild_loss)/len(total_valild_loss))


    cnnmodel.train()
    total_train_loss = []
    for batch_id, (feat, label) in enumerate(tqdm.tqdm(dataloader_train)):
        optimizer.zero_grad()
        feat = feat.squeeze(0)
        label = label.squeeze(0).transpose(1,2)
        out = cnnmodel(feat.squeeze(0))
        out = out.transpose(1,2)

        loss = -metric.pearson_torch(label, out)
        loss.backward()
        optimizer.step()
        
        if np.isnan(loss.detach().item()):
            print('found a nan at {}'.format(batch_id))
        total_train_loss.append(loss.detach().item())
    print("average train loss", sum(total_train_loss)/len(total_train_loss))
    print("lr:", optimizer.state_dict()['param_groups'][0]['lr'])
    scheduler.step()
    
    if epoch % 5 == 0:
        model_filename = './models/model_{}.pt'.format(epoch)
        torch.save(cnnmodel.state_dict(), model_filename)
    
    with open(log_file, "a") as f1:
        f1.write("epoch: " + str(epoch) + "\t" + "average valid loss: " + str(sum(total_valild_loss)/len(total_valild_loss)) + "\t" + "average train loss: " + str(sum(total_train_loss)/len(total_train_loss)) + "\t" + "lr: " + str(optimizer.state_dict()['param_groups'][0]['lr']) + "\n")





