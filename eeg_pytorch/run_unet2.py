import torch
import torch.nn  as nn
import numpy as np 

from unet import Unet
from load_data import EEGdataset
import metric
import tqdm
import os 



n_epochs=100
save_path = "/apdcephfs/share_1316500/qiushizhu/eegdata/eeg_pytorch/models_unet2/"
log_file = "run.log"

if not os.path.exists(save_path):
    os.makedirs(save_path)

unetmodel = Unet()

# unetmodel = torch.nn.DataParallel(unetmodel, device_ids = [0]).cuda()

total_parameters = sum(p.numel() for p in unetmodel.parameters() if p.requires_grad)
print("total_parameters", total_parameters)

eegdata_train = EEGdataset(filepaths="/apdcephfs/share_1316500/qiushizhu/eegdata/split_data", mode="train")
eegdata_val = EEGdataset(filepaths="/apdcephfs/share_1316500/qiushizhu/eegdata/split_data", mode="val")
eegdata_test = EEGdataset(filepaths="/apdcephfs/share_1316500/qiushizhu/eegdata/split_data", mode="test")


optimizer=torch.optim.AdamW(unetmodel.parameters(), lr=1e-3, weight_decay=1e-4)

# scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5)
# num_steps = len(torch.utils.data.DataLoader(eegdata_train, batch_size=1, shuffle=True)) * n_epochs
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
# warmup_scheduler = warmup.LinearWarmup(optimizer,warmup_period=10000)


for epoch in range(0, n_epochs):
    
    print("epoch:{}".format(epoch))

    dataloader_train = torch.utils.data.DataLoader(eegdata_train, batch_size=1, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(eegdata_val, batch_size=1, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(eegdata_test, batch_size=1, shuffle=False)
    
    total_valild_loss = []
    unetmodel.eval()
    with torch.no_grad():
        for batch_id, (feat, label) in enumerate(tqdm.tqdm(dataloader_val)):
            feat = feat.squeeze(0)

            label = label.squeeze(0).squeeze(1)
            out = unetmodel(feat.squeeze(0))
            out = out.squeeze(1)      
  
            val_loss = -metric.pearson_torch(label, out)
            # print(val_loss)
        total_valild_loss.append(val_loss.detach().item())
    print("average valid loss", sum(total_valild_loss)/len(total_valild_loss))


    unetmodel.train()
    total_train_loss = []
    for batch_id, (feat, label) in enumerate(tqdm.tqdm(dataloader_train)):
        optimizer.zero_grad()
        feat = feat.squeeze(0)
        label = label.squeeze(0).squeeze(1)
        out = unetmodel(feat.squeeze(0))
        out = out.squeeze(1)  

        loss = -metric.pearson_torch(label, out)
        loss.backward()
        optimizer.step()
        # with warmup_scheduler.dampening():
        #     lr_scheduler.step()

        if np.isnan(loss.detach().item()):
            print('found a nan at {}'.format(batch_id))
        total_train_loss.append(loss.detach().item())
    print("average train loss", sum(total_train_loss)/len(total_train_loss))
    print("lr:", optimizer.state_dict()['param_groups'][0]['lr'])
    # scheduler.step()
    
    if epoch % 2 == 0:
        model_filename = '{}/model_{}.pt'.format(save_path, epoch)
        torch.save(unetmodel.state_dict(), model_filename)

    with open(save_path + log_file, "a") as f1:
        f1.write("epoch: " + str(epoch) + "\t" + "average valid loss: " + str(sum(total_valild_loss)/len(total_valild_loss)) + "\t" + "average train loss: " + str(sum(total_train_loss)/len(total_train_loss)) + "\t" + "lr: " + str(optimizer.state_dict()['param_groups'][0]['lr']) + "\n")






# if __name__ == "__main__":
#     x = torch.randn(740,64,640)
#     # conv1 = nn.Sequential(
#     #         nn.Conv1d(64, 64, 3, padding=1),
#     #         nn.ReLU(),
#     #         nn.Conv1d(64, 1, 3, padding=1),
#     #         nn.ReLU()
#     #     )
#     # out = conv1(x)
#     # print(out.size())
#     net = CNNnet()
#     out = net(x)
#     print(out.size())