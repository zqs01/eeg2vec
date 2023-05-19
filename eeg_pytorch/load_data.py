import torch 
import numpy as np
import itertools 
import os 
import glob


class EEGdataset(torch.utils.data.Dataset):
    def __init__(self, 
                filepaths,
                frame_length=640,
                frame_step=64,
                mode="train"
                ):
                super().__init__()

                self.filepaths=filepaths
                self.frame_length=frame_length
                self.frame_step=frame_step
                self.mode=mode
                self.new_files = self.load_eeg()

    def load_eeg(self):

        new_files = []
        features = ["eeg"] + ["envelope"]
        if self.mode== "train":
            train_files = [x for x in glob.glob(os.path.join(self.filepaths, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        if self.mode== "val":
            train_files = [x for x in glob.glob(os.path.join(self.filepaths, "val_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        if self.mode== "test":
            train_files = [x for x in glob.glob(os.path.join(self.filepaths, "test_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        
        grouped = itertools.groupby(sorted(train_files), lambda x: "_-_".join(os.path.basename(x).split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            new_files += [sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)]
        
        # data = []
        # new_data_x0 = []
        # new_data_x1 = []
        # for i in range(len(new_files)):
        #     for feature in new_files[i]:
        #         data += [np.load(feature).astype(np.float32)]

        #     x0 = torch.tensor(data[0])
        #     x1 = torch.tensor(data[1])
        #     x0_unfold = x0.unfold(0, 640, 64)
        #     x1_unfold = x1.unfold(0, 640, 64)
        #     new_data_x0.append(x0_unfold)
        #     new_data_x1.append(x1_unfold)
        # new_data_x0 = torch.stack(new_data_x0, dim=0)
        # new_data_x1 = torch.stack(new_data_x1, dim=0)   
        # print(new_data_x0.size())  

        return new_files
    

    def __len__(self):
        return len(self.new_files)
    
    def __getitem__(self, idx):
        data = []
        unfold_data = []
        for feature in self.new_files[idx]:
            data += [np.load(feature).astype(np.float32)]
        

        # Normalize data
        data_mean = np.expand_dims(np.mean(data[0], axis=1), axis=1)
        data_std = np.expand_dims(np.std(data[0], axis=1), axis=1)
        x0 = (data[0] - data_mean) / data_std

        x0 = torch.tensor(x0)
        x1 = torch.tensor(data[1])

        x0_unfold = x0.unfold(0, 640, 64)
        x1_unfold = x1.unfold(0, 640, 64)
        # print(x0.size())
        # print(x0_unfold.size())
        # print(x1.size())
        # print(x1_unfold.size())
        # print(x0_unfold.size())
        return x0_unfold.cuda(), x1_unfold.cuda()
        # return tuple(torch.tensor(x) for x in data)
    
    # def collate(self,batch):


    def new_frame(self, signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
        """
        equivalent of tf.signal.frame
        """
        signal_length = signal.shape[axis]
        if pad_end:
            frames_overlap = frame_length - frame_step
            rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
            pad_size = int(frame_length - rest_samples)
            if pad_size != 0:
                pad_axis = [0] * signal.ndim
                pad_axis[axis] = pad_size
                signal = F.pad(signal, pad_axis, "constant", pad_value)
        frames=signal.unfold(axis, frame_length, frame_step)
        return frames
    
    

if __name__ == "__main__":
    eegdata = EEGdataset(filepaths="/apdcephfs/share_1316500/qiushizhu/eegdata/split_data")
    new_files = eegdata.__getitem__(1)
    print(new_files[0].size())
    print(new_files[1].size())

    # dataloader = torch.utils.data.DataLoader(eegdata, batch_size=1, shuffle=False)
    # batch = next(iter(dataloader))
    # print(batch[0].size())
    # out = batch[0].squeeze(0).unfold(0, 640, 64)
    # print(out.size())
    
    # for i in range(eegdata.__len__()):
    #     new_file = eegdata.__getitem__(i)
    #     print(new_file[0].size())