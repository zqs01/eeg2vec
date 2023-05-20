"""
Sample code to generate labels for test dataset of
match-mismatch task. The requested format for submitting the labels is
as follows:
for each subject a json file containing a python dictionary in the
format of  ==> {'sample_id': prediction, ... }.

"""

import os
import glob
import json
import numpy as np
import envelope
from dilated_convolutional_model import dilation_model


def create_test_samples(eeg_path, envelope_dir):
    with open(eeg_path, 'r') as f:
        sub = json.load(f)
    eeg_data = []
    spch1_data = []
    spch2_data = []
    id_list = []
    for key, sample in sub.items():
        eeg_data.append(sample[0])

        spch1_path = os.path.join(envelope_dir, sample[1])
        spch2_path = os.path.join(envelope_dir, sample[2])
        envelope1 = np.load(spch1_path)
        env1 = envelope1['mel']
        # print("env1", env1.shape)
        envelope2 = np.load(spch2_path)
        env2 = envelope2['mel']

        spch1_data.append(env1)
        spch2_data.append(env2)
        id_list.append(key)
    eeg = np.array(eeg_data)

    padding_zero = np.zeros((np.array(spch1_data).shape[0], 2, 28))
    spch1 = np.concatenate((np.array(spch1_data), padding_zero), axis=1)
    spch2 = np.concatenate((np.array(spch2_data), padding_zero), axis=1)
    return (eeg, spch1, spch2), id_list


def get_label(pred):
    if pred >= 0.5:
        label = 1
    else:
        label = 0
    return label

if __name__ == '__main__':

    # window_length = 3*64
    window_length = 192

    # Root dataset directory containing test set
    # Change the path to the downloaded test dataset dir
    dataset_dir = '/disk2/qszhu/data/eeg/data/task1_test_set/'

    # Path to your pretrained model
    pretrained_model_name = "model.h5"
    pretrained_model_path = "/disk2/qszhu/data/eeg/auditory-eeg-challenge-2023-code/task1_match_mismatch_upload/experiments47a/results/"
    pretrained_model = pretrained_model_path + pretrained_model_name

    # Calculate envelope of the speech files (only if the envelope directory does not exist)
    stimuli_dir = os.path.join(dataset_dir, 'stimuli_segments')
    envelope_dir = os.path.join(dataset_dir, 'preprocessed_stimuli_folder')
    # Create envelope of segments if it has not already been created
    # if not os.path.isdir(envelope_dir):
    #     os.makedirs(envelope_dir, exist_ok=True)
    #     for stimulus_seg in glob.glob(os.path.join(stimuli_dir, '*.npy')):
    #         base_name = os.path.basename(stimulus_seg).split('.')[0]
            # if not os.path.exists(os.path.join(envelope_dir, base_name + '.npz')):
            #     env = envelope.calculate_envelope(stimulus_seg)
            #     target_path = os.path.join(envelope_dir, base_name + '.npz')
            #     np.savez(target_path, envelope=env)


    # Define and load the pretrained model
    model = dilation_model(time_window=window_length, eeg_input_dimension=64, env_input_dimension=28)
    model.load_weights(pretrained_model)

    test_data = glob.glob(os.path.join(dataset_dir, 'sub*.json'))
    # print("test_data", test_data)
    for sub_path in test_data:
        subject = os.path.basename(sub_path).split('.')[0]
        # print("here")
        sub_dataset, id_list = create_test_samples(sub_path, os.path.join(dataset_dir, 'preprocessed_stimuli_folder'))
        # Normalize data
        subject_data = []
        for item in sub_dataset:
            item_mean = np.expand_dims(np.mean(item, axis=1), axis=1)
            item_std = np.expand_dims(np.std(item, axis=1), axis=1)
            subject_data.append((item - item_mean) / item_std)
        sub_dataset = tuple(subject_data)
        # print("sub_dataset0", sub_dataset[0])
        # print("sub_dataset1", sub_dataset[1])
        # print("sub_dataset2", sub_dataset[2])

        predictions = model.predict(sub_dataset)
        predictions = list(np.squeeze(predictions))
        predictions = map(get_label, predictions)
        sub = dict(zip(id_list, predictions))

        prediction_dir = pretrained_model_path + 'predictions'
        os.makedirs(prediction_dir, exist_ok=True)
        with open(os.path.join(prediction_dir, subject + '.json'), 'w') as f:
            json.dump(sub, f)




