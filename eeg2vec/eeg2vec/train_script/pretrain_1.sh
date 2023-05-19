
#/bin/bash
# pip install tensorboardX
# pip install soundfile
# pip install editdistance
# cd /apdcephfs/share_1316500/qiushizhu/eegdata/eeg2vec/fairseq && pip install -e .

# export WORLD_SIZE=$1
# export HOST_GPU_NUM=$1
# export NCCL_SOCKET_IFNAME=eth1
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export RANK=$INDEX
# export MASTER_ADDR=$CHIEF_IP
# export MASTER_PORT=10000
export HYDRA_FULL_ERROR=1

ngpu=$1
updatefreq=$2
model_path=/apdcephfs/share_1316500/qiushizhu/eegdata/eeg2vec/results/pretrain_1_ngpu${ngpu}_updatefreq${updatefreq}

python3 /apdcephfs/share_1316500/qiushizhu/eegdata/eeg2vec/fairseq/fairseq_cli/hydra_train.py \
        --config-dir /apdcephfs/share_1316500/qiushizhu/eegdata/eeg2vec/eeg2vec/config/ \
        --config-name base \
        common.user_dir=/apdcephfs/share_1316500/qiushizhu/eegdata/eeg2vec/eeg2vec \
        checkpoint.save_dir=${model_path} \
        hydra.run.dir=${model_path} \
        task.data=/apdcephfs/share_1316500/qiushizhu/eegdata/eeg2vec/data \
        distributed_training.distributed_world_size=${ngpu} \
        optimization.update_freq=[${updatefreq}] \
        optimization.lr=[5e-4]  \
        dataset.train_subset="train" \
        dataset.valid_subset="valid" \
        task.max_sample_size=100000 \
        task.min_sample_size=20 \




