"""Example experiment for the VLAAI model."""
import glob
import json
import logging
import os
import tensorflow as tf

from vlaai3_drop import vlaai, pearson_loss, pearson_metric
from dataset_generator import RegressionDataGenerator, create_tf_dataset


def evaluate_model(model, test_dict):
    """Evaluate a model.

    Parameters
    ----------
    model: tf.keras.Model
        Model to evaluate.
    test_dict: dict
        Mapping between a subject and a tf.data.Dataset containing the test
        set for the subject.

    Returns
    -------
    dict
        Mapping between a subject and the loss/evaluation score on the test set
    """
    evaluation = {}
    for subject, ds_test in test_dict.items():
        logging.info(f"Scores for subject {subject}:")
        print("subject",subject)
        results = model.evaluate(ds_test, verbose=2)
        metrics = model.metrics_names
        evaluation[subject] = dict(zip(metrics, results))
    return evaluation


if __name__ == "__main__":
    # Parameters
    # Length of the decision window
    window_length = 10 * 64  # 10 seconds
    # Hop length between two consecutive decision windows
    hop_length = 64
    epochs = 100
    patience = 10
    batch_size = 64
    only_evaluate = False
    training_log_filename = "training_log.csv"
    results_filename = 'eval.json'


    # Get the path to the config gile
    experiments_folder = os.path.dirname(__file__)
    task_folder = os.path.dirname(experiments_folder)
    config_path = "/mnt/default/v-qiushizhu/data/challengeeeg/challenge2023/task2_regression/util/config.json"

    # Load the config
    with open(config_path) as fp:
        config = json.load(fp)

    # Provide the path of the dataset
    # which is split already to train, val, test

    data_folder = os.path.join(config["dataset_folder"], config["split_folder"])
    stimulus_features = ["envelope"]
    features = ["eeg"] + stimulus_features

    # Create a directory to store (intermediate) results
    results_folder = os.path.join(experiments_folder, "results_vlaai15")
    os.makedirs(results_folder, exist_ok=True)
    json_log = open(results_folder + '/loss_log.json', mode='wt', buffering=1)

    # create the model
    model = vlaai()
    model.compile(tf.keras.optimizers.Adam(learning_rate=0.00005), loss=pearson_loss, metrics=[pearson_metric])
    model_path = os.path.join(results_folder, "model.h5")

    if only_evaluate:
        pretrained_model_name="37_model.h5"
        pretrained_model_ckpt="/data/zhu/eeg/auditory-eeg-challenge-2023-code/task2_regression_upload/experiments7/results_vlaai15/" + pretrained_model_name
        model = vlaai()
        model.load_weights(pretrained_model_ckpt)
        model.compile(tf.keras.optimizers.Adam(learning_rate=0.00005), loss=pearson_loss, metrics=[pearson_metric])
        results_filename = 'eval_{}.json'.format(pretrained_model_name)
    
    else:

        train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        # Create list of numpy array files
        train_generator = RegressionDataGenerator(train_files, window_length)
        dataset_train = create_tf_dataset(train_generator, window_length, None, hop_length, batch_size)

        # Create the generator for the validation set
        val_files = [x for x in glob.glob(os.path.join(data_folder, "val_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        val_generator = RegressionDataGenerator(val_files, window_length)
        dataset_val = create_tf_dataset(val_generator, window_length, None, hop_length, batch_size)

        # Train the model
        model.fit(
            dataset_train,
            epochs=epochs,
            validation_data=dataset_val,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(results_folder + "/{epoch:02d}_model.h5", monitor='val_accuracy', save_best_only=False, period=1),
                tf.keras.callbacks.CSVLogger(os.path.join(results_folder, training_log_filename)),
                tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
                tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: json_log.write(json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'), on_train_end=lambda logs: json_log.close()),
            ],
        )

    # Evaluate the model on test set
    # Create a dataset generator for each test subject
    test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    # Get all different subjects from the test set
    subjects = list(set([os.path.basename(x).split("_-_")[1] for x in test_files]))
    datasets_test = {}
    # Create a generator for each subject
    for sub in subjects:
        files_test_sub = [f for f in test_files if sub in os.path.basename(f)]
        test_generator = RegressionDataGenerator(files_test_sub, window_length)
        datasets_test[sub] = create_tf_dataset(test_generator, window_length, None, hop_length, 1)

    # Evaluate the model
    # print("here1")
    # print(datasets_test)
    evaluation = evaluate_model(model, datasets_test)

    # We can save our results in a json encoded file
    results_path = os.path.join(results_folder, results_filename)
    with open(results_path, "w") as fp:
        json.dump(evaluation, fp)
    logging.info(f"Results saved at {results_path}")
