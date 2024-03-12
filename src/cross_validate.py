import tensorflow as tf
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold

from dataloader import SpokenDataLoader
from baseline_models import get_rnn_model, get_cnn_2d_model, get_cnn_1d_model, get_cnn_effnet_model, get_transformer_model, get_tiny_cnn_model

PATHS ={"German": Path("../german_numbers"), "French": Path("../french_numbers"), "English": Path("../english_numbers"), 
        "Mandarin": Path("../mandarin_numbers")}


def get_df_metadata():
    # create a df with all the file paths, language, speaker and number
    df = pd.DataFrame(columns=["file_path", "language", "speaker", "number"])
    for language in PATHS.keys():
        for speaker in PATHS[language].iterdir():
            if not speaker.is_dir():
                 continue
            for file in speaker.iterdir():
                    df = pd.concat([df, pd.DataFrame({"file_path": [file], "language": [language], "speaker": [speaker.stem], "number": [file.stem]})])

    return df


def run_cross_validation(model_type, epochs, label_type="language"):
    df = get_df_metadata()
    speakers = df["speaker"].unique()

    # create k folds
    kf = KFold(n_splits=8, shuffle=True, random_state=42)
    idx = 1
    for train_index, test_index in kf.split(speakers):
        train_speakers, test_speakers = speakers[train_index], speakers[test_index]

        df_splits = df.copy()
        df_splits["split"] = ""
        df_splits.loc[df["speaker"].isin(train_speakers), "split"] = "train"
        df_splits.loc[df["speaker"].isin(test_speakers), "split"] = "test"

        # create a dataloader
        dl = SpokenDataLoader(df_splits, batch_size=32)

        # determine the number of classes
        if label_type == "language":
            nb_classes = 4
        elif label_type == "number":
            nb_classes = 100

        # load the model and lr scheduler
        if model_type == "rnn":
            data = dl.load_spectrogram(label_type=label_type)
            model, lr_decay = get_rnn_model(nb_classes)
        elif model_type == "cnn_effnet":
            data = dl.load_spectrogram(label_type=label_type, channel_dim=3)
            model, lr_decay = get_cnn_effnet_model(nb_classes)
        elif model_type == "cnn_2d":
            data = dl.load_spectrogram(label_type=label_type, channel_dim=1)
            model, lr_decay = get_cnn_2d_model(nb_classes)
        elif model_type == "cnn_1d":
            data = dl.load_waveform(label_type=label_type)
            model, lr_decay = get_cnn_1d_model(nb_classes)
        elif model_type == "transformer":
            data = dl.load_spectrogram(label_type=label_type, channel_dim="transformer")
            model, lr_decay = get_transformer_model(nb_classes)
        elif model_type == "tiny_cnn":
            data = dl.load_waveform(label_type=label_type)
            model, lr_decay = get_tiny_cnn_model(nb_classes)
        else:
            raise ValueError("Model type not supported")

        # create checkpoint callback
        save_path = f"../results/cross_validation_{model_type}_{label_type}/model_{idx}/"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path + "model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        # train the model
        history = model.fit(data["train"], validation_data=data["test"], epochs=epochs, callbacks=[lr_decay, checkpoint])

        # save the history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(save_path + "history.csv")

        # load the best model
        model.load_weights(save_path + "model.h5")

        # evaluate the model
        results = model.evaluate(data["test"])

        # save the results
        results_df = pd.DataFrame({"loss": [results[0]], "accuracy": [results[1]]})
        results_df.to_csv(save_path + "results.csv")

        idx += 1

if __name__ == "__main__":
    # limit gpu memory usage
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="", help="The model to use for cross validation")
    parser.add_argument("--epochs", type=int, default=75, help="Number of epochs to train the model")
    parser.add_argument("--label_type", type=str, default="language", help="The type of label to use for training")
    args = parser.parse_args()
    
    run_cross_validation(args.model, args.epochs, args.label_type)