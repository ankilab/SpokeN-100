import librosa
import numpy as np
from pathlib import Path
import pandas as pd
from multiprocessing import get_context
import multiprocessing
from tqdm import tqdm


from helper import get_audio_length, calculate_f0_frequency

PATHS ={"German": Path("../german_numbers"), "French": Path("../french_numbers"), "English": Path("../english_numbers"), 
        "Mandarin": Path("../mandarin_numbers")}

#################### DATA GENERATION ####################
def generate_length_data():
    df = pd.DataFrame(columns=["Language", "Number", "Length"])
    for idx, key in enumerate(PATHS.keys()):
        lengths = []
        for folder in PATHS[key].iterdir():
            if folder.is_dir():
                for file in folder.iterdir():
                    if file.is_file():
                        length = get_audio_length(file)
                        lengths.append((int(file.stem), length))

        df = pd.concat([df, pd.DataFrame(lengths, columns=["Number", "Length"]).assign(Language=key)], ignore_index=True)

    df = df.sort_values(by=["Language", "Number"])
    df.to_csv("../results/audio_lengths.csv", index=False)


def generate_f0_data():
    df = pd.DataFrame(columns=["Language", "Number", "Frequency", "Speaker"])
    for idx, key in enumerate(PATHS.keys()):
        frequencies = []
        folders = [folder for folder in PATHS[key].iterdir() if folder.is_dir()]

        cpu_count = multiprocessing.cpu_count()
        with get_context("spawn").Pool(cpu_count) as pool:
            results = list(tqdm(pool.imap(calculate_f0_frequency, folders), total=len(folders), desc=f"Calculating {key} frequencies"))

        for result in results:
            frequencies.extend(result)
        
        print("Language finished")

        df = pd.concat([df, pd.DataFrame(frequencies, columns=["Speaker", "Number", "Frequency"]).assign(Language=key)], ignore_index=True)

    df = df.sort_values(by=["Language", "Number"])
    df.to_csv("../results/audio_frequencies.csv", index=False)


if __name__ == "__main__":
    generate_length_data()
    generate_f0_data()