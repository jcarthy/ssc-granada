from scipy import signal
from scipy.stats import kurtosis, skew
import numpy as np
import pathlib
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="FFT a dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset",
        default="datasets/llaima/raw",
    )
    parser.add_argument(
        "--stats",
        nargs="*",
        type=str,
        help="Stats to compute",
        default=["kurtosis", "mean", "skew", "std"],
    )
    parser.add_argument("--nfft", type=int, help="FFT size", default=1024)
    parser.add_argument(
        "--n-splits", type=int, help="Number of splits to generate fft on", default=1
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = pathlib.Path(args.dataset)
    output_dataset_fft = dataset.parent / (dataset.stem + f"-fft-split-{args.n_splits}")
    output_dataset_stats = dataset.parent / (
        dataset.stem + f"-stats-splits-{args.n_splits}"
    )
    print(dataset)
    print(output_dataset_fft)
    print(output_dataset_stats)
    stats = args.stats

    for path in dataset.glob("**/*.npy"):
        print("Processing", path)
        data = np.load(path)
        if len(data.shape) == 1:
            raise ValueError("Data is 1D - Check Inputs are correct")

        fft_data = []
        stats_data = []

        for i, trace in enumerate(data):
            diffs = np.diff(trace)
            last_change = np.where(diffs.nonzero())[1][-1] + 2
            if last_change < 5995:
                trace_unpadded = trace[:last_change]
            else:
                trace_unpadded = trace

            fft_splits = []
            stats_splits = []
            for j, data_split in enumerate(
                np.array_split(trace_unpadded, args.n_splits)
            ):
                data_split = data_split - np.mean(data_split)  # Remove DC offset
                if np.max(np.abs(data_split)) == 0:
                    print("Zero trace", i, j)
                    continue
                else:
                    data_split = data_split / np.max(np.abs(data_split))  # Normalise

                split_features = []
                for stat in stats:
                    if stat == "kurtosis":
                        split_features.append(kurtosis(data_split, axis=-1))
                    elif stat == "mean":
                        split_features.append(np.mean(data_split, axis=-1))
                    elif stat == "skew":
                        split_features.append(skew(data_split, axis=-1))
                    elif stat == "std":
                        split_features.append(np.std(data_split, axis=-1))
                    else:
                        raise NotImplementedError(
                            "Statistic not currently implemented: ", stat
                        )
                split_features = np.vstack(split_features).T

                data_split = np.fft.rfft(data_split, n=args.nfft)
                data_split = np.abs(data_split)
                data_split = data_split / data_split.max()

                fft_splits.append(data_split)
                stats_splits.append(split_features)
            fft_data.append(fft_splits)
            stats_data.append(stats_splits)

        features = np.array(stats_data).squeeze()
        fp = output_dataset_stats / f"{path.stem}.npy"
        try:
            np.save(fp, features)
        except FileNotFoundError:
            fp.parent.mkdir(parents=True)
            np.save(fp, features)

        data = np.array(fft_data).squeeze()
        fp = output_dataset_fft / path.name
        fp = output_dataset_fft / f"{path.stem}.npy"
        try:
            np.save(fp, data)
        except FileNotFoundError:
            fp.parent.mkdir(parents=True)
            np.save(fp, data)


if __name__ == "__main__":
    main()
