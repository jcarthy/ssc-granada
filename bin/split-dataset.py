import numpy as np
import argparse
import pathlib
from sklearn.model_selection import train_test_split


def load_raw_data(dir):
    data = np.load(f"{dir}/data.npy")
    labels = np.load(f"{dir}/labels.npy")
    return data, labels


def save_features(
    train_data,
    val_data,
    test_data,
    tag,
    dir,
):
    feature_dir = pathlib.Path(f"{dir}/{tag}")
    feature_dir.mkdir(parents=True, exist_ok=True)
    np.save(feature_dir / "train_data.npy", train_data)
    np.save(feature_dir / "val_data.npy", val_data)
    np.save(feature_dir / "test_data.npy", test_data)


def save_labels(train_labels, val_labels, test_labels, dir):
    np.save(dir / "train_labels.npy", train_labels)
    np.save(dir / "val_labels.npy", val_labels)
    np.save(dir / "test_labels.npy", test_labels)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to dataset",
        default="datasets/llaima",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dir = pathlib.Path(args.dataset_path)
    raw_data, labels = load_raw_data(dir)

    # Remove the mean from each sample ( no need to split as sample-wise operation )
    raw_data = raw_data - np.mean(raw_data, axis=1, keepdims=True)
    raw_data.mean(axis=1)

    # Split data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        raw_data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    # Split into test and validation sets
    test_data, val_data, test_labels, val_labels = train_test_split(
        test_data, test_labels, test_size=0.5, random_state=42, stratify=test_labels
    )

    # Will be the same across all feature sets
    save_labels(train_labels, val_labels, test_labels, dir)

    print(train_data.shape)
    print(val_data.shape)
    print(test_data.shape)

    save_features(train_data, val_data, test_data, "raw", dir)


if __name__ == "__main__":
    main()
