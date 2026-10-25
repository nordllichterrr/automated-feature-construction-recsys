from pathlib import Path
import zipfile

import pandas as pd


DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = Path("data/raw")
ARCHIVE_PATH = DATA_DIR / "ml-1m.zip"
RATINGS_PATH = DATA_DIR / "ratings.dat"


def load_movielens_1m(download=True):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not RATINGS_PATH.exists():
        if not download:
            raise FileNotFoundError(
                "Файл рейтингов не найден: {}".format(RATINGS_PATH)
            )

        from urllib.request import urlretrieve

        urlretrieve(DATA_URL, ARCHIVE_PATH)

        with zipfile.ZipFile(ARCHIVE_PATH) as archive:
            archive.extract("ml-1m/ratings.dat", DATA_DIR)

        extracted_path = DATA_DIR / "ml-1m" / "ratings.dat"
        extracted_path.replace(RATINGS_PATH)
        (DATA_DIR / "ml-1m").rmdir()

    ratings = pd.read_csv(
        RATINGS_PATH,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={
            "user_id": "int32",
            "item_id": "int32",
            "rating": "int8",
            "timestamp": "int64",
        },
    )

    ratings["timestamp"] = pd.to_datetime(
        ratings["timestamp"], unit="s", utc=True
    )
    ratings["label"] = (ratings["rating"] >= 4).astype("int8")

    if ratings.empty:
        raise ValueError("Датасет MovieLens оказался пустым")

    if not ratings["rating"].between(1, 5).all():
        raise ValueError("Обнаружены оценки вне допустимого диапазона [1, 5]")

    return ratings.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
