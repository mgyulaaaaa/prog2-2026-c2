import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

DF_URL = "http://tmp-borza-public-cyx.s3.amazonaws.com/imdb-comp.csv.gz"

RUNS_DIR = Path("runs/run-logs")
SOLUTIONS_DIR = Path("solutions")
TEST_DATA_PATH = Path("full-df.csv.gz")

GENRE_COLS = [
    "drama",
    "history",
    "musical",
    "comedy",
    "music",
    "romance",
    "family",
    "adventure",
    "fantasy",
    "mystery",
    "thriller",
    "sci_fi",
    "biography",
    "crime",
    "horror",
    "documentary",
    "western",
    "action",
    "war",
    "animation",
    "sport",
    "film_noir",
]

output_cols = ["year", "title", "imdb_id"]


def load_test_df() -> pd.DataFrame:
    if TEST_DATA_PATH.exists():
        return pd.read_csv(TEST_DATA_PATH)
    df = pd.read_csv(DF_URL)
    df.to_csv(TEST_DATA_PATH, index=False)
    return df


def main(
    solution: str,
    in_n: int = 1_000,
    q_n: int = 10,
    comparison: str = "",
    seed: int = 742,
) -> list[str] | None:
    s_path = SOLUTIONS_DIR / solution
    assert s_path.exists(), f"solution not found: {s_path}"

    rng = np.random.RandomState(seed)
    in_p, q_p, out_p = map(s_path.joinpath, ["input.csv", "query.csv", "out.csv"])
    test_df = load_test_df()

    def call(comm: str) -> float:
        start = time.time()
        subprocess.call(["make", comm], cwd=s_path.as_posix())
        return time.time() - start

    def dump_input() -> None:
        test_df.sample(in_n, random_state=rng).to_csv(in_p, index=False)

    def dump_query() -> None:
        input_df = pd.read_csv(in_p)
        noise_scale = input_df[["x", "y"]].std().mean() * 2
        queries: list[dict] = []
        while len(queries) < q_n:
            genre = str(rng.choice(GENRE_COLS))
            genre_subset = input_df[input_df[genre]]
            if genre_subset.empty:
                continue
            anchor = genre_subset.sample(1, random_state=rng).iloc[0]
            min_year = int(anchor["year"]) - int(rng.randint(1, 30))
            max_year = int(anchor["year"]) + int(rng.randint(1, 30))
            subset = genre_subset[
                (genre_subset["year"] >= min_year) & (genre_subset["year"] <= max_year)
            ]
            if subset.empty:
                continue
            point = subset[["x", "y"]].sample(1, random_state=rng).iloc[0]
            queries.append(
                {
                    "genre": genre,
                    "min_year": min_year,
                    "max_year": max_year,
                    "x": float(point["x"]) + rng.normal(0, noise_scale),
                    "y": float(point["y"]) + rng.normal(0, noise_scale),
                }
            )
        pd.DataFrame(queries).to_csv(q_p, index=False)

    logs = [f"inputs: {in_n}", f"queries: {q_n}"]
    for comm, prep in [
        ("setup", lambda: None),
        ("preproc", dump_input),
        ("compute", dump_query),
    ]:
        prep()
        logs.append(f"{comm}: {call(comm):.6f}")
    in_p.unlink()
    q_p.unlink()
    call("cleanup")

    try:
        out_df = pd.read_csv(out_p)
    except Exception:
        print(f"ERROR: could not read {out_p}")
        return None

    assert out_df.columns.tolist() == output_cols, (
        f"columns {out_df.columns.tolist()} != {output_cols}"
    )
    assert out_df.shape[0] == q_n, f"output length {out_df.shape[0]} != {q_n}"

    if comparison:
        main(comparison, in_n, q_n, seed=seed)
        comp_df = pd.read_csv(SOLUTIONS_DIR / comparison / out_p.name)
        misses = (comp_df != out_df).any(axis=1)
        if misses.any():
            print("Mismatches found:")
            print(f"{solution}:\n{out_df[misses]}")
            print(f"{comparison}:\n{comp_df[misses]}")

    logstr = "\n".join(logs)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{time.time():.6f}-{solution}").write_text(logstr)
    print(f"\nsuccess! solution: {solution}")
    print(logstr)
    return logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single challenge-2 solution")
    parser.add_argument("solution")
    parser.add_argument("--compare", default="", metavar="SOLUTION")
    parser.add_argument("--in-n", type=int, default=1_000)
    parser.add_argument("--q-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=742)
    args = parser.parse_args()
    main(
        args.solution,
        in_n=args.in_n,
        q_n=args.q_n,
        comparison=args.compare,
        seed=args.seed,
    )
