"""Clean the raw county demographic data.

Reads data/raw/county_data_raw.csv and writes:
  - data/processed/counties_clean.csv  (one row per county)
  - data/processed/states.csv          (one row per state, plus the US total)

Run from the project root:  python src/clean.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "county_data_raw.csv"
OUT = ROOT / "data" / "processed"

# The columns this project actually uses, renamed to shorter, clearer names.
COLUMNS = {
    "FIPS": "fips",
    "State": "state",
    "County": "county",
    # July 2010 estimate: the base the file's own % change columns are measured from.
    "TotalPopEst2010": "pop_2010",
    "TotalPopEst2019": "pop_2019",
    "PopChangeRate1019": "pop_change_pct",
    "NetMigrationRate1019": "net_migration_pct",
    "NaturalChangeRate1019": "natural_change_pct",
    "Net_International_Migration_Rate_2010_2019": "intl_migration_pct",
    "PopDensity2010": "density_2010",
    "Ed5CollegePlusPct": "college_plus_pct",
    "Age65AndOlderPct2010": "age65_plus_pct",
    "ForeignBornPct": "foreign_born_pct",
    "OwnHomePct": "own_home_pct",
}


def load_raw(path: Path = RAW) -> pd.DataFrame:
    # FIPS codes have leading zeros (e.g. "01001"), so keep them as text.
    return pd.read_csv(path, dtype={"FIPS": str})


def split_levels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The raw file mixes the US total, state totals and counties.

    State and US rows have FIPS codes ending in "000".
    """
    df = df.copy()
    df["FIPS"] = df["FIPS"].str.zfill(5)
    is_summary = df["FIPS"].str.endswith("000")
    return df[~is_summary], df[is_summary]


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df[list(COLUMNS)].rename(columns=COLUMNS)
    df["state"] = df["state"].str.strip()
    df["county"] = df["county"].str.strip()
    numeric = [c for c in df.columns if c not in ("fips", "state", "county")]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df


def add_features(counties: pd.DataFrame) -> pd.DataFrame:
    c = counties.copy()
    c["pop_change"] = c["pop_2019"] - c["pop_2010"]
    c["grew"] = c["pop_change"] > 0
    # Which force mattered more over 2010-2019: births minus deaths,
    # or people moving in minus people moving out?
    c["main_driver"] = np.where(
        c["net_migration_pct"].abs() >= c["natural_change_pct"].abs(),
        "migration",
        "natural change",
    )
    # Density spans 0.04 to 70,000+ people per square mile, so use a log scale
    # and split counties into five equal-sized groups from most rural to most urban.
    c["log_density"] = np.log10(c["density_2010"])
    c["density_group"] = pd.qcut(
        c["density_2010"],
        5,
        labels=["Most rural", "Rural", "Middle", "Suburban", "Most urban"],
    )
    return c


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    counties_raw, summary_raw = split_levels(raw)

    counties = tidy(counties_raw)
    before = len(counties)
    key_cols = ["pop_2010", "pop_2019", "pop_change_pct", "net_migration_pct",
                "natural_change_pct", "density_2010", "college_plus_pct"]
    counties = counties.dropna(subset=key_cols)
    counties = counties[counties["density_2010"] > 0]
    dropped = before - len(counties)

    counties = add_features(counties)
    states = tidy(summary_raw)

    counties.to_csv(OUT / "counties_clean.csv", index=False)
    states.to_csv(OUT / "states.csv", index=False)
    print(f"Counties kept: {len(counties):,} (dropped {dropped} with missing key values)")
    print(f"State/US summary rows: {len(states)}")


if __name__ == "__main__":
    main()
