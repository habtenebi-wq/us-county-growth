"""Analyze U.S. county population change, 2010-2019.

Questions:
  1. How evenly was U.S. population growth spread across counties?
  2. Was growth driven more by migration or by births minus deaths?
  3. How did growth differ between rural and urban counties?
  4. Which county characteristics are linked to faster growth?
  5. How does North Carolina compare, and which NC counties grew fastest?

Run from the project root, after src/clean.py:  python src/analyze.py
Writes charts to figures/ and a summary of every number to results/summary.md.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FIG = ROOT / "figures"
RES = ROOT / "results"

sns.set_theme(style="whitegrid", context="notebook")
BLUE, GRAY, RED = "#2b6cb0", "#a0aec0", "#c53030"

TRIANGLE = ["Wake", "Durham", "Orange", "Chatham", "Johnston"]
PREDICTORS = {
    "college_plus_pct": "% adults with a bachelor's degree or more",
    "age65_plus_pct": "% of residents 65 or older",
    "log_density": "Population density (log10)",
    "foreign_born_pct": "% foreign-born residents",
}

lines: list[str] = []  # collected for results/summary.md


def note(text: str = "") -> None:
    print(text)
    lines.append(text)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    counties = pd.read_csv(DATA / "counties_clean.csv", dtype={"fips": str})
    order = ["Most rural", "Rural", "Middle", "Suburban", "Most urban"]
    counties["density_group"] = pd.Categorical(counties["density_group"], order, ordered=True)
    states = pd.read_csv(DATA / "states.csv", dtype={"fips": str})
    return counties, states


# 1. How evenly was growth spread? ------------------------------------------
def growth_concentration(c: pd.DataFrame, states: pd.DataFrame) -> None:
    note("## 1. How evenly was growth spread across counties?")
    us = states[states["fips"] == "00000"].iloc[0]
    shrank = (~c["grew"]).mean()
    note(f"- The U.S. grew {us['pop_change_pct']:.1f}% from 2010 to 2019, "
         f"yet {shrank:.0%} of counties ({(~c['grew']).sum():,} of {len(c):,}) lost population.")

    gains = c.loc[c["pop_change"] > 0, "pop_change"].sort_values(ascending=False)
    top50_share = gains.head(50).sum() / c["pop_change"].sum()
    note(f"- The 50 fastest-gaining counties (by number of people added) account for "
         f"{top50_share:.0%} of the country's net population gain.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    clipped = c["pop_change_pct"].clip(-30, 60)
    ax.hist(clipped[clipped < 0], bins=np.arange(-30, 0.1, 2.5), color=RED, alpha=0.8, label="Lost population")
    ax.hist(clipped[clipped >= 0], bins=np.arange(0, 60.1, 2.5), color=BLUE, alpha=0.8, label="Gained population")
    ax.axvline(us["pop_change_pct"], color="black", ls="--", lw=1)
    ax.text(us["pop_change_pct"] + 1, ax.get_ylim()[1] * 0.9, f"U.S. overall: +{us['pop_change_pct']:.1f}%", fontsize=9)
    ax.set(title="Most counties shrank even as the U.S. grew (2010-2019)",
           xlabel="Population change, 2010-2019 (%)  [clipped at -30% and +60%]",
           ylabel="Number of counties")
    ax.legend()
    save(fig, "01_growth_distribution.png")


# 2. Migration vs natural change -------------------------------------------
def growth_drivers(c: pd.DataFrame) -> None:
    note("\n## 2. Migration or births minus deaths?")
    mig = (c["main_driver"] == "migration").mean()
    note(f"- In {mig:.0%} of counties, net migration moved population more than "
         f"natural change (births minus deaths) did.")
    deaths_exceed = (c["natural_change_pct"] < 0).mean()
    note(f"- {deaths_exceed:.0%} of counties had more deaths than births over the decade.")
    both_neg = ((c["natural_change_pct"] < 0) & (c["net_migration_pct"] < 0)).sum()
    note(f"- {both_neg:,} counties lost people both ways: more deaths than births and more people moving out than in.")

    fig, ax = plt.subplots(figsize=(7, 6))
    sub = c[c["pop_2010"] >= 1000]
    colors = np.where(sub["grew"], BLUE, RED)
    ax.scatter(sub["natural_change_pct"], sub["net_migration_pct"], s=6, c=colors, alpha=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlim=(-12, 25), ylim=(-30, 60),
           title="Where county growth came from (2010-2019)",
           xlabel="Natural change: births minus deaths (% of 2010 pop.)",
           ylabel="Net migration: moved in minus moved out (% of 2010 pop.)")
    ax.text(14, 50, "Grew", color=BLUE, fontweight="bold")
    ax.text(-11, -27, "Shrank", color=RED, fontweight="bold")
    save(fig, "02_growth_drivers.png")


# 3. Rural vs urban --------------------------------------------------------
def rural_urban(c: pd.DataFrame) -> None:
    note("\n## 3. Rural vs. urban counties")
    g = c.groupby("density_group", observed=True).agg(
        median_change=("pop_change_pct", "median"),
        share_grew=("grew", "mean"),
        counties=("fips", "count"),
    )
    for name, row in g.iterrows():
        note(f"- {name}: median change {row['median_change']:+.1f}%, "
             f"{row['share_grew']:.0%} of counties grew ({int(row['counties'])} counties)")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [RED if v < 0 else BLUE for v in g["median_change"]]
    ax.bar(g.index.astype(str), g["median_change"], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    for i, v in enumerate(g["median_change"]):
        ax.text(i, v + (0.15 if v >= 0 else -0.15), f"{v:+.1f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_ylim(g["median_change"].min() - 1, g["median_change"].max() + 1)
    ax.set(title="Median county population change by density (2010-2019)",
           xlabel="Counties grouped into fifths by 2010 population density",
           ylabel="Median change (%)")
    save(fig, "03_rural_vs_urban.png")


# 4. What is linked to faster growth? --------------------------------------
def drivers_model(c: pd.DataFrame) -> None:
    note("\n## 4. Which county traits are linked to faster growth?")
    # Very small counties swing wildly in % terms, so leave out counties under 1,000 people.
    d = c[c["pop_2010"] >= 1000].dropna(subset=list(PREDICTORS) + ["pop_change_pct"])
    note(f"- Counties used: {len(d):,} (those with at least 1,000 residents and complete data)")

    for col, label in PREDICTORS.items():
        r, p = stats.pearsonr(d[col], d["pop_change_pct"])
        note(f"- Correlation with growth, {label}: r = {r:+.2f}")

    X = StandardScaler().fit_transform(d[list(PREDICTORS)])
    y = d["pop_change_pct"].to_numpy()
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    cv_r2 = cross_val_score(LinearRegression(), X, y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2").mean()
    note(f"- A linear regression on these four traits explains {r2:.0%} of the variation in "
         f"county growth (5-fold cross-validated R^2 = {cv_r2:.2f}).")
    note("- Standardized coefficients (percentage points of growth per 1 standard deviation):")
    for (col, label), coef in sorted(zip(PREDICTORS.items(), model.coef_), key=lambda t: -abs(t[1])):
        note(f"  - {label}: {coef:+.2f}")
    note("- These are associations, not causes: for example, college-educated workers may "
         "move to places that are already growing.")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.regplot(data=d, x="college_plus_pct", y="pop_change_pct", ax=ax,
                scatter_kws={"s": 6, "alpha": 0.35, "color": GRAY},
                line_kws={"color": BLUE})
    r = d[["college_plus_pct", "pop_change_pct"]].corr().iloc[0, 1]
    ax.set(ylim=(-30, 60),
           title=f"Counties with more college graduates grew faster (r = {r:.2f})",
           xlabel="Adults with a bachelor's degree or more (%)",
           ylabel="Population change, 2010-2019 (%)")
    save(fig, "04_education_vs_growth.png")


# 5. North Carolina --------------------------------------------------------
def north_carolina(c: pd.DataFrame, states: pd.DataFrame) -> None:
    note("\n## 5. North Carolina")
    nc_state = states[states["state"] == "NC"].iloc[0]
    us = states[states["fips"] == "00000"].iloc[0]
    nc = c[c["state"] == "NC"].sort_values("pop_change_pct", ascending=False)
    note(f"- NC grew {nc_state['pop_change_pct']:.1f}% vs. {us['pop_change_pct']:.1f}% for the U.S., "
         f"but {(~nc['grew']).sum()} of its {len(nc)} counties lost population.")

    tri = nc[nc["county"].isin(TRIANGLE)]
    tri_share = tri["pop_change"].sum() / nc["pop_change"].sum()
    note(f"- Five Triangle counties ({', '.join(TRIANGLE)}) added {tri['pop_change'].sum():,.0f} people: "
         f"{tri_share:.0%} of NC's total net gain.")
    top = nc.head(3)
    note("- Fastest-growing NC counties: " +
         ", ".join(f"{r.county} ({r.pop_change_pct:+.1f}%)" for r in top.itertuples()))
    bottom = nc.tail(3).iloc[::-1]
    note("- Fastest-shrinking NC counties: " +
         ", ".join(f"{r.county} ({r.pop_change_pct:+.1f}%)" for r in bottom.itertuples()))

    nc[["county", "pop_2010", "pop_2019", "pop_change", "pop_change_pct",
        "net_migration_pct", "natural_change_pct", "college_plus_pct"]].to_csv(
        RES / "nc_counties_ranked.csv", index=False)

    show = pd.concat([nc.head(10), nc.tail(10)])
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = [BLUE if v >= 0 else RED for v in show["pop_change_pct"]]
    ax.barh(show["county"], show["pop_change_pct"], color=colors)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title="North Carolina: 10 fastest-growing and 10 fastest-shrinking counties",
           xlabel="Population change, 2010-2019 (%)", ylabel="")
    save(fig, "05_nc_counties.png")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    counties, states = load()
    note("# Results: U.S. county population change, 2010-2019")
    note(f"\nCounties analyzed: {len(counties):,}\n")
    growth_concentration(counties, states)
    growth_drivers(counties)
    rural_urban(counties)
    drivers_model(counties)
    north_carolina(counties, states)
    (RES / "summary.md").write_text("\n".join(lines) + "\n")
    print("\nSaved charts to figures/ and results to results/")


if __name__ == "__main__":
    main()
