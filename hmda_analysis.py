"""
Mortgage denial and applicant characteristics: a reproducible empirical analysis
Author: Miracle Agu

Research question
    How large is the raw difference in mortgage denial rates between Black and
    white applicants in the 1990 Boston HMDA data, how much of it is accounted for
    by observable credit and loan characteristics, and how well can those
    characteristics predict denial out of sample?

Data (public)
    HMDA cross-section from the Boston Fed study (Munnell et al., 1996, American
    Economic Review 86(1): 25-53), 2,380 applications, distributed in the R
    package AER and mirrored in the Rdatasets repository.

Run:  python hmda_analysis.py      (needs pandas, numpy, statsmodels, scikit-learn,
                                    matplotlib; internet access on first run)
"""
import os, json
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

SEED = 2027
URL = ("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/"
       "master/csv/AER/HMDA.csv")
OUT = "outputs"; os.makedirs(OUT, exist_ok=True)

# 1. Load and clean --------------------------------------------------------
raw = pd.read_csv(URL).drop(columns="rownames")
assert raw.shape == (2380, 14)
print("Missing values per column:\n", raw.isna().sum()[raw.isna().sum() > 0])
df = raw.dropna().copy()
for c in ["deny", "phist", "selfemp", "insurance", "condomin", "afam", "single", "hschool"]:
    df[c] = (df[c] == "yes").astype(int)
df["chist"] = df["chist"].astype(int)   # consumer credit history score, 1-6
df["mhist"] = df["mhist"].astype(int)   # mortgage payment history score, 1-4
print("Rows used:", len(df))
# Data checks: ratios should be positive and mostly below 1.
print("Share with pirat > 1:", round((df.pirat > 1).mean(), 4))

# 2. Descriptives ----------------------------------------------------------
desc = df.groupby("afam")[["deny", "pirat", "hirat", "lvrat", "chist", "mhist",
                           "phist", "unemp", "selfemp", "hschool"]].mean().T
desc.columns = ["White", "Black"]
n_by = df.afam.value_counts().sort_index()
print(desc.round(3)); print("N white, Black:", n_by.tolist())
desc.round(3).to_csv(f"{OUT}/descriptives.csv")

# 3. Regression: raw gap versus conditional gap ----------------------------
controls = ("pirat + hirat + lvrat + C(chist) + C(mhist) + phist + unemp + "
            "selfemp + insurance + condomin + single + hschool")
m_raw  = smf.ols("deny ~ afam", df).fit(cov_type="HC1")
m_ctrl = smf.ols(f"deny ~ afam + {controls}", df).fit(cov_type="HC1")
m_logit = smf.logit(f"deny ~ afam + {controls}", df).fit(disp=0, cov_type="HC1")
# 'insurance' (denied mortgage insurance) is decided in the same lending process,
# so it may be a post-treatment ("bad") control. Check the gap without it.
ctrl_noins = controls.replace(" + insurance", "")
m_noins = smf.ols(f"deny ~ afam + {ctrl_noins}", df).fit(cov_type="HC1")
ame = m_logit.get_margeff(at="overall").summary_frame().loc["afam"]

rows = [
    ("OLS, no controls (linear probability)", m_raw.params["afam"], m_raw.bse["afam"]),
    ("OLS, with controls", m_ctrl.params["afam"], m_ctrl.bse["afam"]),
    ("OLS, with controls, excluding insurance denial", m_noins.params["afam"], m_noins.bse["afam"]),
    ("Logit average marginal effect, with controls", ame["dy/dx"], ame["Std. Err."]),
]
res = pd.DataFrame(rows, columns=["Specification", "Black minus white denial gap", "Robust SE"])
res["95% CI low"]  = res.iloc[:, 1] - 1.96 * res.iloc[:, 2]
res["95% CI high"] = res.iloc[:, 1] + 1.96 * res.iloc[:, 2]
print(res.round(3).to_string(index=False)); res.round(6).to_csv(f"{OUT}/gap_estimates.csv", index=False)

# 4. Sensitivity: unobserved confounding (Oster-style bounding is not attempted).
#    Simple check: how does the gap move as controls are added in blocks?
blocks = {
    "None": "",
    "+ payment/housing ratios, loan-to-value": "pirat + hirat + lvrat",
    "+ credit history": "pirat + hirat + lvrat + C(chist) + C(mhist) + phist",
    "+ all controls": controls,
}
steps = []
for name, rhs in blocks.items():
    f = "deny ~ afam" + (" + " + rhs if rhs else "")
    mm = smf.ols(f, df).fit(cov_type="HC1")
    steps.append((name, mm.params["afam"], mm.bse["afam"]))
steps = pd.DataFrame(steps, columns=["Controls added", "Gap", "SE"])
print(steps.round(3).to_string(index=False)); steps.round(6).to_csv(f"{OUT}/gap_by_controls.csv", index=False)

# 5. Out-of-sample prediction (5-fold stratified CV) ------------------------
X = pd.get_dummies(df.drop(columns="deny"), columns=["chist", "mhist"], drop_first=True).astype(float)
y = df["deny"].values
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
models = {
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Gradient boosting": GradientBoostingClassifier(random_state=SEED, max_depth=2,
                                                    n_estimators=150, learning_rate=0.05),
}
pred_rows, probs = [], {}
for name, mdl in models.items():
    p = cross_val_predict(mdl, X, y, cv=cv, method="predict_proba")[:, 1]
    probs[name] = p
    pred_rows.append((name, roc_auc_score(y, p), brier_score_loss(y, p)))
base = brier_score_loss(y, np.full_like(y, y.mean(), dtype=float))
pred_rows.append(("Base rate only", 0.5, base))
pred = pd.DataFrame(pred_rows, columns=["Model", "CV AUC", "CV Brier score"])
print(pred.round(3).to_string(index=False)); pred.round(6).to_csv(f"{OUT}/prediction.csv", index=False)

# Does the prediction error differ by group? (a basic fairness diagnostic)
p = probs["Logistic regression"]; thr = np.quantile(p, 1 - y.mean())
flag = (p >= thr).astype(int)
diag = df.assign(flag=flag).groupby("afam").apply(
    lambda g: pd.Series({"n": len(g), "actual denial rate": g.deny.mean(),
                         "flagged rate": g.flag.mean(),
                         "mean predicted prob": p[g.index.map(df.index.get_loc)].mean()}),
    include_groups=False)
diag.index = ["White", "Black"]
print(diag.round(3)); diag.round(6).to_csv(f"{OUT}/group_diagnostics.csv")

# 6. Figures ---------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
bins = pd.cut(df.pirat, [0, .2, .3, .35, .4, .5, 3])
t = df.groupby([bins, "afam"], observed=True).deny.mean().unstack()
t.columns = ["White", "Black"]
t.plot(marker="o", ax=ax[0]); ax[0].set_xticks(range(len(t))); ax[0].set_xticklabels([str(i) for i in t.index], rotation=30, fontsize=7)
ax[0].set_ylabel("Denial rate"); ax[0].set_xlabel("Payments-to-income ratio bin")
ax[0].set_title("Denial rate by payment burden and group", fontsize=9)
ax[1].errorbar(range(len(steps)), steps.Gap, yerr=1.96 * steps.SE, fmt="o", capsize=4)
ax[1].axhline(0, color="grey", lw=.8); ax[1].set_xticks(range(len(steps)))
ax[1].set_xticklabels(["None", "+ ratios", "+ credit\nhistory", "+ all"], fontsize=8)
ax[1].set_ylabel("Black minus white denial gap"); ax[1].set_title("Gap as controls are added (95% CI)", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/figure1.png", dpi=200)

json.dump({"n": int(len(df)), "seed": SEED}, open(f"{OUT}/meta.json", "w"))
