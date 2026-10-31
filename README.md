# hmda-mortgage-denial-analysis

# Mortgage denial and applicant characteristics

Reproducible Python analysis of the public Boston HMDA mortgage data
(Munnell et al., 1996, American Economic Review).

Question: how large is the raw gap in denial rates between Black and white
applicants, how much do observable credit and loan characteristics explain,
and how well can they predict denial out of sample?

## Run
pip install pandas numpy statsmodels scikit-learn matplotlib
python hmda_analysis.py

The script downloads the data from the public Rdatasets repository, uses a
fixed random seed (2027), and writes tables and a figure to an outputs/ folder.

## Limitations
Observational data from one city in 1990. The results describe conditional
differences and do not show discrimination.
