# GAN-LMNN-FARCI-Plus
# Does Hybridizing GAN–LMNN Preprocessing with Choquet-Based Fuzzy Inference Help Imbalanced Classification?

Code and benchmark datasets for our leakage-free empirical study evaluating **GAN-LMNN-FARCI+**, a hybrid framework combining Generative Adversarial Network (GAN) oversampling and Mahalanobis Large Margin Nearest Neighbor (LMNN) metric learning with FARCI+-style 3-tuples support tuning and non-averaging Choquet integral aggregation ($F_{GL}-F_{BPC}$).

## Repository Structure
- `run_experiment.py` — Leakage-free stratified 5-fold cross-validation benchmark across four fuzzy classifier configurations (`FRBCS`, `GAN-LMNN-FRBCS`, `FARCI+-style`, and `GAN-LMNN-FARCI+`).
- `generate_figures.py` — Script to reproduce the publication figures (`fig_metrics.pdf` and `fig_analysis.pdf`).
- `datasets_csv/` — The eight binarized UCI/KEEL benchmark datasets (`wisconsin`, `wine`, `haberman`, `vehicle`, `ecoli`, `digits_9`, `glass`, and `yeast`).
- `hasil_eksperimen_detail.csv` — Per-dataset Precision, Recall, $F_1$-score, and runtime results.

## Quick Start
```bash
pip install numpy scipy pandas scikit-learn matplotlib
python run_experiment.py
python generate_figures.py
