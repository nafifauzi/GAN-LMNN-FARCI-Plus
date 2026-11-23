"""
Leakage-Free Empirical Evaluation of GAN-LMNN Preprocessing with Choquet-Based
Fuzzy Rule-Based Classification (GAN-LMNN-FARCI+) on Binary Imbalanced Datasets.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
from sklearn.cluster import KMeans
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_digits,
    load_wine,
)
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
np.random.seed(42)


# ==============================================================================
# 1. TABULAR GAN OVERSAMPLING (Kang et al., 2025 - Section III-A & IV-B)
# ==============================================================================
class TabularGAN:
    """Lightweight 3-layer Generative Adversarial Network for minority synthesis."""

    def __init__(self, input_dim, noise_dim=16, lr=0.002, epochs=250, batch_size=32):
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._init_weights()

    def _init_weights(self):
        rng = np.random.default_rng(42)
        # Generator: noise_dim -> 128 -> 256 -> input_dim
        self.g_w1 = rng.normal(0, 0.1, (self.noise_dim, 128))
        self.g_b1 = np.zeros(128)
        self.g_w2 = rng.normal(0, 0.1, (128, 256))
        self.g_b2 = np.zeros(256)
        self.g_w3 = rng.normal(0, 0.1, (256, self.input_dim))
        self.g_b3 = np.zeros(self.input_dim)

        # Discriminator: input_dim -> 256 -> 128 -> 1
        self.d_w1 = rng.normal(0, 0.1, (self.input_dim, 256))
        self.d_b1 = np.zeros(256)
        self.d_w2 = rng.normal(0, 0.1, (256, 128))
        self.d_b2 = np.zeros(128)
        self.d_w3 = rng.normal(0, 0.1, (128, 1))
        self.d_b3 = np.zeros(1)

    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    def _g_forward(self, z):
        h1 = self._relu(z @ self.g_w1 + self.g_b1)
        h2 = self._relu(h1 @ self.g_w2 + self.g_b2)
        out = self._sigmoid(h2 @ self.g_w3 + self.g_b3)
        return out, (z, h1, h2)

    def _d_forward(self, x):
        h1 = self._relu(x @ self.d_w1 + self.d_b1)
        h2 = self._relu(h1 @ self.d_w2 + self.d_b2)
        logits = h2 @ self.d_w3 + self.d_b3
        prob = self._sigmoid(logits)
        return prob, (x, h1, h2)

    def fit_resample(self, X, y):
        """Train GAN on minority samples and generate synthetic instances to reach IR = 1."""
        classes, counts = np.unique(y, return_counts=True)
        min_class = classes[np.argmin(counts)]
        n_maj = np.max(counts)
        n_min = np.min(counts)
        n_to_gen = n_maj - n_min
        if n_to_gen <= 0:
            return X, y

        X_min = X[y == min_class]
        rng = np.random.default_rng(42)
        bs = min(self.batch_size, len(X_min))

        for _ in range(self.epochs):
            idx = rng.choice(len(X_min), bs, replace=True)
            real_x = X_min[idx]
            z = rng.normal(0.0, 1.0, (bs, self.noise_dim))
            fake_x, _ = self._g_forward(z)

            # Update Discriminator parameters
            p_real, (rx, rh1, rh2) = self._d_forward(real_x)
            p_fake, (fx, fh1, fh2) = self._d_forward(fake_x)

            d_grad_real = (p_real - 1.0) / bs
            d_grad_fake = p_fake / bs

            for grad, (inp, h1, h2) in [
                (d_grad_real, (rx, rh1, rh2)),
                (d_grad_fake, (fx, fh1, fh2)),
            ]:
                dw3 = h2.T @ grad
                db3 = np.sum(grad, axis=0)
                dh2 = (grad @ self.d_w3.T) * (h2 > 0)
                dw2 = h1.T @ dh2
                db2 = np.sum(dh2, axis=0)
                dh1 = (dh2 @ self.d_w2.T) * (h1 > 0)
                dw1 = inp.T @ dh1
                db1 = np.sum(dh1, axis=0)

                self.d_w3 -= self.lr * dw3
                self.d_b3 -= self.lr * db3
                self.d_w2 -= self.lr * dw2
                self.d_b2 -= self.lr * db2
                self.d_w1 -= self.lr * dw1
                self.d_b1 -= self.lr * db1

            # Update Generator parameters
            z = rng.normal(0.0, 1.0, (bs, self.noise_dim))
            fake_x, (gz, gh1, gh2) = self._g_forward(z)
            p_fake, (_, dfh1, dfh2) = self._d_forward(fake_x)
            g_grad_d = (p_fake - 1.0) / bs

            dh2 = (g_grad_d @ self.d_w3.T) * (dfh2 > 0)
            dh1 = (dh2 @ self.d_w2.T) * (dfh1 > 0)
            dx = (dh1 @ self.d_w1.T) * (fake_x * (1.0 - fake_x))

            gw3 = gh2.T @ dx
            gb3 = np.sum(dx, axis=0)
            gh2_g = (dx @ self.g_w3.T) * (gh2 > 0)
            gw2 = gh1.T @ gh2_g
            gb2 = np.sum(gh2_g, axis=0)
            gh1_g = (gh2_g @ self.g_w2.T) * (gh1 > 0)
            gw1 = gz.T @ gh1_g
            gb1 = np.sum(gh1_g, axis=0)

            self.g_w3 -= self.lr * gw3
            self.g_b3 -= self.lr * gb3
            self.g_w2 -= self.lr * gw2
            self.g_b2 -= self.lr * gb2
            self.g_w1 -= self.lr * gw1
            self.g_b1 -= self.lr * gb1

        z_gen = rng.normal(0.0, 1.0, (n_to_gen, self.noise_dim))
        X_syn, _ = self._g_forward(z_gen)
        # Anchor generated vectors to real minority instances to mitigate mode collapse
        anchors = X_min[rng.choice(len(X_min), n_to_gen, replace=True)]
        X_syn = 0.7 * X_syn + 0.3 * anchors

        X_res = np.vstack([X, X_syn])
        y_res = np.concatenate([y, np.full(n_to_gen, min_class)])
        return X_res, y_res


# ==============================================================================
# 2. MAHALANOBIS LMNN METRIC LEARNING (Kang et al., 2025 - Section III-B)
# ==============================================================================
class MahalanobisLMNN:
    """Large Margin Nearest Neighbor (LMNN) linear projection initialized via S^(-1/2)."""

    def __init__(self, k=3, gamma=0.5, lr=0.01, max_iter=60):
        self.k = k
        self.gamma = gamma
        self.lr = lr
        self.max_iter = max_iter
        self.L = None
        self.scaler = MinMaxScaler()

    def fit_transform(self, X, y):
        """Learn linear transformation L on training fold and project into [0, 1]^m."""
        n, d = X.shape
        cov = np.cov(X, rowvar=False) + np.eye(d) * 1e-4
        eigvals, eigvecs = np.linalg.eigh(cov)
        inv_sqrt = (
            eigvecs
            @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-4)))
            @ eigvecs.T
        )
        L = inv_sqrt.copy()

        # Identify k same-class target neighbors for each instance
        target_pairs = []
        for idx in range(n):
            same_idx = np.where((y == y[idx]) & (np.arange(n) != idx))[0]
            if len(same_idx) == 0:
                continue
            dists = np.sum((X[same_idx] - X[idx]) ** 2, axis=1)
            nn_idx = same_idx[np.argsort(dists)[: min(self.k, len(same_idx))]]
            for j in nn_idx:
                target_pairs.append((idx, j))

        if len(target_pairs) > 300:
            rng = np.random.default_rng(42)
            sel = rng.choice(len(target_pairs), 300, replace=False)
            target_pairs = [target_pairs[s] for s in sel]

        for _ in range(self.max_iter):
            grad = np.zeros((d, d))
            Y_proj = X @ L
            for i, j in target_pairs:
                diff_ij = X[i] - X[j]
                d_ij = np.sum((Y_proj[i] - Y_proj[j]) ** 2)
                grad += 2.0 * np.outer(diff_ij, diff_ij) @ L

                diff_cls = np.where(y != y[i])[0]
                if len(diff_cls) > 0:
                    d_ik_all = np.sum((Y_proj[diff_cls] - Y_proj[i]) ** 2, axis=1)
                    k_imp = diff_cls[np.argmin(d_ik_all)]
                    d_ik = np.min(d_ik_all)
                    if 1.0 + d_ij - d_ik > 0:
                        diff_ik = X[i] - X[k_imp]
                        grad += (
                            2.0
                            * self.gamma
                            * (np.outer(diff_ij, diff_ij) - np.outer(diff_ik, diff_ik))
                            @ L
                        )

            L -= (self.lr / max(len(target_pairs), 1)) * grad

        self.L = L
        return self.scaler.fit_transform(X @ self.L)

    def transform(self, X):
        """Project unseen test instances using the learned transformation L and scaler."""
        return self.scaler.transform(X @ self.L)


# ==============================================================================
# 3. FRBCS & FARCI+-STYLE CLASSIFIER (Sanz & Sesma-Sara, 2026 - Section III)
# ==============================================================================
class FuzzyRuleClassifier:
    """
    Fuzzy Rule-Based Classification System supporting standard additive reasoning
    and FARCI+-style 3-tuples support tuning with non-averaging Choquet aggregation.
    """

    def __init__(self, n_labels=5, use_farci_plus=False, max_rules_per_class=25):
        self.n_labels = n_labels
        self.use_farci_plus = use_farci_plus
        self.max_rules_per_class = max_rules_per_class
        self.rules = []       # List of tuples: (antecedent_indices, class_label, rule_weight)
        self.alpha = None     # Lateral symbolic shifts per feature
        self.beta = None      # Support width modifiers per feature
        self.q_params = None  # Power measure exponents per class

    def _triangular_mf(self, x, centers, widths):
        N, D = x.shape
        L = self.n_labels
        mu = np.zeros((N, D, L))
        for l in range(L):
            c = centers[:, l]
            w = np.maximum(widths[:, l], 1e-3)
            mu[:, :, l] = np.maximum(0.0, 1.0 - np.abs(x - c) / w)
        return mu

    def _get_mf_params(self, D):
        base_c = np.linspace(0.0, 1.0, self.n_labels)
        base_w = 1.0 / (self.n_labels - 1)
        centers = np.tile(base_c, (D, 1))
        widths = np.full((D, self.n_labels), base_w)

        if self.use_farci_plus and self.alpha is not None:
            # 3-tuples fuzzy linguistic model: (s_i, alpha_i, beta_i), alpha, beta in [-0.5, 0.5)
            centers = np.clip(centers + self.alpha[:, None] * base_w, 0.0, 1.0)
            widths = widths * (1.0 + self.beta[:, None])
        return centers, widths

    def _extract_rules(self, X, y):
        N, D = X.shape
        centers, widths = self._get_mf_params(D)
        mu = self._triangular_mf(X, centers, widths)
        best_labels = np.argmax(mu, axis=2)

        self.rules = []
        for cls in [0, 1]:
            X_c_idx = np.where(y == cls)[0]
            if len(X_c_idx) == 0:
                continue
            n_clusters = min(self.max_rules_per_class, max(3, len(X_c_idx) // 2))
            if len(X_c_idx) <= n_clusters:
                prototypes = best_labels[X_c_idx]
            else:
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
                km.fit(X[X_c_idx])
                proto_mu = self._triangular_mf(km.cluster_centers_, centers, widths)
                prototypes = np.argmax(proto_mu, axis=2)

            unique_protos = np.unique(prototypes, axis=0)
            for ant in unique_protos:
                # Rule firing degree via geometric mean across antecedents
                fire_all = (
                    np.prod(
                        mu[np.arange(N)[:, None], np.arange(D)[None, :], ant[None, :]],
                        axis=1,
                    )
                    ** (1.0 / D)
                )
                sum_c = np.sum(fire_all[y == cls])
                sum_not_c = np.sum(fire_all[y != cls])
                denom = np.sum(fire_all) + 1e-8
                # Penalized certainty for FRBCS vs. confidence for FARCI+-style
                rw = (
                    (sum_c - sum_not_c) / denom
                    if not self.use_farci_plus
                    else sum_c / denom
                )
                if rw > 0.05:
                    self.rules.append((ant, cls, rw))

    def _choquet_fgl_fbpc(self, activations, q):
        """Non-averaging C_{F1, F2} Choquet integral with F1 = F_GL and F2 = F_BPC."""
        n = len(activations)
        if n == 0:
            return 0.0
        x_sorted = np.sort(activations)
        # Symmetric power measure: m_PM(A_(i)) = ((n - i + 1) / n)^q
        sizes = np.arange(n, 0, -1) / float(n)
        m_vals = sizes ** q

        total = x_sorted[0]
        if n > 1:
            xi = x_sorted[1:]
            xim1 = x_sorted[:-1]
            mi = m_vals[1:]
            # F_GL(x, y) = sqrt(x * (y + 1) / 2)
            f1 = np.sqrt(np.clip(xi * (mi + 1.0) / 2.0, 0.0, 1.0))
            # F_BPC(x, y) = x * y^2
            f2 = xim1 * (mi ** 2)
            total += np.sum(f1 - f2)
        return min(1.0, max(0.0, total))

    def _predict_with_params(self, X):
        N, D = X.shape
        centers, widths = self._get_mf_params(D)
        mu = self._triangular_mf(X, centers, widths)

        scores = np.zeros((N, 2))
        if not self.use_farci_plus:
            # Standard additive fuzzy reasoning method
            for ant, cls, rw in self.rules:
                fire = (
                    np.prod(
                        mu[np.arange(N)[:, None], np.arange(D)[None, :], ant[None, :]],
                        axis=1,
                    )
                    ** (1.0 / D)
                )
                scores[:, cls] += fire * rw
        else:
            # Non-averaging Choquet integral aggregation per class
            rules_c0 = [(ant, rw) for ant, cls, rw in self.rules if cls == 0]
            rules_c1 = [(ant, rw) for ant, cls, rw in self.rules if cls == 1]

            act_c0 = np.zeros((N, len(rules_c0)))
            for idx, (ant, rw) in enumerate(rules_c0):
                fire = (
                    np.prod(
                        mu[np.arange(N)[:, None], np.arange(D)[None, :], ant[None, :]],
                        axis=1,
                    )
                    ** (1.0 / D)
                )
                act_c0[:, idx] = fire * rw

            act_c1 = np.zeros((N, len(rules_c1)))
            for idx, (ant, rw) in enumerate(rules_c1):
                fire = (
                    np.prod(
                        mu[np.arange(N)[:, None], np.arange(D)[None, :], ant[None, :]],
                        axis=1,
                    )
                    ** (1.0 / D)
                )
                act_c1[:, idx] = fire * rw

            for i in range(N):
                nz0 = act_c0[i][act_c0[i] > 1e-6]
                nz1 = act_c1[i][act_c1[i] > 1e-6]
                scores[i, 0] = self._choquet_fgl_fbpc(nz0, self.q_params[0])
                scores[i, 1] = self._choquet_fgl_fbpc(nz1, self.q_params[1])

        return np.argmax(scores, axis=1)

    def fit(self, X, y):
        _, D = X.shape
        self.alpha = np.zeros(D)
        self.beta = np.zeros(D)
        self.q_params = np.array([1.0, 1.0])
        self._extract_rules(X, y)

        if self.use_farci_plus and len(self.rules) > 0:
            # Random-perturbation hill climbing on training F1-score
            rng = np.random.default_rng(42)
            best_f1 = f1_score(y, self._predict_with_params(X), zero_division=0)
            best_state = (self.alpha.copy(), self.beta.copy(), self.q_params.copy())

            for _ in range(35):
                cand_alpha = np.clip(
                    best_state[0] + rng.normal(0.0, 0.12, D), -0.45, 0.45
                )
                cand_beta = np.clip(
                    best_state[1] + rng.normal(0.0, 0.15, D), -0.45, 0.45
                )
                cand_q = np.clip(
                    best_state[2] + rng.normal(0.0, 0.25, 2), 0.1, 1.9
                )

                self.alpha, self.beta, self.q_params = cand_alpha, cand_beta, cand_q
                preds = self._predict_with_params(X)
                score = f1_score(y, preds, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_state = (
                        cand_alpha.copy(),
                        cand_beta.copy(),
                        cand_q.copy(),
                    )

            self.alpha, self.beta, self.q_params = best_state
        return self

    def predict(self, X):
        if len(self.rules) == 0:
            return np.zeros(len(X), dtype=int)
        return self._predict_with_params(X)


# ==============================================================================
# 4. BENCHMARK RUNNER (Stratified 5-Fold Cross-Validation)
# ==============================================================================
def load_and_save_csv_datasets(output_dir="datasets_csv"):
    """Load eight binary imbalanced benchmarks and export them as CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    loaded = []

    # Built-in scikit-learn datasets converted to binary tasks
    bc = load_breast_cancer()
    y_bc = (bc.target == 0).astype(int)  # Malignant (212) vs. Benign (357)
    loaded.append(("wisconsin", bc.data, y_bc))

    wine = load_wine()
    y_wine = (wine.target == 2).astype(int)  # Class 2 (48) vs. Rest (130)
    loaded.append(("wine", wine.data, y_wine))

    digits = load_digits()
    y_dig = (digits.target == 9).astype(int)  # Digit 9 (180) vs. Rest (1617)
    loaded.append(("digits_9", digits.data, y_dig))

    # OpenML UCI/KEEL benchmarks
    openml_specs = [
        ("ecoli", 1, "imU"),               # IR ~ 8.60 (35 vs. 301)
        ("glass", 1, "vehic wind float"),  # IR ~ 11.59 (17 vs. 197)
        ("yeast", 1, "ME2"),               # IR ~ 28.10 (51 vs. 1433)
        ("haberman", 1, "2"),              # IR ~ 2.78 (81 vs. 225)
        ("vehicle", 1, "van"),             # IR ~ 3.25 (199 vs. 647)
    ]

    for name, ver, min_label in openml_specs:
        try:
            ds = fetch_openml(name=name, version=ver, as_frame=False, parser="auto")
            X_d = np.nan_to_num(ds.data.astype(float), nan=0.0)
            y_d = (ds.target.astype(str) == min_label).astype(int)
            loaded.append((name, X_d, y_d))
        except Exception as exc:
            print(f"[Warning] Skipped dataset '{name}': {exc}")

    for name, X_d, y_d in loaded:
        cols = [f"f{i + 1}" for i in range(X_d.shape[1])] + ["class"]
        df_csv = pd.DataFrame(np.column_stack([X_d, y_d]), columns=cols)
        df_csv.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)

    print(f"Prepared {len(loaded)} binary datasets in '{output_dir}/'.")
    return loaded


def run_benchmark():
    """Execute leakage-free 5-fold CV across all four classifier configurations."""
    print("Preparing binary imbalanced benchmark datasets...")
    dataset_list = load_and_save_csv_datasets()

    models = ["FRBCS", "GAN-LMNN-FRBCS", "FARCI+", "GAN-LMNN-FARCI+"]
    records = []

    for name, X_raw, y_raw in dataset_list:
        n_pos = int(np.sum(y_raw == 1))
        n_neg = int(np.sum(y_raw == 0))
        ir = n_neg / max(n_pos, 1)

        print(
            f">>> Evaluating: {name:10s} | Samples: {len(y_raw):4d} | "
            f"Features: {X_raw.shape[1]:2d} | IR: {ir:.2f}"
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_metrics = {m: {"P": [], "R": [], "F1": [], "Time": []} for m in models}

        for train_idx, test_idx in skf.split(X_raw, y_raw):
            scaler = MinMaxScaler()
            X_tr = scaler.fit_transform(X_raw[train_idx])
            X_te = scaler.transform(X_raw[test_idx])
            y_tr, y_te = y_raw[train_idx], y_raw[test_idx]

            # 1. Baseline FRBCS (additive inference)
            t0 = time.time()
            clf1 = FuzzyRuleClassifier(n_labels=5, use_farci_plus=False)
            clf1.fit(X_tr, y_tr)
            pred1 = clf1.predict(X_te)
            t1 = time.time() - t0
            fold_metrics["FRBCS"]["P"].append(precision_score(y_te, pred1, zero_division=0))
            fold_metrics["FRBCS"]["R"].append(recall_score(y_te, pred1, zero_division=0))
            fold_metrics["FRBCS"]["F1"].append(f1_score(y_te, pred1, zero_division=0))
            fold_metrics["FRBCS"]["Time"].append(t1)

            # Data-level preprocessing fitted strictly on training fold
            t_prep0 = time.time()
            gan = TabularGAN(input_dim=X_tr.shape[1], epochs=200)
            X_tr_gan, y_tr_gan = gan.fit_resample(X_tr, y_tr)
            lmnn = MahalanobisLMNN(k=3, gamma=0.5, max_iter=50)
            X_tr_lmnn = lmnn.fit_transform(X_tr_gan, y_tr_gan)
            X_te_lmnn = lmnn.transform(X_te)
            t_prep = time.time() - t_prep0

            # 2. GAN-LMNN-FRBCS (Kang et al., 2025)
            t0 = time.time()
            clf2 = FuzzyRuleClassifier(n_labels=5, use_farci_plus=False)
            clf2.fit(X_tr_lmnn, y_tr_gan)
            pred2 = clf2.predict(X_te_lmnn)
            t2 = t_prep + (time.time() - t0)
            fold_metrics["GAN-LMNN-FRBCS"]["P"].append(precision_score(y_te, pred2, zero_division=0))
            fold_metrics["GAN-LMNN-FRBCS"]["R"].append(recall_score(y_te, pred2, zero_division=0))
            fold_metrics["GAN-LMNN-FRBCS"]["F1"].append(f1_score(y_te, pred2, zero_division=0))
            fold_metrics["GAN-LMNN-FRBCS"]["Time"].append(t2)

            # 3. FARCI+-style (Sanz & Sesma-Sara, 2026)
            t0 = time.time()
            clf3 = FuzzyRuleClassifier(n_labels=5, use_farci_plus=True)
            clf3.fit(X_tr, y_tr)
            pred3 = clf3.predict(X_te)
            t3 = time.time() - t0
            fold_metrics["FARCI+"]["P"].append(precision_score(y_te, pred3, zero_division=0))
            fold_metrics["FARCI+"]["R"].append(recall_score(y_te, pred3, zero_division=0))
            fold_metrics["FARCI+"]["F1"].append(f1_score(y_te, pred3, zero_division=0))
            fold_metrics["FARCI+"]["Time"].append(t3)

            # 4. Hybrid GAN-LMNN-FARCI+
            t0 = time.time()
            clf4 = FuzzyRuleClassifier(n_labels=5, use_farci_plus=True)
            clf4.fit(X_tr_lmnn, y_tr_gan)
            pred4 = clf4.predict(X_te_lmnn)
            t4 = t_prep + (time.time() - t0)
            fold_metrics["GAN-LMNN-FARCI+"]["P"].append(precision_score(y_te, pred4, zero_division=0))
            fold_metrics["GAN-LMNN-FARCI+"]["R"].append(recall_score(y_te, pred4, zero_division=0))
            fold_metrics["GAN-LMNN-FARCI+"]["F1"].append(f1_score(y_te, pred4, zero_division=0))
            fold_metrics["GAN-LMNN-FARCI+"]["Time"].append(t4)

        row = {
            "Dataset": name,
            "Samples": len(y_raw),
            "Features": X_raw.shape[1],
            "IR": round(ir, 2),
        }
        for m in models:
            row[f"{m}_P"] = round(float(np.mean(fold_metrics[m]["P"])), 4)
            row[f"{m}_R"] = round(float(np.mean(fold_metrics[m]["R"])), 4)
            row[f"{m}_F1"] = round(float(np.mean(fold_metrics[m]["F1"])), 4)
            row[f"{m}_Time"] = round(float(np.mean(fold_metrics[m]["Time"])), 2)
        records.append(row)

    df = pd.DataFrame(records)
    csv_path = "experiment_results_detail.csv"
    txt_path = "experiment_summary.txt"
    df.to_csv(csv_path, index=False)

    summary_lines = [
        "=== EXPERIMENT RESULTS SUMMARY (STRATIFIED 5-FOLD CV) ===",
        df.to_string(index=False),
        "\n=== OVERALL MEAN PERFORMANCE ACROSS DATASETS ===",
    ]
    for m in models:
        mean_p = df[f"{m}_P"].mean()
        mean_r = df[f"{m}_R"].mean()
        mean_f1 = df[f"{m}_F1"].mean()
        mean_t = df[f"{m}_Time"].mean()
        summary_lines.append(
            f"{m:18s} | Precision: {mean_p:.4f} | Recall: {mean_r:.4f} | "
            f"F1-Score: {mean_f1:.4f} | Avg Time: {mean_t:.2f}s"
        )

    summary_lines.append(
        "\n=== NON-PARAMETRIC STATISTICAL TESTS (ONE-SIDED WILCOXON ON F1) ==="
    )
    prop_f1 = df["GAN-LMNN-FARCI+_F1"].values
    for base in ["FRBCS", "GAN-LMNN-FRBCS", "FARCI+"]:
        base_f1 = df[f"{base}_F1"].values
        try:
            stat, p_val = wilcoxon(prop_f1, base_f1, alternative="greater")
        except ValueError:
            stat, p_val = 0.0, 1.0
        summary_lines.append(
            f"GAN-LMNN-FARCI+ vs. {base:15s} -> W-stat: {stat:.2f}, p-value: {p_val:.4f}"
        )

    fr_stat, fr_p = friedmanchisquare(
        df["FRBCS_F1"].values,
        df["GAN-LMNN-FRBCS_F1"].values,
        df["FARCI+_F1"].values,
        df["GAN-LMNN-FARCI+_F1"].values,
    )
    summary_lines.append(
        f"Omnibus Friedman Test (4 methods)   -> Chi2: {fr_stat:.4f}, p-value: {fr_p:.4f}"
    )

    output_text = "\n".join(summary_lines)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    print("\n" + output_text)
    print(f"\n[DONE] Saved per-dataset results to '{csv_path}' and summary to '{txt_path}'.")


if __name__ == "__main__":
    run_benchmark()