"""
MNIST classification with several approaches:
  1. Neural network (CNN) in PyTorch
  2. Linear Discriminant Analysis (LDA) via scikit-learn
  3. Quadratic Discriminant Analysis (QDA) via scikit-learn
  4. Support Vector Machine (SVM), RBF kernel, via scikit-learn
  5. SVM with an Earth-Mover's-Distance (EMD) kernel, which -- unlike plain
     RBF -- weighs differences by how far apart (in actual pixel
     coordinates) the mismatched ink is, not just by intensity difference.

Requirements:
    pip install torch torchvision scikit-learn numpy
    pip install pot   # only needed for --method svm-emd (Python Optimal Transport)

Run:
    python mnist_classifiers.py --method all
    python mnist_classifiers.py --method nn
    python mnist_classifiers.py --method lda
    python mnist_classifiers.py --method qda
    python mnist_classifiers.py --method svm
    python mnist_classifiers.py --method svm-emd
"""

import argparse
import time

import numpy as np


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------

def load_mnist_arrays():
    """Download MNIST via torchvision and return flat numpy arrays.

    Returns:
        X_train, y_train, X_test, y_test  (X arrays are float32 in [0, 1],
        shape (N, 784); y arrays are int64, shape (N,))
    """
    from torchvision import datasets, transforms

    to_tensor = transforms.ToTensor()
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=to_tensor)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=to_tensor)

    def ds_to_numpy(ds):
        X = ds.data.numpy().astype(np.float32) / 255.0  # (N, 28, 28)
        X = X.reshape(len(ds), -1)                        # (N, 784)
        y = ds.targets.numpy().astype(np.int64)
        return X, y

    X_train, y_train = ds_to_numpy(train_ds)
    X_test, y_test = ds_to_numpy(test_ds)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# 1. Neural network (CNN) in PyTorch
# ---------------------------------------------------------------------------

def run_neural_net(epochs=5, batch_size=128, lr=1e-3, device=None):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NN] Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=2)

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28x28 -> 28x28
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 -> 14x14
            self.pool = nn.MaxPool2d(2)                               # halves spatial dims
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)              # -> 14x14
            x = F.relu(self.conv2(x))
            x = self.pool(x)              # -> 7x7
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.dropout2(x)
            return self.fc2(x)            # raw logits

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        avg_loss = total_loss / len(train_ds)

        # quick eval each epoch
        model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
        acc = correct / len(test_ds)
        print(f"[NN] Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  test_acc={acc:.4f}")

    print(f"[NN] Training took {time.time() - start:.1f}s. Final test accuracy: {acc:.4f}")
    return model, acc


# ---------------------------------------------------------------------------
# 2. Linear Discriminant Analysis
# ---------------------------------------------------------------------------

def run_lda():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, classification_report

    X_train, y_train, X_test, y_test = load_mnist_arrays()

    print("[LDA] Fitting LinearDiscriminantAnalysis...")
    start = time.time()
    # 'svd' solver handles high-dimensional, singular covariance data well (no need to invert)
    clf = LinearDiscriminantAnalysis(solver="svd")
    clf.fit(X_train, y_train)
    print(f"[LDA] Fit took {time.time() - start:.1f}s")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[LDA] Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    return clf, acc


# ---------------------------------------------------------------------------
# 2b. Quadratic Discriminant Analysis
# ---------------------------------------------------------------------------
#
# QDA is LDA's more flexible cousin: instead of assuming every class shares
# one covariance matrix (giving linear decision boundaries), it fits a
# separate covariance matrix per class (giving quadratic boundaries). The
# catch on raw MNIST pixels: many pixels (e.g. the corners) are almost
# always zero, so several classes' 784x784 covariance matrices come out
# singular or near-singular -- QDA needs to invert each one individually,
# so it's more exposed to this than LDA. We fix that two ways: reduce
# dimensionality with PCA first, and use `reg_param` to shrink each class's
# covariance estimate toward a scaled identity matrix (regularization),
# which keeps every matrix invertible.

def run_qda(pca_components=50, reg_param=0.5):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score, classification_report

    X_train, y_train, X_test, y_test = load_mnist_arrays()

    print(f"[QDA] Fitting PCA({pca_components}) + QuadraticDiscriminantAnalysis "
          f"(reg_param={reg_param})...")
    clf = make_pipeline(
        PCA(n_components=pca_components, random_state=0),
        QuadraticDiscriminantAnalysis(reg_param=reg_param),
    )

    start = time.time()
    clf.fit(X_train, y_train)
    print(f"[QDA] Fit took {time.time() - start:.1f}s")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[QDA] Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    return clf, acc


# ---------------------------------------------------------------------------
# 3. Support Vector Machine
# ---------------------------------------------------------------------------

def run_svm(n_train_subset=10000, pca_components=50):
    """SVM on MNIST.

    Full 60k-sample RBF-SVM training is very slow (SVM training scales
    roughly quadratically-to-cubically with sample count), so by default
    this subsamples the training set and reduces dimensionality with PCA.
    Set n_train_subset=None to use the full training set (expect it to be
    much slower).
    """
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_train, y_train, X_test, y_test = load_mnist_arrays()

    if n_train_subset is not None:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(X_train), size=n_train_subset, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        print(f"[SVM] Using a random subset of {n_train_subset} training samples")

    # PCA to make RBF-SVM tractable on pixel data; StandardScaler helps the SVM converge
    clf = make_pipeline(
        StandardScaler(),
        PCA(n_components=pca_components, random_state=0),
        SVC(kernel="rbf", C=5, gamma="scale"),
    )

    print("[SVM] Fitting SVC (RBF kernel) pipeline...")
    start = time.time()
    clf.fit(X_train, y_train)
    print(f"[SVM] Fit took {time.time() - start:.1f}s")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[SVM] Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    return clf, acc


# ---------------------------------------------------------------------------
# 4. SVM with a spatially-aware kernel: Earth Mover's Distance (EMD)
# ---------------------------------------------------------------------------
#
# A standard RBF kernel is  K(x, y) = exp(-gamma * ||x - y||^2), where the
# distance is just the sum of squared per-pixel intensity differences. Two
# images that are identical but shifted by one pixel can look *very* far
# apart under that metric, even though visually they're nearly the same
# digit -- RBF has no notion of *where* in the image a mismatch occurs.
#
# The Earth Mover's Distance (a.k.a. Wasserstein distance) fixes this by
# treating each image as a pile of "ink mass" spread over pixel locations,
# and defining the distance between two images as the minimum cost to move
# one pile of mass onto the other, where cost = (amount of mass moved) x
# (actual Euclidean pixel distance it's moved). This literally weighs
# differences by physical pixel distance, not just intensity difference.
# We then build a kernel from it:
#
#     K(x, y) = exp(-gamma * EMD(x, y))
#
# Computing exact EMD per image pair is expensive, so we use the
# entropy-regularized approximation (Sinkhorn distance) from the POT
# library, and downsample images + subsample the dataset to keep runtime
# reasonable for a demo. This kernel must be precomputed and passed to
# sklearn's SVC via kernel="precomputed".

def _downsample_images(X, orig_size=28, new_size=14):
    """Average-pool each flattened (orig_size*orig_size,) image down to
    (new_size*new_size,) to shrink the cost of the OT computation."""
    factor = orig_size // new_size
    imgs = X.reshape(-1, orig_size, orig_size)
    imgs = imgs.reshape(-1, new_size, factor, new_size, factor).mean(axis=(2, 4))
    return imgs.reshape(-1, new_size * new_size)


def _pixel_ground_distance(grid_size):
    """Pairwise Euclidean distance between pixel (row, col) coordinates on a
    grid_size x grid_size image -- the 'actual pixel distance' used as the
    OT ground cost between two units of mass."""
    import ot  # POT: pip install pot

    coords = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)],
                       dtype=np.float64)
    M = ot.dist(coords, coords, metric="euclidean")
    M /= M.max()  # normalize to [0, 1] for numerical stability with Sinkhorn
    return M


def _images_to_histograms(X, eps=1e-6):
    """Turn flattened images into probability distributions over pixel
    locations (non-negative, sums to 1), as EMD requires."""
    X = X.astype(np.float64) + eps  # avoid exact zeros, which are unstable for Sinkhorn
    return X / X.sum(axis=1, keepdims=True)


def _emd_kernel_matrix(A_hist, B_hist, M, gamma, reg=0.05):
    """Build K[i, j] = exp(-gamma * sinkhorn_distance(A_hist[i], B_hist[j]))
    using POT's batched Sinkhorn (each row of A is compared against every
    column of B in one vectorized call, instead of one pair at a time)."""
    import ot

    n_a = A_hist.shape[0]
    n_b = B_hist.shape[0]
    K = np.empty((n_a, n_b), dtype=np.float64)
    B_T = B_hist.T  # (n_pixels, n_b), the batched-target format POT expects

    for i in range(n_a):
        # sinkhorn2 with a 2D target matrix returns the distance from A_hist[i]
        # to every column of B_T in one call.
        dists = ot.sinkhorn2(A_hist[i], B_T, M, reg=reg, numItermax=200)
        K[i, :] = np.exp(-gamma * np.asarray(dists))
        if (i + 1) % 100 == 0 or i == n_a - 1:
            print(f"[SVM-EMD]   kernel row {i + 1}/{n_a}", end="\r")
    print()
    return K


def run_svm_emd(n_train_subset=1000, n_test_subset=500, downsample_to=14,
                 gamma=5.0, sinkhorn_reg=0.05, C=5):
    """SVM with a precomputed EMD-based kernel instead of plain RBF.

    Kept deliberately small (n_train_subset, n_test_subset, downsample_to)
    because computing an OT-based kernel matrix is much more expensive than
    a Euclidean one: building the train-train kernel here costs
    n_train_subset Sinkhorn solves, each vectorized over the rest of the
    training set. Increase the subset sizes only if you're prepared to wait.
    """
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report

    X_train, y_train, X_test, y_test = load_mnist_arrays()

    rng = np.random.RandomState(0)
    train_idx = rng.choice(len(X_train), size=n_train_subset, replace=False)
    test_idx = rng.choice(len(X_test), size=n_test_subset, replace=False)
    X_train, y_train = X_train[train_idx], y_train[train_idx]
    X_test, y_test = X_test[test_idx], y_test[test_idx]

    print(f"[SVM-EMD] Downsampling images to {downsample_to}x{downsample_to}...")
    X_train_ds = _downsample_images(X_train, new_size=downsample_to)
    X_test_ds = _downsample_images(X_test, new_size=downsample_to)

    print("[SVM-EMD] Building pixel ground-distance matrix...")
    M = _pixel_ground_distance(downsample_to)

    print("[SVM-EMD] Converting images to mass distributions...")
    train_hist = _images_to_histograms(X_train_ds)
    test_hist = _images_to_histograms(X_test_ds)

    print(f"[SVM-EMD] Computing train-train kernel ({n_train_subset}x{n_train_subset})...")
    start = time.time()
    K_train = _emd_kernel_matrix(train_hist, train_hist, M, gamma, reg=sinkhorn_reg)
    print(f"[SVM-EMD] Train kernel took {time.time() - start:.1f}s")

    print(f"[SVM-EMD] Computing test-train kernel ({n_test_subset}x{n_train_subset})...")
    start = time.time()
    K_test = _emd_kernel_matrix(test_hist, train_hist, M, gamma, reg=sinkhorn_reg)
    print(f"[SVM-EMD] Test kernel took {time.time() - start:.1f}s")

    print("[SVM-EMD] Fitting SVC with precomputed kernel...")
    clf = SVC(kernel="precomputed", C=C)
    clf.fit(K_train, y_train)

    y_pred = clf.predict(K_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[SVM-EMD] Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    return clf, acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Classify MNIST via NN, LDA, and/or SVM")
    parser.add_argument("--method", choices=["nn", "lda", "qda", "svm", "svm-emd", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=5, help="epochs for the neural net")
    parser.add_argument("--svm-subset", type=int, default=10000,
                         help="number of training samples for SVM (use 0 for full 60k)")
    parser.add_argument("--svm-emd-train", type=int, default=1000,
                         help="number of training samples for the EMD-kernel SVM (kept small: expensive)")
    parser.add_argument("--svm-emd-test", type=int, default=500,
                         help="number of test samples for the EMD-kernel SVM")
    args = parser.parse_args()

    results = {}

    if args.method in ("nn", "all"):
        _, acc = run_neural_net(epochs=args.epochs)
        results["nn"] = acc

    if args.method in ("lda", "all"):
        _, acc = run_lda()
        results["lda"] = acc

    if args.method in ("qda", "all"):
        _, acc = run_qda()
        results["qda"] = acc

    if args.method in ("svm", "all"):
        subset = None if args.svm_subset == 0 else args.svm_subset
        _, acc = run_svm(n_train_subset=subset)
        results["svm"] = acc

    if args.method in ("svm-emd", "all"):
        _, acc = run_svm_emd(n_train_subset=args.svm_emd_train, n_test_subset=args.svm_emd_test)
        results["svm-emd"] = acc

    if results:
        print("\n=== Summary ===")
        for name, acc in results.items():
            print(f"{name.upper():>4}: {acc:.4f}")


if __name__ == "__main__":
    main()