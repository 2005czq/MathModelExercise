#!/usr/bin/env python3
"""
Fit C = w1 * D^2 + w2 * H using gradient descent on the 4 points in data/price.csv.
Also compute the closed-form (normal equations) solution as a check.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path


DATA_PATH = Path(__file__).parent / "data" / "price.csv"


def load_data(path: Path):
    """Load CSV with header D,H,C and return lists x1=D^2, x2=H, y=C."""
    x1, x2, y = [], [], []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            D = float(row["D"])  # distance
            H = float(row["H"])  # height
            C = float(row["C"])  # cost
            x1.append(D * D)
            x2.append(H)
            y.append(C)
    return x1, x2, y


def gradient_descent(x1, x2, y, *, lr=1e-2, iters=200000, scale1=10000.0, scale2=100.0):
    """Gradient descent without intercept using feature scaling by divisors scale1, scale2.

    We minimize MSE: (1/n) * sum (w1*x1 + w2*x2 - y)^2.

    To improve conditioning, we optimize w' over scaled features x1' = x1/scale1, x2' = x2/scale2
    and then map back: w1 = w1'/scale1, w2 = w2'/scale2.
    """
    n = len(y)
    xs1 = [v / scale1 for v in x1]
    xs2 = [v / scale2 for v in x2]

    w1p = 0.0
    w2p = 0.0

    for t in range(iters):
        g1 = 0.0
        g2 = 0.0
        loss = 0.0
        for i in range(n):
            pred = w1p * xs1[i] + w2p * xs2[i]
            err = pred - y[i]
            loss += err * err
            g1 += err * xs1[i]
            g2 += err * xs2[i]
        g1 *= 2.0 / n
        g2 *= 2.0 / n

        w1p -= lr * g1
        w2p -= lr * g2

        if (t + 1) % 10000 == 0:
            rmse = math.sqrt(loss / n)
            # print(f"iter {t+1:7d}  rmse={rmse:10.6f}  w1'={w1p:.8f} w2'={w2p:.8f}")
            if max(abs(g1), abs(g2)) < 1e-6:
                break

    w1 = w1p / scale1
    w2 = w2p / scale2
    return w1, w2


def normal_equations_2x2(x1, x2, y):
    """Solve [S11 S12; S12 S22] [w1 w2]^T = [t1 t2]^T without numpy."""
    S11 = sum(v * v for v in x1)
    S22 = sum(v * v for v in x2)
    S12 = sum(a * b for a, b in zip(x1, x2))
    t1 = sum(a * c for a, c in zip(x1, y))
    t2 = sum(b * c for b, c in zip(x2, y))

    det = S11 * S22 - S12 * S12
    if abs(det) < 1e-12:
        raise ZeroDivisionError("Design matrix is singular or ill-conditioned for 2x2 system.")

    w1 = (t1 * S22 - t2 * S12) / det
    w2 = (S11 * t2 - S12 * t1) / det
    return w1, w2


def main():
    x1, x2, y = load_data(DATA_PATH)

    w1_gd, w2_gd = gradient_descent(x1, x2, y)  # use tuned defaults
    w1_cf, w2_cf = normal_equations_2x2(x1, x2, y)

    print("Gradient Descent Solution (no intercept):")
    print(f"  w1 = {w1_gd:.10f}")
    print(f"  w2 = {w2_gd:.10f}")
    print()
    print("Closed-form (normal equations) Solution:")
    print(f"  w1 = {w1_cf:.10f}")
    print(f"  w2 = {w2_cf:.10f}")

    def mse(w1, w2):
        n = len(y)
        return sum((w1 * x1[i] + w2 * x2[i] - y[i]) ** 2 for i in range(n)) / n

    print()
    print(f"MSE GD: {mse(w1_gd, w2_gd):.6f}")
    print(f"MSE CF: {mse(w1_cf, w2_cf):.6f}")


if __name__ == "__main__":
    main()
