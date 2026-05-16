import numpy as np

cm = np.array([
    [141, 28, 59],
    [0, 0, 0],
    [107, 33, 0]
])

tp = np.diag(cm)
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp
tn = cm.sum() - (tp + fp + fn)

def safe_div(a, b):
    return np.where(b == 0, 0, a / b)

precision = safe_div(tp, tp + fp)
recall = safe_div(tp, tp + fn)
f1 = safe_div(2 * precision * recall, precision + recall)

print("Per-class precision:", precision)
print("Per-class recall:", recall)
print("Per-class f1:", f1)

print("Macro precision:", precision.mean())
print("Macro recall:", recall.mean())
print("Macro f1:", f1.mean())