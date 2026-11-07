import numpy as np
import pandas as pd
from collections import Counter

TERMINALES = {"Conversion", "Sin_conversion"}

def conteos(journeys):
    c = Counter()
    for j in journeys:
        for a, b in zip(j[:-1], j[1:]):
            c[(a, b)] += 1
    return c

def p_conversion(counts, removed=None):
    estados = sorted({s for par in counts for s in par} - TERMINALES)
    idx = {s: i for i, s in enumerate(estados)}
    n = len(estados)
    Q, R, tot = np.zeros((n, n)), np.zeros(n), np.zeros(n)
    for (a, b), v in counts.items():
        tot[idx[a]] += v
        if b == removed or b == "Sin_conversion":
            continue
        if b == "Conversion":
            R[idx[a]] += v
        else:
            Q[idx[a], idx[b]] += v
    Q, R = Q / tot[:, None], R / tot
    return np.linalg.solve(np.eye(n) - Q, R)[idx["Inicio"]]

def efecto_remocion(journeys):
    cnt = conteos(journeys)
    base = p_conversion(cnt)
    canales = sorted({s for par in cnt for s in par} - TERMINALES - {"Inicio"})
    ef = pd.Series({c: (base - p_conversion(cnt, removed=c)) / base for c in canales})
    return ef / ef.sum()