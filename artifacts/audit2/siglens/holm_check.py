import numpy as np, pandas as pd
def holm(ps):
    m = len(ps); s = np.sort(ps); out = []; run = 0.0
    for i, raw in enumerate(s):
        run = max(run, min(1.0, raw * (m - i))); out.append(run)
    return np.array(out)
cases = [[0.01,0.02,0.30],[0.0244]*6,[0.0244,0.0488,0.1,0.2,0.5,0.9],
         [0.4,0.001,0.02],[0.5,0.5,0.5,0.5,0.5]]
try:
    from statsmodels.stats.multitest import multipletests
    ok = True
except Exception:
    ok = False
for c in cases:
    mine = holm(c)
    line = f"{c} -> {np.round(mine,5)}"
    if ok:
        ref = multipletests(np.sort(c), method="holm")[1]
        line += f"   statsmodels {np.round(ref,5)}  match={np.allclose(mine, ref)}"
    print(line)
print("\nmonotone in every case:", all(np.all(np.diff(holm(c))>=-1e-15) for c in cases))
print("\nFLOOR ANALYSIS with 40 placebos (min p = 1/41 = %.4f):" % (1/41))
for m in [5, 6]:
    print(f"  m={m} tests -> smallest attainable p_holm = {1/41*m:.4f}"
          f"   significant at 0.10? {'YES' if 1/41*m<=0.10 else 'NO - impossible'}")
