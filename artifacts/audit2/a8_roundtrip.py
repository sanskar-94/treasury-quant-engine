import sys; sys.path.insert(0,"src")
import pandas as pd, numpy as np
from tqe.config import load_config
from tqe.backtest.costs import CostModel
from tqe.backtest.engine import _core_loop
cfg=load_config(); cm=CostModel(cfg.costs)
idx=pd.bdate_range("2020-01-01",periods=10)
p=np.zeros(10); p[2:7]=10e6           # build day2, hold days 3-6, unwind day7
pos=pd.DataFrame({"10 Yr":p},index=idx)
rets=pd.DataFrame(0.0,index=idx,columns=["10 Yr"])
net,g,cost,trades,fin=_core_loop(pos,rets,cm,{"10 Yr":"10y"},1e7,True,1.0,funding_rate=None,include_financing=False)
print("daily costs:", np.round(cost.to_numpy(),2))
one=float(cm.total_cost(10e6,"10y"))
print("one-way closed form: spread %.2f + impact %.2f + comm %.2f = %.2f"%(
  cm.spread_cost(10e6,"10y"),cm.impact_cost(10e6,"10y"),cm.commission(10e6),one))
print("engine total %.2f  vs 2 x one-way %.2f"%(cost.sum(),2*one))
print("cost on held (non-trade) days:", cost.iloc[3:7].sum())
