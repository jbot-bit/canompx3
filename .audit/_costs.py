from pipeline.cost_model import COST_SPECS
for k,v in COST_SPECS.items():
    print(k, "::", v)
print()
# inspect what pnl_r in orb_outcomes represents - is it cost-inclusive?
import inspect, pipeline.cost_model as cm
print("cost_model functions:", [n for n in dir(cm) if not n.startswith("_")])
