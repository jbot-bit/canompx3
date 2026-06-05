from trading_app import holdout_policy as hp
print("HOLDOUT_SACRED_FROM:", getattr(hp,"HOLDOUT_SACRED_FROM",None))
print("HOLDOUT_GRANDFATHER_CUTOFF:", getattr(hp,"HOLDOUT_GRANDFATHER_CUTOFF",None))
print([n for n in dir(hp) if n.isupper()])
