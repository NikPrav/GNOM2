import pandas as pd

smerp = pd.read_csv("smerp17.csv", header=0, index_col=0)
smerp = smerp.sample(frac=1.)
smerp_sel = smerp.sample(frac=0.565)

smerp_train = smerp_sel.sample(frac=0.6)
print(smerp_train.shape)
smerp_sel = smerp_sel.drop(smerp_train.index)
smerp_val = smerp_sel.sample(frac=0.1)
print(smerp_val.shape)
smerp_sel = smerp_sel.drop(smerp_val.index)
smerp_test = smerp_sel
print(smerp_test.shape)

smerp_train.to_csv('smerp17_train.csv', index=False, header=True)
smerp_val.to_csv('smerp17_val.csv', index=False, header=True)
smerp_test.to_csv('smerp17_test.csv', index=False, header=True)
