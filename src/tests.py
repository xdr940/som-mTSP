import numpy as np

aa=np.array([1,2,3,4])
bb=np.array([2,3,4,5])

ab = [aa,bb]

print('ok')
for idx,itm in enumerate(ab):
    ab[idx] = list(itm)

print('ok')


