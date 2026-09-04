# Generation diff vs reference — base

- problems compared: **30**
- ours: **6/30 = 20.00%**
- Kim et al.: **4/30 = 13.33%**
- per-problem verdict agreement: **28/30**
- we solve but they don't: **2** [86, 87]
- they solve but we don't: **0** []

## Are the traces the same text?

- byte-identical greedy traces: **4/30**
- mean shared prefix (fraction of reference trace): **36.2%**
- problems diverging within the first 1% of the trace: **5/30**

## Length / truncation

| | ours | reference |
| --- | --- | --- |
| mean response chars | 3377 | 3608 |
| max response chars | 21002 | 22485 |
| responses containing \boxed | 25/30 | 27/30 |
