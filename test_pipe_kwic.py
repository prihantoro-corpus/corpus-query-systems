import sys
import os
sys.path.insert(0, '.')
from core.modules.concordance import generate_kwic

db = 'c:/Users/priha/Documents/cortex/corpora/indonesian/uam_trial/TUFS2023KOMSHI314.db'
q = '<segment tag="yg0|nom0|vrbs">'
res = generate_kwic(db, q, 5, 5, 'test')

print("Count:", len(res[0]))
for r in res[0]:
    print("Node:", r['Node'], "| Metadata:", r['Metadata'])
