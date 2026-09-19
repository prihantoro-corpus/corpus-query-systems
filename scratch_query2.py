import duckdb
con = duckdb.connect('C:/Users/priha/Documents/cortex/corpora/english/EN-BPPT-tagged.db', read_only=True)

for min_freq in [5, 10, 20, 50]:
    lemmas_freq = con.execute(f"SELECT COUNT(*) FROM (SELECT lower(lemma) as l, COUNT(*) as c FROM corpus WHERE pos LIKE 'NN%' OR pos LIKE 'VB%' OR pos LIKE 'JJ%' OR pos LIKE 'RB%' GROUP BY l HAVING c >= {min_freq})").fetchone()[0]
    print(f'Unique Lemmas (Freq >= {min_freq}):', lemmas_freq)

con.close()
