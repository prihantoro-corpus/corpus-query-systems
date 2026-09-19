import duckdb
con = duckdb.connect('C:/Users/priha/Documents/cortex/corpora/english/EN-BPPT-tagged.db', read_only=True)
lemmas = con.execute("SELECT COUNT(DISTINCT lower(lemma)) FROM corpus WHERE pos LIKE 'NN%' OR pos LIKE 'VB%' OR pos LIKE 'JJ%' OR pos LIKE 'RB%'").fetchone()[0]
print('Unique Lemmas (N, V, Adj, Adv):', lemmas)
tokens = con.execute("SELECT COUNT(DISTINCT _token_low) FROM corpus WHERE pos LIKE 'NN%' OR pos LIKE 'VB%' OR pos LIKE 'JJ%' OR pos LIKE 'RB%'").fetchone()[0]
print('Unique Tokens (N, V, Adj, Adv):', tokens)
total = con.execute("SELECT COUNT(*) FROM corpus").fetchone()[0]
print('Total Tokens in Corpus:', total)
con.close()
