import sys
sys.path.append('C:/Users/priha/Documents/cortex')
from core.modules.divergence import calculate_divergence_for_nodes

corpora_details = [
    {'path': 'C:/Users/priha/Documents/cortex/corpora/english/EN-BPPT-tagged.db', 'name': 'Corpus 1', 'xml_where': '', 'xml_params': [], 'stats': None},
    {'path': 'C:/Users/priha/Documents/cortex/corpora/english/EN-BPPT-tagged.db', 'name': 'Corpus 2', 'xml_where': '', 'xml_params': [], 'stats': None}
]
try:
    calculate_divergence_for_nodes(
        corpora_details=corpora_details,
        nodes_list=["test"],
        window_size=5,
        assoc_measure='Log-Likelihood',
        min_pair_freq=2,
        min_collocate_freq=2,
        top_k=20,
        metric='Jensen-Shannon Divergence',
        is_raw_mode=False,
        token_filter="",
        pos_filter="",
        lemma_filter="",
        min_node_collocates=2
    )
except Exception as e:
    import traceback
    traceback.print_exc()
