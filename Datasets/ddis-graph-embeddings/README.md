
TransE embeddings for the ddis movie graph
==========================================

* Recall the TransE scoring function:
    || head + relation - tail ||_2

* The embedding dimension is 256

* The embeddings are stored as raw numpy matrices:
    * entity_embeds:   158'901 x 256
    * relation_embeds:     248 x 256

* The mapping between qid and index is stored in entity_ids.del and relation_ids.del, respectively.

* Note that there's a ddis:indirectSubclassOf relation which was not part of the graph yet.
  It links type entities (e.g. "bridge") to high-level types (e.g. "geographic entity"), surpassing
  the intermediate class hierarchy.

* See the notebook assignment.ipynb for some examples

