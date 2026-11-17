"""Semantic Search Engine — Streamlit app (AIPI 510, Week 13, Interactive #2).

Placeholder: implemented in step 3, after the Chroma index exists in chromadb/.

Planned sections
---------------
1. Config      : MODEL_NAME, DB path ("chromadb"), COLLECTION_NAME, TOP_K
2. Resources   : cached embedding model + cached Chroma collection
                 (st.cache_resource — load once per process, not per query)
3. Search      : encode query -> collection.query(n_results=TOP_K)
                 -> return documents, metadata, distances -> similarity = 1 - distance
4. UI          : search box (st.text_input) + result cards showing text,
                 source/metadata and score; sidebar with dataset + top-k control
5. Guard rails : empty query, empty results, index missing -> friendly message
"""
