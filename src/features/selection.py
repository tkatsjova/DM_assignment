import numpy as np

# Feature selection by mutual information between term occurrence and class

def mutual_information(X, y):
    # Mutual information (bits) between the presence of each term and the class label.

    # Same estimate as in the course (slides 32-33): the document-term matrix is made binary
    # (term present / absent) and MI is computed from the 2 x 2 table of counts per term.
    # Written out instead of sklearn's mutual_info_classif, which treats counts as continuous
    # and is slow on large vocabularies. Usable as score_func in SelectKBest.
    present = (X > 0).astype(np.float64)
    y = np.asarray(y)
    n = len(y)
    classes = np.unique(y)

    n_present = np.asarray(present.sum(axis=0)).ravel()  # documents containing the term
    mi = np.zeros(present.shape[1])
    for c in classes:
        n_class = np.sum(y == c)
        n_both = np.asarray(present[y == c].sum(axis=0)).ravel()  # term present and class c
        # the two cells of the 2 x 2 table for class c: term present / term absent
        for n_cell, n_term in ((n_both, n_present), (n_class - n_both, n - n_present)):
            with np.errstate(divide="ignore", invalid="ignore"):
                term = n_cell / n * np.log2(n * n_cell / (n_term * n_class))
            mi += np.nan_to_num(term)  # 0 * log 0 = 0
    return mi
