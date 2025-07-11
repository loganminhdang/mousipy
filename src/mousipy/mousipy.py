import os
from re import search

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from tqdm import tqdm

# --- MODIFIED: Added paths for Zebrafish orthology tables ---
# Biomart tables
path = os.path.abspath(os.path.dirname(__file__))
# Mouse-Human tables (original)
h2m_tab_biomart = pd.read_csv(os.path.join(path, './biomart/human_to_mouse.csv')).set_index('Gene name')
m2h_tab_biomart = pd.read_csv(os.path.join(path, './biomart/mouse_to_human.csv')).set_index('Gene name')
# Zebrafish-Human tables (new)
z2h_tab_biomart = pd.read_csv(os.path.join(path, './biomart/zebrafish_to_human.csv')).set_index('Gene name')
h2z_tab_biomart = pd.read_csv(os.path.join(path, './biomart/human_to_zebrafish.csv')).set_index('Gene name')


# HCOP tables (placeholders for potential zebrafish HCOP data)
h2m_tab_hcop = pd.read_csv(os.path.join(path, './hcop/human_to_mouse.csv')).set_index('Gene name')
m2h_tab_hcop = pd.read_csv(os.path.join(path, './hcop/mouse_to_human.csv')).set_index('Gene name')
# z2h_tab_hcop = pd.read_csv(os.path.join(path, './hcop/zebrafish_to_human.csv')).set_index('Gene name') # Placeholder
# h2z_tab_hcop = pd.read_csv(os.path.join(path, './hcop/human_to_zebrafish.csv')).set_index('Gene name') # Placeholder


def make_dense(X):
    """Robustly make an array dense."""
    if issparse(X):
        return X.toarray()
    else:
        return np.asarray(X)


def identify_format_and_organism(txt):
    '''Identify the format and organism of a gene symbol.
    
    Now supports Human, Mouse, and Zebrafish.
    
    Parameters
    ----------
    txt : str
        A gene string.

    Returns
    -------
    format: str
        Either 'Ensembl' or 'Symbol'.
    organism: str
        'Human', 'Mouse', or 'Zebrafish'.
    '''
    # --- MODIFIED: Added Zebrafish Ensembl ID format ---
    if search('^ENSG[0-9]{11}', txt):
        return 'Ensembl', 'Human'
    elif search('^ENSMUSG[0-9]{11}', txt):
        return 'Ensembl', 'Mouse'
    elif search('^ENSDARG[0-9]{11}', txt):
        return 'Ensembl', 'Zebrafish'
    else:
        # --- MODIFIED: Improved organism detection based on typical gene symbol case ---
        # Human gene symbols are typically all uppercase (e.g., 'TP53').
        # Mouse gene symbols are typically title case (e.g., 'Tbx5').
        # Zebrafish gene symbols are typically all lowercase (e.g., 'sox10').
        if txt.isupper():
            organism = 'Human'
        elif txt.islower():
            organism = 'Zebrafish'
        else:
            organism = 'Mouse' # Assumes Title/mixed case is mouse
        return 'Symbol', organism


def check_orthologs(var_names, target=None, tab=None, verbose=False, source='biomart'):
    """Check for orthologs from a list of gene symbols in a biomart table."""
    if target not in ['Ensembl', 'Symbol', None]:
        raise ValueError('target must be either "Ensembl", "Symbol", or None')

    source_lower = source.lower()
    input_format, input_organism = identify_format_and_organism(var_names[0])

    # --- MODIFIED: Handle Zebrafish as an input organism ---
    if input_organism == 'Human':
        output_organism = 'Zebrafish' # Or Mouse, would need more logic for multiple targets
        if source_lower == 'biomart':
            tab = h2z_tab_biomart if not isinstance(tab, pd.DataFrame) else tab
        # elif source_lower == 'hcop':
            # tab = h2z_tab_hcop if not isinstance(tab, pd.DataFrame) else tab
    elif input_organism == 'Mouse':
        output_organism = 'Human'
        if source_lower == 'biomart':
            tab = m2h_tab_biomart if not isinstance(tab, pd.DataFrame) else tab
        elif source_lower == 'hcop':
            tab = m2h_tab_hcop if not isinstance(tab, pd.DataFrame) else tab
    elif input_organism == 'Zebrafish':
        output_organism = 'Human'
        if source_lower == 'biomart':
            tab = z2h_tab_biomart if not isinstance(tab, pd.DataFrame) else tab
        # elif source_lower == 'hcop':
            # tab = z2h_tab_hcop if not isinstance(tab, pd.DataFrame) else tab
    else:
        raise ValueError(f"Unsupported input organism: {input_organism}")

    if not isinstance(tab, pd.DataFrame):
        raise ValueError(f"Orthology table for {input_organism} to {output_organism} not loaded.")

    if input_format == 'Ensembl':
        tab = tab.reset_index().set_index('Gene stable ID')
        target = 'Ensembl' if target is None else target
    else:
        # tab is already indexed by gene symbol
        target = 'Symbol' if target is None else target

    target_key = f'{output_organism} gene stable ID' if target == 'Ensembl' else f'{output_organism} gene name'
    if target_key not in tab.columns:
        raise ValueError(f"Target key '{target_key}' not found in the provided table.")

    direct, multiple, no_hit, no_index = {}, {}, [], []
    fct = tqdm if verbose else lambda x: x
    for gene in fct(var_names):
        if gene in tab.index:
            x = tab.loc[gene, target_key]
            if isinstance(x, pd.Series):
                vals = pd.unique(x.dropna())
                if len(vals) > 1:
                    multiple[gene] = vals
                elif len(vals) == 1:
                    direct[gene] = vals[0]
                else:
                    no_hit.append(gene)
            elif pd.notna(x):
                direct[gene] = x
            else:
                no_hit.append(gene)
        else:
            no_index.append(gene)
            
    return direct, multiple, no_hit, no_index


def translate_direct(adata, direct, no_index):
    """Translate direct hit genes and guess non-indexed genes by uppercasing."""
    _, input_organism = identify_format_and_organism(adata.var_names[0])
    
    # --- MODIFIED: Organism-specific filtering rules ---
    # Define genes to exclude from simple uppercasing "guess"
    if input_organism == 'Mouse':
        # Rules for mouse genes (original)
        guess_genes = [
            x for x in no_index if
            not x.startswith('Gm') and
            'Rik' not in x and
            not x.startswith('RP') and
            'Hist' not in x and
            'Olfr' not in x and
            '.' not in x
        ]
    elif input_organism == 'Zebrafish':
        # Rules for zebrafish genes (new)
        guess_genes = [
            x for x in no_index if
            not x.startswith('si:') and      # Exclude characterized loci
            not x.startswith('zgc:') and     # Exclude zebrafish genome consortium genes
            '.' not in x
        ]
    else:
        guess_genes = no_index # No special rules for other organisms like Human

    # Combine direct hits and guessed genes
    genes_to_keep = list(direct.keys()) + guess_genes
    ndata = adata[:, genes_to_keep].copy()
    ndata.var['original_gene_symbol'] = genes_to_keep
    
    # Create new var_names by translating direct hits and uppercasing guesses
    new_var_names = list(direct.values()) + [g.upper() for g in guess_genes]
    ndata.var_names = new_var_names
    
    return ndata


def translate_multiple(adata, original_data, multiple, stay_sparse=False, verbose=False):
    """
    Adds the counts of multiple-hit genes to ALL their orthologs.
    This robust version handles cases where multiple source genes map to the same new target gene.
    """
    if not multiple:
        return adata

    # --- REVISED LOGIC ---
    # We will work with a dense matrix for easier addition and convert back to sparse at the end if needed.
    X = make_dense(adata.X).copy()
    var = adata.var.copy()
    
    # This lookup table for genes that ALREADY exist in `bdata` will NOT be changed.
    initial_ortholog_indices = {gene: i for i, gene in enumerate(var.index)}

    # This new dictionary will collect data for genes that need to be ADDED.
    # It maps a new gene name to its summed data array.
    new_genes_data = {}

    fct = tqdm(multiple.items(), desc="Handling multiple orthologs") if verbose else multiple.items()
    for source_gene, target_genes in fct:
        source_gene_data = make_dense(original_data[:, source_gene].X).ravel()
        
        for target_gene in target_genes:
            # Case 1: The ortholog already exists in the input AnnData.
            if target_gene in initial_ortholog_indices:
                idx = initial_ortholog_indices[target_gene]
                X[:, idx] += source_gene_data
            
            # Case 2: The ortholog is a new gene that we need to add.
            else:
                # If we're seeing this new gene for the first time, initialize its data.
                if target_gene not in new_genes_data:
                    new_genes_data[target_gene] = source_gene_data.copy()
                # If we've already marked this gene for addition, just add the new counts.
                else:
                    new_genes_data[target_gene] += source_gene_data

    # After the loop, if we collected any new genes, add them to X and var.
    if new_genes_data:
        new_var_rows = []
        new_X_cols = []
        for gene_name, data_col in new_genes_data.items():
            new_X_cols.append(data_col.reshape(-1, 1))
            
            new_row = pd.Series(name=gene_name, dtype='object')
            new_row['original_gene_symbol'] = 'multiple'
            new_var_rows.append(new_row)

        # Combine new columns and rows with existing data
        if new_X_cols:
            X = np.hstack([X] + new_X_cols)
            var = pd.concat([var, pd.DataFrame(new_var_rows)])
            
    # Re-apply sparsity if it was requested at the beginning.
    if stay_sparse:
        X = csr_matrix(X)

    return AnnData(X, obs=adata.obs, var=var, uns=adata.uns, obsm=adata.obsm)


def collapse_duplicate_genes(adata, stay_sparse=False):
    """Collapse duplicate genes by summing up counts to unique entries."""
    index = adata.var.index
    is_duplicated = index.duplicated(keep=False)
    
    if not np.any(is_duplicated):
        return adata

    # Separate non-duplicated from duplicated genes
    unique_mask = ~is_duplicated
    duplicated_genes = pd.unique(index[is_duplicated])
    
    # Start with the data and metadata from non-duplicated genes
    new_X = adata.X[:, unique_mask]
    new_var = adata.var[unique_mask].copy()

    # Process duplicated genes
    dup_cols = []
    dup_vars = []
    
    for gene in tqdm(duplicated_genes, leave=False, desc="Collapsing duplicates"):
        idxs = np.where(index == gene)[0]
        
        # Sum counts across all duplicates for the current gene
        if issparse(adata.X):
            collapsed_counts = adata.X[:, idxs].sum(axis=1)
        else:
            collapsed_counts = np.sum(adata.X[:, idxs], axis=1, keepdims=True)
            
        dup_cols.append(collapsed_counts)
        # Keep the metadata from the first occurrence of the duplicate
        dup_vars.append(adata.var.iloc[idxs[0]].copy())

    # Combine non-duplicated data with the newly collapsed duplicated data
    if dup_cols:
        if issparse(new_X):
            from scipy.sparse import hstack
            new_X = hstack([new_X] + dup_cols).tocsr()
        else:
            new_X = np.hstack([new_X] + dup_cols)
        
        # Combine the metadata DataFrames
        new_var = pd.concat([new_var, pd.DataFrame(dup_vars)])
    
    # --- BUG FIX ---
    # The original code mistakenly returned the old `X`. 
    # The corrected code returns the newly constructed `new_X`.
    return AnnData(new_X, obs=adata.obs, var=new_var, uns=adata.uns, obsm=adata.obsm)


def translate(adata, target=None, stay_sparse=False, verbose=True, source='biomart'):
    """
    Translates adata.var between species using orthologs from the specified source.
    This corrected version handles duplicates at each stage to prevent index errors.
    """
    source_lower = source.lower()
    if source_lower not in ['hcop', 'biomart']:
        raise ValueError('source must be either "HCOP" or "biomart"')

    # 1. Check for all orthologs
    direct, multiple, no_hit, no_index = check_orthologs(
        adata.var_names, tab=None, target=target, verbose=verbose, source=source_lower
    )
    if verbose:
        print(f'Found direct orthologs for {len(direct)} genes.')
        print(f'Found multiple orthologs for {len(multiple)} genes.')
        print(f'Found no orthologs for {len(no_hit)} genes.')
        print(f'Found no index in the table for {len(no_index)} genes.')

    # 2. Translate direct hits and simple guesses
    bdata = translate_direct(adata, direct, no_index)

    # 3. Collapse duplicates created by translate_direct BEFORE proceeding
    bdata = collapse_duplicate_genes(bdata, stay_sparse=stay_sparse)

    # 4. Add data from genes with multiple orthologs
    bdata = translate_multiple(bdata, adata, multiple, stay_sparse=stay_sparse, verbose=verbose)

    # 5. Collapse any new duplicates that may have been introduced by translate_multiple
    bdata = collapse_duplicate_genes(bdata, stay_sparse=stay_sparse)
    
    return bdata
