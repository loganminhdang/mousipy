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
    """Adds the counts of multiple-hit genes to ALL their orthologs."""
    if not multiple:
        return adata

    var = adata.var.copy()
    X = adata.X.copy() if stay_sparse else make_dense(adata.X).copy()
    
    ortholog_indices = {gene: i for i, gene in enumerate(var.index)}
    new_gene_cols = []
    new_gene_vars = []

    fct = tqdm if verbose else lambda item: item
    for source_gene, target_genes in fct(multiple.items()):
        source_gene_data = make_dense(original_data[:, source_gene].X).ravel()
        for target_gene in target_genes:
            if target_gene in ortholog_indices:
                idx = ortholog_indices[target_gene]
                if stay_sparse:
                    X[:, idx] += csr_matrix(source_gene_data).reshape(-1, 1)
                else:
                    X[:, idx] += source_gene_data
            else:
                # Add as a new gene
                ortholog_indices[target_gene] = len(var) + len(new_gene_vars)
                new_gene_cols.append(source_gene_data)
                
                new_row = pd.Series(name=target_gene, dtype='object')
                new_row['original_gene_symbol'] = 'multiple'
                new_gene_vars.append(new_row)

    if new_gene_cols:
        new_vars_df = pd.DataFrame(new_gene_vars)
        var = pd.concat([var, new_vars_df])
        
        if stay_sparse:
            X = csr_matrix(np.hstack([X.toarray()] + [col.reshape(-1, 1) for col in new_gene_cols]))
        else:
            X = np.hstack([X] + [col.reshape(-1, 1) for col in new_gene_cols])

    return AnnData(X, obs=adata.obs, var=var, uns=adata.uns, obsm=adata.obsm)


def collapse_duplicate_genes(adata, stay_sparse=False):
    """Collapse duplicate genes by summing up counts to unique entries."""
    index = adata.var.index
    is_duplicated = index.duplicated(keep=False)
    
    if not np.any(is_duplicated):
        # print('No duplicate genes found. Stopping...')
        return adata

    unique_genes = index[~is_duplicated]
    duplicated_genes = pd.unique(index[is_duplicated])
    
    X = adata.X if stay_sparse else make_dense(adata.X)
    
    # Create new matrix for unique genes
    new_X = X[:, ~is_duplicated]
    new_var = adata.var[~is_duplicated].copy()

    # Process and add collapsed duplicated genes
    dup_cols = []
    dup_vars = []
    
    for gene in tqdm(duplicated_genes, leave=False, desc="Collapsing duplicates"):
        idxs = np.where(index == gene)[0]
        # Sum counts of all duplicates for this gene
        collapsed_counts = X[:, idxs].sum(axis=1)
        dup_cols.append(np.asarray(collapsed_counts).reshape(-1, 1))
        
        # Keep the first var entry for the duplicated gene
        dup_vars.append(adata.var.iloc[idxs[0]].copy())

    if dup_cols:
        new_X = np.hstack([new_X] + dup_cols)
        new_var = pd.concat([new_var, pd.DataFrame(dup_vars)])

    if stay_sparse and not issparse(adata.X):
         X = csr_matrix(new_X)
    else:
        X = new_X

    return AnnData(X, obs=adata.obs, var=new_var, uns=adata.uns, obsm=adata.obsm)


def translate(adata, target=None, stay_sparse=False, verbose=True, source='biomart'):
    """Translates adata.var between species using orthologs from the specified source."""
    source_lower = source.lower()
    if source_lower not in ['hcop', 'biomart']:
        raise ValueError('source must be either "HCOP" or "biomart"')

    direct, multiple, no_hit, no_index = check_orthologs(
        adata.var_names, tab=None, target=target, verbose=verbose, source=source_lower
    )
    if verbose:
        print(f'Found direct orthologs for {len(direct)} genes.')
        print(f'Found multiple orthologs for {len(multiple)} genes.')
        print(f'Found no orthologs for {len(no_hit)} genes.')
        print(f'Found no index in the table for {len(no_index)} genes.')

    bdata = translate_direct(adata, direct, no_index)
    bdata = translate_multiple(bdata, adata, multiple, stay_sparse=stay_sparse, verbose=verbose)
    bdata = collapse_duplicate_genes(bdata, stay_sparse=stay_sparse)
    
    return bdata
