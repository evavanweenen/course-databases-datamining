import numpy as np
import gzip

from pyrwr.ppr import PPR #personalized pagerank
#import networkx as nx

import goatools.obo_parser as obo
from goatools.go_enrichment import GOEnrichmentStudy as GO_en
from Bio.UniProt.GOA import gafiterator

#from bioservices.kegg import KEGG
#from bioservices import *

#import sharepathway as sp


c = 0.5 #restart/jumping probability
epsilon = 1e-6 #error tolerance for power iteration
max_iters = 100
PPR_thres = 1e-5#threshold for Random Walk with Restart
perm_thres = 0.05#threshold for permutation test
int_thres = 900#threshold for interaction test
enrich_thres = 0.8 # threshold for GO-enrichment test
np.random.seed(42)

def in_set(X, ref):
    count = 0
    for i in ref:
        if len(np.where(X == i)[0]) !=0:
            count += 1
    return count

def to_ensembl_string(IDs):
    stringIDs = []
    for i in IDs:
        a = 'ENSP' 
        for j in range(1, 12-len(str(i))):
            a = a + '0'
        a = a + str(i)
        stringIDs.append(a)
    return stringIDs

def to_ensembl_int(IDs):
    intIDs = list(map(lambda x: int(x.replace('ENSP','')), IDs))
    return np.array(intIDs).astype('<i8')

#----------------------------------------Read data--------------------------------------------------
"""
#read ppi network and remove 9606.ENSP string
N = np.genfromtxt(fname='9606.protein.links.v10.5.txt', names=True, dtype=['U20', 'U20', '<i8'])#, max_rows=10)
N['protein1'] = list(map(lambda x: int(x.replace('9606.ENSP','')), N['protein1']))
N['protein2'] = list(map(lambda x: int(x.replace('9606.ENSP','')), N['protein2']))
N = N.astype([('protein1', '<i8'), ('protein2', '<i8'), ('combined_score', '<i8')])
np.savetxt('N.csv', N, delimiter=',', fmt='%d %d %d')

#rewrite indices of N
N = np.genfromtxt(fname='N.csv', delimiter=' ', skip_header=1, dtype=None)
uni, inv = np.unique(N[:,:2], return_inverse=True)

np.savetxt('N_inv.csv', np.concatenate((inv.reshape(N[:,:2].shape),N[:,2].reshape(len(N),1)), axis=1))
np.savetxt('N_uni_flat.csv', uni)
"""

#load N,S, G, V
S = np.genfromtxt(fname='S.csv', names=True, delimiter=',', dtype=['<U10', '<i8'])
G = np.genfromtxt(fname='G.csv', names=True, delimiter=',', dtype=['<i8', 'U10', '<f8', '<f8', '<i8', '<f8'])
V = np.genfromtxt(fname='V.csv', names=True, delimiter=',', dtype=['U10', '<i8'])

#----------------------------------------Preprocessing----------------------------------------------
#create mapping
uni = np.genfromtxt(fname='N_uni_flat.csv').astype('<i8')
mapping = {j : i for i,j in enumerate(uni)}
inv_mapping = {i : j for i,j in enumerate(uni)}

#delete values of S, G and V that are not in N
deletelist = np.genfromtxt(fname='not_in_s.txt').astype('<i8')
for d in deletelist:
    S = np.delete(S, (np.where(S['Ensembl_ID']==d)[0][0]), axis=0)

G = np.delete(G, (np.where(G['Ensembl_ID']==379625)[0][0]), axis=0)
V = np.delete(V, (np.where(V['Ensembl_ID']==379625)[0][0]), axis=0)

#rewrite indices of S, G, V
new_S = list(map(lambda x: mapping[x], S['Ensembl_ID']))
new_G = list(map(lambda x: mapping[x], G['Ensembl_ID']))
new_V = list(map(lambda x: mapping[x], V['Ensembl_ID']))

#----------------------------------------PPR algorithm----------------------------------------------
#PPR
ppr = PPR()
ppr.read_graph('N_inv.csv', graph_type='directed')
r_ppr = ppr.compute(new_S, c, epsilon, max_iters)
np.savetxt('R_PPR.csv', r_ppr)

#Select the proteins that come above the score of 1e-5 as candidates
C_PPR_prob = r_ppr[r_ppr > PPR_thres]
C_PPR_proteins = np.where(r_ppr > PPR_thres)[0]
C_PPR_ensembl = list(map(lambda x: inv_mapping[x], C_PPR_proteins))

#Save tables
C_PPR = np.vstack((C_PPR_ensembl, C_PPR_proteins, C_PPR_prob)).T
C_PPR_s = np.core.records.fromarrays(C_PPR.transpose(), names='Ensembl_ID, protein, probability', formats='<i8, <i8, <f8')
C_PPR_s.sort(order='probability')
C_PPR_s = C_PPR_s[::-1]
np.savetxt('results/C_PPR.csv', C_PPR_s, fmt='%.u %.u %.2e')

#Compare with previously known sets
print("Number of candidates from PPR algorithm: ", len(C_PPR_proteins))
print("Number of candidates from PPR algorithm in S: ", in_set(C_PPR_proteins, new_S), " of ", len(new_S))
print("Number of candidates from PPR algorithm in G: ", in_set(C_PPR_proteins, new_G), " of ", len(new_G))
print("Number of candidates from PPR algorithm in V: ", in_set(C_PPR_proteins, new_V), " of ", len(new_V))

#----------------------------------------Permutation test-------------------------------------------
#permutation test
p_value_vector = np.zeros(len(r_ppr))

for i in range(0,1000):
    seeds = np.random.choice(np.arange(len(r_ppr)), len(new_S))
    random_ppr = ppr.compute(seeds, c, epsilon, max_iters)
    p_value_vector[[True if (random_ppr[j] > r_ppr[j]) else False for j in range(len(random_ppr))]] += 1/1000

#reduce p_value_vector to only genes with prob > ppr_thres
p_value_vector = p_value_vector.reshape(len(p_value_vector),1)
p_value_vector = p_value_vector[r_ppr > PPR_thres]

#select only genes with p-value lower than perm_thres
C_perm_pvalue = p_value_vector[p_value_vector < perm_thres]
C_perm_prob = C_PPR_prob[p_value_vector < perm_thres]
C_perm_proteins = C_PPR_proteins[p_value_vector < perm_thres]
C_perm_ensembl = list(map(lambda x: inv_mapping[x], C_perm_proteins))

#Save tables
C_perm = np.vstack((C_perm_ensembl, C_perm_proteins, C_perm_prob, C_perm_pvalue)).T
C_perm_s = np.core.records.fromarrays(C_perm.transpose(), names='Ensembl_ID, protein, probability, p-value', formats='<i8, <i8, <f8, <f8')
C_perm_s.sort(order='probability')
C_perm_s = C_perm_s[::-1]
C_perm_s.sort(order='p-value')
np.savetxt('results/C_perm.csv', C_perm_s, fmt='%.u %.u %.2e %.2e')

#Compare with previously known sets
print("Number of candidates from permutation test: ", len(C_perm_proteins))
print("Number of candidates from permutation test in S: ", in_set(C_perm_proteins, new_S), " of ", len(new_S))
print("Number of candidates from permutation test in G: ", in_set(C_perm_proteins, new_G), " of ", len(new_G))
print("Number of candidates from permutation test in V: ", in_set(C_perm_proteins, new_V), " of ", len(new_V))


#----------------------------------------Interaction test-------------------------------------------
#interaction test
C_int_mask = np.array([False for i in C_perm_proteins])
score = np.zeros(len(C_int_mask))
for index, i in enumerate(C_perm_proteins):
    maximum = 0
    for j in new_S:
        maximum = np.max((ppr.A[i,j], ppr.A[j,i],maximum))
    
    score[index] = maximum
    if maximum > int_thres:
        C_int_mask[index] = True

#select genes with mask
C_int_proteins = C_perm_proteins[C_int_mask]
C_int_prob = C_perm_prob[C_int_mask]
C_int_pvalue = C_perm_pvalue[C_int_mask]
C_int_ensembl = list(map(lambda x: inv_mapping[x], C_int_proteins))
C_int_score = score[C_int_mask]

#Save tables     
C_int = np.vstack((C_int_ensembl, C_int_proteins, C_int_prob, C_int_pvalue, C_int_score)).T
C_int_s = np.core.records.fromarrays(C_int.transpose(), names='Ensembl_ID, protein, probability, p-value, score', formats='<i8, <i8, <f8, <f8, <f8')

C_int_s.sort(order='probability')
C_int_s = C_int_s[::-1]

np.savetxt('results/C_int.csv', C_int_s, fmt='%.u %.u %.2e %.2e %.u')

#Compare with previously known sets
print("Number of candidates from interaction test: ", len(C_int_proteins))    
print("Number of candidates from interaction test in S: ", in_set(C_int_proteins, new_S), " of ", len(new_S))
print("Number of candidates from interaction test in G: ", in_set(C_int_proteins, new_G), " of ", len(new_G))
print("Number of candidates from interaction test in V: ", in_set(C_int_proteins, new_V), " of ", len(new_V))


C_int_100 = C_int_s[:100]
#Compare with previously known sets
print("Number of candidates from interaction test: ", len(C_int_100))    
print("Number of candidates from interaction test in S: ", in_set(C_int_100['protein'], new_S), " of ", len(new_S))
print("Number of candidates from interaction test in G: ", in_set(C_int_100['protein'], new_G), " of ", len(new_G))
print("Number of candidates from interaction test in V: ", in_set(C_int_100['protein'], new_V), " of ", len(new_V))

np.savetxt('results/C_int_100.csv', C_int_100, fmt='%.u %.u %.2e %.2e %.u')


##---------------------------------------- GO Enrichment test-------------------------------------------- 
EnsemblIDs = to_ensembl_string(C_int_ensembl)
np.savetxt('proteins.txt', EnsemblIDs, fmt='%.s')

#convert ID's to UniProtKB (https://www.uniprot.org/uploadlists/); saved as "UniProtIDs.csv"; proteins not able to convert: 'not_in_proteins.txt'
UP_ID = np.genfromtxt(fname='UniProtIDs.csv', names=True, delimiter=',', dtype=['U15','U6','U25','U25','U25','U25'])
ensembl_to_up = {j : UP_ID['UniProtID'][i] for i, j in enumerate(UP_ID['EnsemblID'])}
up_to_ensembl = {UP_ID['UniProtID'][i] : j for i, j in enumerate(UP_ID['EnsemblID'])}


newEnsemblIDs = []
for i in EnsemblIDs:
    if i in UP_ID['EnsemblID']:
        newEnsemblIDs.append(i)

C_int_UP = list(map(lambda x: ensembl_to_up[x], newEnsemblIDs))


#Enrichment Analysis
go = obo.GODag('/disks/strw13/DBDM/A4_2/go-basic.obo')

with gzip.open('goa_human.gaf.gz', 'rt') as fp:
    funcs = {}
    for entry in gafiterator(fp):
        uniprot_id = entry.pop('DB_Object_ID')
        funcs[uniprot_id] = entry

pop = funcs.keys()
assoc = {}

for x in funcs:
    if x not in assoc:
        assoc[x] = set()
    assoc[x].add(str(funcs[x]['GO_ID']))
    
dictionary = {x: funcs[x]
               for x in funcs 
               if x in C_int_UP}

GO_IDs = {x: assoc[x]
         for x in assoc 
         if x in C_int_UP}

         
study = dictionary.keys()  

g = GO_en(pop, assoc, go,
          propagate_counts=True,
          alpha=0.05,
          methods=["bonferroni", "sidak", "holm", "fdr"])
g_res = g.run_study(study)


#Select GO terms based on Bonferroni Correction
s_bonferroni = []
s_fdr = []
for x in g_res:
    if x.p_bonferroni <= 0.01:
        s_bonferroni.append((x.goterm.id, x.p_bonferroni))
    if x.p_fdr <= 0.01:
        s_fdr.append((x.goterm.id, x.p_fdr))
        
enriched_GO_ID_bon = [i[0] for i in s_bonferroni]
enriched_GO_ID_fdr = [i[0] for i in s_fdr]

#Only select genes with GO terms that are enriched
C_GO_UP= []
for i in GO_IDs:
    for ID in GO_IDs[i]:
        if ID in enriched_GO_ID_bon:
            C_GO_UP.append(i)
C_GO_UP = set(C_GO_UP)
    
C_GO_ensembl = list(map(lambda x: up_to_ensembl[x], C_GO_UP))
C_GO_ensembl = to_ensembl_int(C_GO_ensembl)

C_GO_mask = np.array([False for i in C_int_ensembl])
for index, i in enumerate(C_GO_ensembl):
    C_GO_mask[np.where(C_int_ensembl==i)[0]] = True


C_GO_proteins = C_int_proteins[C_GO_mask]
C_GO_prob = C_int_prob[C_GO_mask]
C_GO_pvalue = C_int_pvalue[C_GO_mask]
C_GO_score = C_int_score[C_GO_mask]

print("Number of candidates from enrichment test: ", len(C_GO_proteins))   
print("Number of candidates from enrichment test in S: ", in_set(C_GO_proteins, new_S), " of ", len(new_S))
print("Number of candidates from enrichment test in G: ", in_set(C_GO_proteins, new_G), " of ", len(new_G))
print("Number of candidates from enrichment test in V: ", in_set(C_GO_proteins, new_V), " of ", len(new_V))


#Save tables     
C_GO = np.vstack((C_GO_ensembl, C_GO_proteins, C_GO_prob, C_GO_pvalue, C_GO_score)).T
C_GO_s = np.core.records.fromarrays(C_GO.transpose(), names='Ensembl_ID, protein, probability, p-value-perm, score', formats='<i8, <i8, <f8, <f8, <f8')

C_GO_s.sort(order='probability')
C_GO_s = C_GO_s[::-1]

np.savetxt('results/C_GO.csv', C_GO_s, fmt='%.u %.u %.2e %.2e %.u')

C = C_GO_s[:100]
#Compare with previously known sets
print("Number of candidates from enrichment  test: ", len(C))    
print("Number of candidates from enrichment  test in S: ", in_set(C['protein'], new_S), " of ", len(new_S))
print("Number of candidates from enrichment  test in G: ", in_set(C['protein'], new_G), " of ", len(new_G))
print("Number of candidates from enrichment  test in V: ", in_set(C['protein'], new_V), " of ", len(new_V))


np.savetxt('results/C_GO_100.csv', C, fmt='%.u %.u %.2e %.2e %.u')
np.savetxt('results/C.txt', np.array(to_ensembl_string(C['Ensembl_ID'])))


new_G = np.array(new_G)
name_G = list(map(lambda x: inv_mapping[x], new_G))

pos_G = np.zeros_like(new_G)
pos_G_int = np.zeros_like(new_G)
for index, i in enumerate(new_G):
    loc = np.where(C['protein'] == i)[0]
    loc_int = np.where(C_int_100['protein'] == i)[0]
    if len(loc) != 0:
        pos_G[index] = loc[0]
    if len(loc_int) != 0:
        pos_G_int[index] = loc_int[0]


#----------------------------------------Plot network-------------------------------------------- 

"""
C_int_interactions = ppr.A
C_int_interactions = C_int_interactions[C_int_proteins,:].tocsc()
C_int_interactions = C_int_interactions[:,C_int_proteins]

C_interactions = ppr.A
C_interactions = C_interactions[C['protein'],:].tocsc()
C_interactions = C_interactions[:,C['protein']]



graph = nx.from_scipy_sparse_matrix(ppr.A)
nx.draw(graph)
"""


#----------------------------------------KEGG enrichment-------------------------------------------- 
'''
#KEGG Enrichment
#https://bioservices.readthedocs.io/en/master/kegg_tutorial.html
k = KEGG()
genes = UP_ID['Gene_names_primary']
save_paths = []
skipped_genes = []
skipped_paths = []

for i,gene in enumerate(genes):
    try:
        paths = k.get_pathway_by_gene(gene, 'hsa')
        print('skip gene ' + gene)
    except:
        skipped_genes.append(i)
    try:
        for j, path in enumerate(paths):
            a = k.get(path)
            b = k.parse(a)
            genes_in_path = b['GENE']
    except:    
        skipped_paths.append((i, j))
        print('skip path ' + path + ' of gene ' + gene)

#TODO make "genelist file" see: https://pypi.org/project/sharepathway/
filein="genelists.txt"
fileout="result"
sp.Run(fi=filein,fo=fileout,species='hsa',r=0.1)
'''