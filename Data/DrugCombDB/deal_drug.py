import pandas as pd

smiles = pd.read_csv('drugbank_smiles.csv')

s_dict = smiles[['drugbank_id','smiles']].set_index('drugbank_id')['smiles'].to_dict()


entities = pd.read_csv('entities.tsv', sep='\t', header=None, names=['Index','DrugBank_id'])
entities.set_index('Index', inplace=True)
drug = entities[0:764]
drug['smiles'] = [0] * len(drug)
drug['smiles'] = drug['DrugBank_id'].map(s_dict)

drug.to_csv('drug_smiles.tsv', sep='\t', index = False)