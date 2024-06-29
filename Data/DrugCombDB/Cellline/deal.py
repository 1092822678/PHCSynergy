import pandas as pd

gene_express = pd.read_csv('Cellline_Expression.csv')

cell_del = gene_express['cell_line_display_name'].to_list()
value = [int(i) for i in range(len(cell_del))]
cell_del_dict = {key: value for key, value in zip(cell_del, value)}
cell_del_reverse = {value:key for key, value in cell_del_dict.items()}
# 读取数据并创建 DataFrame
coloum = ['drug1', 'drug2', 'cell', 'synergy', 'flod']
dds = pd.read_csv('../comb_final.txt', sep=',', header=None)
cell = pd.read_csv('cellline.tsv', sep='\t', header=None)
dds.columns = coloum

ori_dict = cell.set_index(0)[1].to_dict()
ori_dict_reverse = {value:key for key, value in ori_dict.items()}
dds['cell'] = dds['cell'].map(ori_dict)
dds['cell'] = dds['cell'].map(cell_del_dict)

dds = dds.dropna()

dds['cell'] = dds['cell'].map(cell_del_reverse)
dds['cell'] = dds['cell'].map(ori_dict_reverse)

dds['cell'] = dds['cell'].astype(int)
dds.to_csv('dds_del.csv', index=False)
print(dds)

# deal gene_express
gene_express = gene_express.drop('depmap_id', axis=1)
gene_express = gene_express.drop('lineage_1', axis=1)
gene_express = gene_express.drop('lineage_2', axis=1)
gene_express = gene_express.drop('lineage_3', axis=1)
gene_express = gene_express.drop('lineage_4', axis=1)
gene_express = gene_express.drop('lineage_5', axis=1)
gene_express = gene_express.drop('lineage_6', axis=1)
gene_express['cell_line_display_name'] = gene_express['cell_line_display_name'].map(ori_dict_reverse)
gene_express.to_csv('gene_expression.csv',index=False)


