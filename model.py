import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
import dhg
from dhg.nn import HGNNPConv

cellline = 76
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
class HGNNP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate),
        )
        self.layers.append(
            HGNNPConv(hid_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate),
            # HGNNPConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        for layer in self.layers:
            X = layer(X, hg)
        return X

class PHCSynergy(nn.Module):

    def __init__(self, config, entity_pre_embed, structure_pre_embed, gene_expression, cell_map, hgnn_1, hgnn_2):

        super(PHCSynergy, self).__init__()

        # embedding data
        self.hgnn_drug = hgnn_1
        self.hgnn_cell = hgnn_2
        self.gene = gene_expression
        self.cell_map = cell_map
        self.gene_dim = self.gene.shape[1]
        self.structure_pre_embed = structure_pre_embed
        self.entity_pre_embed = entity_pre_embed
        self.n_approved_drug = structure_pre_embed.shape[0]

        # embedding setting
        self.structure_dim = self.structure_pre_embed.shape[1]
        self.e_dim = self.entity_pre_embed.shape[1]
        self.hyper_dim = config['model']['args']['hyper_dim']
        
        self.entity_dim = config['model']['args']['e_dim']
        self.cell_dim = config['model']['args']['cell_dim']

        self.eps = config['model']['args']['EMB_INIT_EPS']

        # drug layers
        self.druglayer_structure = nn.Linear(self.structure_dim, self.entity_dim)
        self.druglayer_KG = nn.Linear(self.e_dim, self.entity_dim)
    
        self.add_drug = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim))
        self.cross_add_drug = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim))
        
        self.multi_drug = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim))
        self.activate = nn.ReLU()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=config['model']['args']['in_channels'], out_channels=config['model']['args']['out_channels'], 
                      kernel_size=(config['model']['args']['kernel'], config['model']['args']['kernel'])),
            nn.BatchNorm2d(config['model']['args']['out_channels']), nn.MaxPool2d((config['model']['args']['pooling_size'], config['model']['args']['pooling_size'])), 
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=config['model']['args']['out_channels'], out_channels=config['model']['args']['out_channels'], 
                      kernel_size=(config['model']['args']['kernel'], config['model']['args']['kernel'])),
            nn.BatchNorm2d(config['model']['args']['out_channels']), nn.MaxPool2d((config['model']['args']['pooling_size'], config['model']['args']['pooling_size'])), 
            nn.ReLU())
        
        self.conv1_out = (self.entity_dim - config['model']['args']['kernel'] + 1)/2
        self.conv2_out = (self.conv1_out - config['model']['args']['kernel'] + 1)/2

        self.fc1 = nn.Sequential(nn.Linear(int(self.conv2_out * self.conv2_out * config['model']['args']['out_channels']), self.entity_dim), 
                                    nn.BatchNorm1d(self.entity_dim),
                                    nn.ReLU(True))

        self.fc2_global = nn.Sequential(
            nn.Linear(self.entity_dim * self.entity_dim + self.entity_dim, self.entity_dim),
            nn.ReLU(True))
        self.fc2_global_reverse = nn.Sequential(
            nn.Linear(self.entity_dim * self.entity_dim + self.entity_dim, self.entity_dim),
            nn.ReLU(True))
        self.fc2_cross = nn.Sequential(
            nn.Linear(self.entity_dim * 4, self.entity_dim),
            nn.ReLU(True))

        self.all_embedding_dim = (self.entity_dim * 2 + self.structure_dim + self.e_dim + self.hyper_dim) * 2
    
        
        self.celllayer_gene = nn.Linear(self.gene_dim, self.entity_dim)
        self.celllayer_KG = nn.Linear(self.e_dim, self.entity_dim)
        
        self.c_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=config['model']['args']['in_channels'], out_channels=config['model']['args']['out_channels'], 
                      kernel_size=(config['model']['args']['kernel'], config['model']['args']['kernel'])),
            nn.BatchNorm2d(config['model']['args']['out_channels']), nn.MaxPool2d((config['model']['args']['pooling_size'], config['model']['args']['pooling_size'])), 
            nn.ReLU())

        self.c_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=config['model']['args']['out_channels'], out_channels=config['model']['args']['out_channels'], 
                      kernel_size=(config['model']['args']['kernel'], config['model']['args']['kernel'])),
            nn.BatchNorm2d(config['model']['args']['out_channels']), nn.MaxPool2d((config['model']['args']['pooling_size'], config['model']['args']['pooling_size'])), 
            nn.ReLU())
        
        self.c_fc1 = nn.Sequential(nn.Linear(int(self.conv2_out * self.conv2_out * config['model']['args']['out_channels']), self.entity_dim), 
                                    nn.BatchNorm1d(self.entity_dim),
                                    nn.ReLU(True))

        self.c_fc2_global = nn.Sequential(
            nn.Linear(self.entity_dim * self.entity_dim + self.entity_dim, self.entity_dim),
            nn.ReLU(True))
        self.c_fc2_global_reverse = nn.Sequential(
            nn.Linear(self.entity_dim * self.entity_dim + self.entity_dim, self.entity_dim),
            nn.ReLU(True))

        self.multi_cell = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim))
        self.activate = nn.ReLU()
        
        # decoder layers
        self.layer1 = nn.Sequential(nn.Linear(6310, config['model']['args']['n_hidden_1']), nn.BatchNorm1d(config['model']['args']['n_hidden_1']),
                                    nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(config['model']['args']['n_hidden_1'], config['model']['args']['n_hidden_2']), 
                                    nn.BatchNorm1d(config['model']['args']['n_hidden_2']),
                                    nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(config['model']['args']['n_hidden_2'], 1))

    def generate_fusion_feature(self, fts, G):
        
        self.entity_embed_pre = self.entity_pre_embed[:self.n_approved_drug, :]
        structure = self.druglayer_structure(self.structure_pre_embed)
        entity = self.druglayer_KG(self.entity_embed_pre)

        structure_embed_reshape = structure.unsqueeze(-1)  # batch_size * embed_dim * 1
        entity_embed_reshape = entity.unsqueeze(-1)  # batch_size * embed_dim * 1

        entity_matrix = structure_embed_reshape * entity_embed_reshape.permute(
            (0, 2, 1))  # batch_size * embed_dim * embed_dim
        entity_matrix_reverse = entity_embed_reshape * structure_embed_reshape.permute(
            (0, 2, 1))  # batch_size * embed_dim * embed_dim

        entity_global = entity_matrix.view(entity_matrix.size(0), -1)
        entity_global_reverse = entity_matrix_reverse.view(entity_matrix.size(0), -1)

        entity_matrix_reshape = entity_matrix.unsqueeze(1)
        entity_data = entity_matrix_reshape
        entity_matrix_reshape_reverse = entity_matrix_reverse.unsqueeze(1)
        entity_reverse = entity_matrix_reshape_reverse

        out = self.conv1(entity_data)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        out2 = self.conv1(entity_reverse)
        out2 = self.conv2(out2)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.fc1(out2)

        global_local_before = torch.cat((out, entity_global), 1)
        cross_embedding_pre = self.fc2_global(global_local_before)
        
        global_local_before_reverse = torch.cat((out2, entity_global_reverse), 1)
        cross_embedding_pre_reverse = self.fc2_global_reverse(global_local_before_reverse)
        
        out3 = self.hgnn_drug(fts, G)
 
        out_concat = torch.cat(
                (self.structure_pre_embed, self.entity_embed_pre, cross_embedding_pre, cross_embedding_pre_reverse, out3), 1)
   
        return out_concat

    
    def forward(self, *input):
        # drug
        self.drug_embed = self.generate_fusion_feature(input[3], input[4])
        drug1_embed = self.drug_embed[input[0]]
        drug2_embed = self.drug_embed[input[1]]
        drug_data = torch.cat((drug1_embed, drug2_embed), 1)
        
        # cell
        # gene = torch.tensor(np.array(self.gene.loc[input[2].cpu()]).astype('float32')).to(device)
        gene = torch.tensor(np.array(self.gene).astype('float32')).to(device)
        cell_kg_embded = self.celllayer_KG(self.entity_pre_embed[list(self.cell_map.keys())])
        cell_gene_emebed = self.celllayer_gene(gene)
       
        gene_embed_reshape = cell_gene_emebed.unsqueeze(-1)  # batch_size * embed_dim * 1
        kg_embed_reshape = cell_kg_embded.unsqueeze(-1)  # batch_size * embed_dim * 1
        
        cell_entity_matrix = gene_embed_reshape * kg_embed_reshape.permute(
            (0, 2, 1))  # batch_size * embed_dim * embed_dim
        cell_entity_matrix_reverse = kg_embed_reshape * gene_embed_reshape.permute(
            (0, 2, 1))  # batch_size * embed_dim * embed_dim
        
        cell_entity_global = cell_entity_matrix.view(cell_entity_matrix.size(0), -1)
        cell_entity_global_reverse = cell_entity_matrix_reverse.view(cell_entity_matrix.size(0), -1)

        cell_entity_matrix_reshape = cell_entity_matrix.unsqueeze(1)
        cell_entity_data = cell_entity_matrix_reshape
        cell_entity_matrix_reshape_reverse = cell_entity_matrix_reverse.unsqueeze(1)
        cell_entity_reverse = cell_entity_matrix_reshape_reverse

        c_out = self.c_conv1(cell_entity_data)
        c_out = self.c_conv2(c_out)
        c_out = c_out.view(c_out.size(0), -1)
        c_out = self.c_fc1(c_out)

        c_out2 = self.c_conv1(cell_entity_reverse)
        c_out2 = self.c_conv2(c_out2)
        c_out2 = c_out2.view(c_out2.size(0), -1)
        c_out2 = self.c_fc1(c_out2)
        
        cell_global_local_before = torch.cat((c_out, cell_entity_global), 1)
        cell_cross_embedding_pre = self.c_fc2_global(cell_global_local_before)
        
        cell_global_local_before_reverse = torch.cat((c_out2, cell_entity_global_reverse), 1)
        cell_cross_embedding_pre_reverse = self.c_fc2_global_reverse(cell_global_local_before_reverse)
        
        # c_out3 = self.activate(self.multi_drug(cell_kg_embded * cell_gene_emebed))
        c_out3 = self.hgnn_cell(input[5], input[6])
        cell_out_concat = torch.cat(
                (gene, self.entity_pre_embed[list(self.cell_map.keys())], cell_cross_embedding_pre, cell_cross_embedding_pre_reverse, c_out3), 1)

        # drug1_embed = self.drug_embed[input[0]]
        cell = torch.tensor([self.cell_map[element.item()] for element in input[2]])
        cell_embed = cell_out_concat[cell]
        
        # cell_kg_final = self.cellline(cell_kg_embded)
        all_emb = torch.cat((drug_data, cell_embed), 1)
        x = self.layer1(all_emb)
        x = self.layer2(x)
        x = self.layer3(x)
        return torch.sigmoid(x).squeeze()
    

