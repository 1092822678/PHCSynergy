import argparse
import random
import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, accuracy_score, roc_curve
from tqdm import tqdm  # 产生进度条
import load_data
from model import PHCSynergy, HGNNP
import copy
from collections import OrderedDict
from utils import evaluate
from utils import hypergraph_utils as hgut
from dhg import Hypergraph
import dhg

from torch.utils.data import DataLoader
import sklearn.metrics as m
import math
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_classification(labels, logits):
    logits = logits.to('cpu')
    auc = roc_auc_score(y_true=labels, y_score=logits)
    p, r, t = precision_recall_curve(y_true=labels, probas_pred=logits)
    aupr = m.auc(r, p)
    fpr, tpr, threshold = roc_curve(labels, logits)
    # 利用Youden's index计算阈值
    spc = 1 - fpr
    j_scores = tpr - fpr
    best_youden, youden_thresh, youden_sen, youden_spc = sorted(zip(j_scores, threshold, tpr, spc))[-1]
    predicted_label = copy.deepcopy(logits)
    youden_thresh = round(youden_thresh, 3)
    print(youden_thresh)

    predicted_label = [1 if i >= youden_thresh else 0 for i in predicted_label]
    p_1 = evaluate.precision(y_true=labels, y_pred=predicted_label)
    r_1 = evaluate.recall(y_true=labels, y_pred=predicted_label)
    acc = accuracy_score(y_true=labels, y_pred=predicted_label)
    f1 = f1_score(y_true=labels, y_pred=predicted_label)
    return p_1, r_1, acc, auc, aupr, f1

def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            k_ = k[7:]  # remove 'module.' of DistributedDataParallel instance
            state_dict[k_] = v
        model.load_state_dict(state_dict)

    model.eval()
    return model

def train():
    now = time.strftime("%Y-%m-%d-%H_%M", time.localtime(time.time()))
    hours = time.strftime("%Y-%m-%d-%H_%M", time.localtime(time.time()))

    parser = argparse.ArgumentParser()
    parser.add_argument('--CV', type=int, default=1, help='the number of CV')
    parser.add_argument('-c', '--config_file', default='config/DrugCombDB_config.json', type=str,
                      help='config file path (default: None)')

    parser.add_argument('-d', '--device', default='1', type=str,
                      help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    auc_all = []
    aupr_all = []
    acc_all = []
    for cv in range(1):
        auc_cv = []
        aupr_cv = []
        acc_cv = []
        drug1, drug2, cells, triples = load_data.readRecData(config['data']['args']['data_dir']) 
        
        # Load Pretrained Data
        print('KG-embedding loading...')
        transE_entity_data = np.load(os.path.join(config['data']['args']['data_dir'], 'KG_TransR_l2_entity.npy'))
        print('Graph-embedding loading...')
        masking_entity_data = np.load(os.path.join(config['data']['args']['data_dir'], 'drug_smiles.npy'))
        print('Cellline expression loading...')
        gene_expression = pd.read_csv(os.path.join(config['data']['args']['data_dir'], 'Cellline/expression_final.csv'))
        values = [int(i) for i in range(len(gene_expression))]
        cell_map = {key:value for key, value in zip(gene_expression['cell_line_display_name'], values)}   
        gene_expression = gene_expression.set_index('cell_line_display_name')
        
        
        print('Loading pretrain data down!')
        
        drug_num = masking_entity_data.shape[0]
        cell_num = gene_expression.shape[0]
        merge_feat_drug = torch.tensor(np.concatenate((transE_entity_data[:drug_num, :], masking_entity_data), 1))
        hg_drug = dhg.Hypergraph.from_feature_kNN(merge_feat_drug, 5)
        
        
        cell_del_feat = transE_entity_data[list(cell_map.keys())]
        merge_feat_cell = torch.tensor(np.concatenate((cell_del_feat, np.array(gene_expression).astype('float32')), 1))
        hg_cell = dhg.Hypergraph.from_feature_kNN(merge_feat_cell, 5)
        
        merge_feat_drug = merge_feat_drug.to(device)
        merge_feat_cell = merge_feat_cell.to(device)
        hg_drug = hg_drug.to(device)
        hg_cell = hg_cell.to(device)
        
        structure_pre_embed = torch.tensor(masking_entity_data).to(device)
        entity_pre_embed = torch.tensor(transE_entity_data).to(device).float()
        
        loss_fcn = nn.BCELoss()
        np.random.seed(23)
        random.shuffle(triples)
        triples_DF = pd.DataFrame(triples) 
        test_fold = 0
        
        #五折交叉验证
        for i in range(5):
            auc_kfold, aupr_kfold, acc_kfold = [], [], []
            idx_test = np.where(triples_DF[4] == test_fold)
            idx_train = np.where(triples_DF[4] != test_fold)
            
            test_set = [triples[xx] for xx in idx_test[0]]
            train_set = [triples[xx] for xx in idx_train[0]]

            print("batch = {}".format(len(train_set) // config['data']['args']['batch_size'])) 
            
            # construct model & optimizer
            model = PHCSynergy(config, entity_pre_embed, structure_pre_embed, gene_expression, cell_map,
                                HGNNP(in_channels=merge_feat_drug.shape[1],
                                hid_channels=config['model']['args']['hyper_dim']),
                                HGNNP(in_channels=merge_feat_cell.shape[1],
                                hid_channels=config['model']['args']['hyper_dim'])).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer']['args']['lr'], 
                                         weight_decay=config['optimizer']['args']['weight_decay'])

            if config['train_test_mode'] == 0:
                model.load_state_dict(torch.load('model/2024-03-19-06_38_decoder0.pkl'))
                
            if config['train_test_mode'] == 1:
                t_total = time.time()
                for e in range(config['n_epochs']):
                    t = time.time()
                    model.train()
                    all_loss = 0.0
                    for drug1, drug2, cell, r, fold in DataLoader(train_set, batch_size=config['data']['args']['batch_size'], shuffle=True):
                        drug1 = drug1.to(device)
                        drug2 = drug2.to(device)
                        cell = cell.to(device)
                        r = r.to(device)
                        logits = model(drug1, drug2, cell, merge_feat_drug, hg_drug, merge_feat_cell, hg_cell)
                        optimizer.zero_grad()
                        loss = loss_fcn(logits, r.float())
                        loss.backward()
                        optimizer.step()
                        all_loss += loss.item()

                    loss_train = all_loss / (len(train_set) // config['data']['args']['batch_size'])
                    print('[test_fold {}, epoch {}],avg_loss={:.4f}'.format(test_fold, e, loss_train))
                    
                    # early stop
                    if (e == 0):
                        best_train_loss = loss_train
                        torch.save(model.state_dict(), 'model/{}_decoder{}.pkl'.format(now, test_fold)) 
                        print("save model")
                        earlystop_count = 0

                    else:
                        if best_train_loss > loss_train:
                            best_train_loss = loss_train
                            torch.save(model.state_dict(), 'model/{}_decoder{}.pkl'.format(now, test_fold))
                            print("save model")
                            earlystop_count = 0
                        
                        # 如果10次loss没有降低就提早结束
                        if earlystop_count != 10:
                            earlystop_count += 1
                        else:
                            print("early stop!!!!")
                            break
                        
                print("\nOptimization Finished!")
                        
                print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
            
            model = PHCSynergy(config, entity_pre_embed, structure_pre_embed, gene_expression, cell_map,
                                HGNNP(in_channels=merge_feat_drug.shape[1],
                                hid_channels=config['model']['args']['hyper_dim']),
                                HGNNP(in_channels=merge_feat_cell.shape[1],
                                hid_channels=config['model']['args']['hyper_dim'])).to(device)
            
            print('load model/{}_decoder{}.pkl'.format(now, test_fold))
            model.load_state_dict(torch.load('model/{}_decoder{}.pkl'.format(now, test_fold)))
            # model.load_state_dict(torch.load('model/2024-03-19-06_38_decoder0.pkl'))
            
            test_set = torch.LongTensor(test_set)
            with torch.no_grad():
                model.eval()
                drug1_ids = test_set[:, 0]
                drug2_ids = test_set[:, 1]
                cell_ids = test_set[:, 2]
                labels = test_set[:, 3]
                logits = model(drug1_ids, drug2_ids, cell_ids, merge_feat_drug, hg_drug, merge_feat_cell, hg_cell)

                p, r, acc, auc, aupr, f1 = eval_classification(labels, logits)
                print(
                    'test: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f} | auc {:.4f} | aupr {:.4f} | F1 {:.4f}'.format(
                        p, r, acc, auc, aupr, f1))
                auc_kfold.append(auc)
                aupr_kfold.append(aupr)
                acc_kfold.append(acc)
                auc_cv.append(auc)
                aupr_cv.append(aupr)
                acc_cv.append(acc)
                
            del model
            test_fold += 1
            
        auc_mean = np.mean(auc_cv)
        aupr_mean = np.mean(aupr_cv)
        acc_mean = np.mean(acc_cv)
        auc_all.append(auc_mean)
        aupr_all.append(aupr_mean)
        acc_all.append(acc_mean)
        print('result: auc {:.4f} | aupr {:.4f} | accuracy {:.4f}'.format(auc_mean, aupr_mean, acc_mean))
        # f2.write('%.6f' % auc_mean + '\t' + '%.6f' % aupr_mean + '\t' + '%.6f' % acc_mean + '\n')

    final_auc = np.mean(auc_all)
    final_aupr = np.mean(aupr_all)
    final_acc = np.mean(acc_all)
    print('Final result: auc {:.4f} | aupr {:.4f} | accuracy {:.4f}'.format(final_auc, final_aupr, final_acc))


if __name__ == '__main__':
    seed = 55
    random.seed(seed)
    np.random.seed(seed)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train()
