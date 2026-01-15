import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter,save_model_dict
from model_Tran0814 import *
from sklearn.metrics import precision_score, recall_score
import datetime
import pandas as pd
import os
import random
import dgl
import argparse
from torch.utils.data import Dataset, DataLoader
import logging
import warnings
import copy

# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='warnings.log'  # 日志文件
)

# 捕获 warnings 并重定向到 logging
logging.captureWarnings(True)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
dgl.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # 关闭自动优化

start_time = datetime.datetime.now()
print (start_time)

def mkdir(path):
    folder=os.path.exists(path)
    
    if not folder:
        os.makedirs(path)
        print ("-----new folder-----")
    else:
        print ("-----There is this folder!------")


if torch.cuda.is_available():
    device = torch.device('cuda:1')
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(device))
else:
    print('No GPU available, using CPU instead.')
#device = 'cpu'
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and test Transformer')
    
    parser.add_argument('-c', '--cancertype', help='cancertype',
                        dest='cancertype',
                        default='COAD',
                        type=str
                        )
    parser.add_argument('-f', '--fold', help='number of fold (default: 1)',
                        dest='fold',
                        default=1,
                        type=int
                        )
    parser.add_argument('-e', '--epochs', help='maximum number of epochs (default: 1000)',
                        dest='epochs',
                        default=50,
                        type=int
                        )
    parser.add_argument('-pe', '--pretrain_epochs', help='number of pretrain epochs (default: 50)',
                   dest='pretrain_epochs',
                   default=50,
                   type=int
                   )
    parser.add_argument('-he', '--head', help='head (default: 1)',
                        dest='head',
                        default=1,
                        type=int
                        )
    parser.add_argument('-dp', '--dropout', help='the dropout rate (default: 0.1)',
                        dest='dp',
                        default=0.1,
                        type=float
                        )
    parser.add_argument('-bs', '--batch_size', help='batch_size (default: 64)',
                        dest='batch_size',
                        default=64,
                        type=int
                        )
    parser.add_argument('-lr_e', '--learningrate_e', help='the learning rate (default: 0.001)',
                        dest='lr_e',
                        default=0.0001,
                        type=float
                        )
    parser.add_argument('-lr_c', '--learningrate_c', help='the learning rate (default: 0.001)',
                        dest='lr_c',
                        default=0.0001,
                        type=float
                        )
    parser.add_argument('-df', '--dim_feedforward', help='dim_feedforward',
                        dest='dim_feedforward',
                        default=512,
                        type=int
                        )
    parser.add_argument('-dm', '--d_model', help='d_model',
                        dest='d_model',
                        default=512,
                        type=int
                        )
    parser.add_argument('-seed', '--seed', help='the random seed (default: 42)',
                        dest='seed',
                        default=42,
                        type=int
                        )
    args = parser.parse_args()
    return args
    
def main(args):
    
    # 确保完全确定性
    os.environ['PYTHONHASHSEED'] = str(42)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    
    cancer = args['cancertype']
    kf_num=args['fold']
    lr_e=args['lr_e']
    lr_c=args['lr_c']
    batch_size = args['batch_size']  # 批大小 
    num_epoch=args['epochs'] 
    d_model=args['d_model']
    nhead=args['head'] 
    dim_feedforward=args['dim_feedforward']
    dropout=args['dp']
    num_epoch_pretrain=args['pretrain_epochs']
    num_layers=2
    test_inverval=50
    view_list = [1,2,3]
    num_view = len(view_list)
    
    results_cwd='/home/mengping/experiment/01.csc1220/3.results/5.构建模型_dg+nocdg_svcl1/'+cancer+'/'+str(kf_num)+'.KF/bs'+str(batch_size)+'/dm'+str(d_model)+'/df'+str(dim_feedforward)+'/lr_e'+str(lr_e)+'/lr_c'+str(lr_c)+'/head'+str(nhead)
    mkdir(results_cwd)
    
    data_fea_folder = "/home/mengping/experiment/01.csc1220/3.results/3.提取特征_svc_l1/"+cancer+'/MultiOmics/WODG/'+str(kf_num)+'.KF'
    data_dg_folder = "/home/mengping/experiment/01.csc1220/3.results/3.提取特征_svc_l1/"+cancer+'/MultiOmics/'+str(kf_num)+'.KF/没进行svcl1的dg/'
    data_fea_tr_list, data_fea_all_list, data_dg_tr_list, data_dg_all_list, trte_idx, labels_trte = prepare_trte_data(data_fea_folder,data_dg_folder, view_list)
    # 
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    labels_te_tensor = torch.LongTensor(labels_trte[trte_idx["te"]])
    data_fea_te_list =[data_fea_all_list[0][trte_idx["te"]],data_fea_all_list[1][trte_idx["te"]],data_fea_all_list[2][trte_idx["te"]]]
    data_dg_te_list =[data_dg_all_list[0][trte_idx["te"]],data_dg_all_list[1][trte_idx["te"]],data_dg_all_list[2][trte_idx["te"]]]
    
    
    num_class = len(np.unique(labels_trte))
    dim_hvcdn = pow(num_class,num_view)
    dim_fea_list = [x.shape[1] for x in data_fea_tr_list]
    dim_dg_list = [x.shape[1] for x in data_dg_tr_list]
        
    # 创建数据集
    train_dataset = MultiOmicsListDataset(lasso_features_list=data_fea_tr_list,dg_features_list=data_dg_tr_list,
    labels=labels_trte[trte_idx['tr']])

# 测试集同理
    test_dataset = MultiOmicsListDataset(lasso_features_list=[data_fea_all_list[0][trte_idx["te"]],data_fea_all_list[1][trte_idx["te"]],data_fea_all_list[2][trte_idx["te"]]],
    dg_features_list=[data_dg_all_list[0][trte_idx["te"]],data_dg_all_list[1][trte_idx["te"]],data_dg_all_list[2][trte_idx["te"]]],
    labels=labels_trte[trte_idx['te']]
)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    #print (sample_weight_tr)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 模型初始化
    model_dict = init_model_dict1(num_view, num_class, dim_fea_list,dim_dg_list,d_model,dim_feedforward,dropout,nhead) 
    for m in model_dict:
        model_dict[m].to(device)
    
    # ========== 新增：定义最佳模型保存路径 ==========

    def train(model_dict, optim_dict, train_loader, use_vcdn=True):
        """
        纯训练模块（不包含测试）

        参数:
            model_dict: 模型字典
            optim_dict: 优化器字典
            train_loader: 训练数据加载器
            device: 计算设备
            use_vcdn: 是否使用VCDN融合

        返回:
            train_metrics: 包含训练指标的字典
        """
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        num_view = len(view_list)

        # 设置为训练模式
        for m in model_dict:
            model_dict[m].train()

        # 初始化统计量
        train_loss = 0
        correct = 0
        total = 0
        loss_dict = {f"C{i+1}": 0 for i in range(num_view)}
        if use_vcdn and num_view >= 2:
            loss_dict["C"] = 0

        # 训练过程
        for batch_idx, (data_lasso, data_dg, targets, batch_indices) in enumerate(train_loader):
            # 数据转移到设备
            data_lasso = [x.to(device) for x in data_lasso]
            data_dg = [x.to(device) for x in data_dg]
            targets = targets.to(device)
            batch_sample_weight = sample_weight_tr[batch_indices].to(device)

            ci_list = []

            # 各视图训练
            for i in range(num_view):
                optim_dict[f"C{i+1}"].zero_grad()

                # 前向传播
                ei = model_dict[f"E{i+1}"](data_lasso[i], data_dg[i])
                ci = model_dict[f"C{i+1}"](ei)
                ci_list.append(ci.detach())

                # 计算损失
                ci_loss = torch.mean(torch.mul(criterion(ci, targets),batch_sample_weight))
                #ci_loss = criterion(ci, targets)
                ci_loss.backward()
                optim_dict[f"C{i+1}"].step()
                
                loss_dict[f"C{i+1}"] += ci_loss.item()
                
                # -------- 预训练阶段统计单视图平均准确率 --------
                if not use_vcdn:
                    _, predicted = ci.max(1)
                    correct += predicted.eq(targets).sum().item()
                    total += targets.size(0)
                
            # VCDN训练
            if use_vcdn and num_view >= 2:
                optim_dict["C"].zero_grad()
                vcdn_out = model_dict["C"](ci_list)
                vcdn_loss = torch.mean(torch.mul(criterion(vcdn_out, targets), batch_sample_weight))
                vcdn_loss.backward()
                optim_dict["C"].step()

                loss_dict["C"] += vcdn_loss.item()
                _, predicted = vcdn_out.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        # 计算训练指标
        avg_loss = sum(loss_dict.values()) / len(train_loader)
        train_accuracy = 100. * correct / total

        # 返回训练指标
        return {
            'train_loss': avg_loss,
            'train_acc': train_accuracy,
            'loss_breakdown': loss_dict  # 各部分的损失明细
        }
    
    def test(model_dict, test_loader):
        """
        仅测试VCDN融合后的模型性能

        参数:
            model_dict: 模型字典
            test_loader: 测试数据加载器
            device: 计算设备

        返回:
            dict: 包含测试指标和预测结果
        """
        # 设置为评估模式
        for m in model_dict:
            model_dict[m].eval()

        num_view = len([k for k in model_dict.keys() if k.startswith("E")])

        # 初始化存储
        all_targets = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch_idx, (data_lasso, data_dg, targets,batch_indices) in enumerate(test_loader):
                # 数据转移到设备
                data_lasso = [x.to(device) for x in data_lasso]
                data_dg = [x.to(device) for x in data_dg]
                targets = targets.to(device)

                # 各视图前向传播
                ci_list = []
                for i in range(num_view):
                    ei = model_dict[f"E{i+1}"](data_lasso[i], data_dg[i])
                    ci = model_dict[f"C{i+1}"](ei)
                    ci_list.append(ci)

                # VCDN融合预测
                vcdn_out = model_dict["C"](ci_list)
                vcdn_probs = torch.softmax(vcdn_out, dim=1)
                _, vcdn_pred = torch.max(vcdn_out, 1)

                # 存储结果
                all_targets.append(targets.cpu())
                all_preds.append(vcdn_pred.cpu())
                all_probs.append(vcdn_probs.cpu())

        # 合并结果
        all_targets = torch.cat(all_targets).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_probs = torch.cat(all_probs).numpy()

        # 计算指标
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')
        return accuracy, f1, precision, recall, all_preds, all_probs


    print("\nPretraining encoders and classifiers...")
    # 预训练阶段只训练编码器和分类器，不训练VCDN
    optim_dict_pretrain = init_optim(num_view, model_dict, lr_e, lr_c)
    for epoch in range(num_epoch_pretrain):
        train_result = train(model_dict, optim_dict_pretrain, train_loader, use_vcdn=False)
    
    optim_dict = init_optim(num_view, model_dict,lr_e, lr_c)
    

    best_epoch = 0
    best_acc =0 
    best_pred_total = np.array([])  # 最佳预测标签
    best_probs_total = np.array([])  # 新增：最佳预测概率
    best_model_dict = None  # 最佳模型参数
    # ========== 新增：定义最佳模型保存路径 ==========
    best_model_path = os.path.join(results_cwd, "best_model.pth")
    for e in range(num_epoch):
        train_result = train(model_dict, optim_dict, train_loader, use_vcdn=True)
        accuracy, f1, precision, recall, all_preds, all_probs = test(model_dict,test_loader)
        if accuracy > best_acc:
            best_acc = accuracy
            best_epoch = e
            best_pred_total = all_preds
            best_probs_total = all_probs
            
            # ========== 新增：保存最佳模型权重 ==========
            torch.save({
                'model_state_dict': {k: v.state_dict() for k, v in model_dict.items()},
                'epoch': e,
                'best_acc': best_acc
            }, best_model_path)
            
            # 保存当前最佳模型的状态字典（深拷贝）
            best_model_dict = {key: copy.deepcopy(value.state_dict()) for key, value in model_dict.items()}
            for module, state_dict in best_model_dict.items():
                torch.save(state_dict, os.path.join(results_cwd, module + ".pth"))
            
                        
    print (train_result)

    # 首先，将tensor移动到CPU
    best_pred_total_cpu = best_pred_total
    # 然后，将tensor_cpu转换为NumPy数组
    best_pred_total_np = best_pred_total_cpu
    np.savetxt(results_cwd+'/best_pred_total_DG.csv',best_pred_total_np,fmt='%.4f',delimiter=',')
    np.savetxt(results_cwd+'/best_probs_total_DG.csv',best_probs_total,fmt='%.8f',delimiter=',')
    
    acc = accuracy_score(labels_trte[trte_idx['te']], best_pred_total_np)
    pre_weighted = precision_score(labels_trte[trte_idx['te']], best_pred_total_np,average='weighted')
    pre_macro = precision_score(labels_trte[trte_idx['te']], best_pred_total_np,average='macro')
    recall_weighted = recall_score(labels_trte[trte_idx['te']], best_pred_total_np,average='weighted')
    recall_macro = recall_score(labels_trte[trte_idx['te']], best_pred_total_np,average='macro')
    f1_weighted = f1_score(labels_trte[trte_idx['te']], best_pred_total_np,average='weighted')
    f1_macro = f1_score(labels_trte[trte_idx['te']], best_pred_total_np,average='macro')
    
    print ("ACC: ",acc)
    print ("pre_weighted: ",pre_weighted)
    print ("pre_macro: ",pre_macro)
    print ("recall_weighted: ",recall_weighted)
    print ("recall_macro: ",recall_macro)
    print ("f1_weighted: ",f1_weighted)
    print ("f1_macro: ",f1_macro)
    
    # 训练循环结束后
    print(f"Epoch {best_epoch}, Test Accuracy: {accuracy:.2f}%, Best Accuracy: {best_acc:.2f}%") 
    
    # ========== 新增：加载最佳模型 ==========
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device)
    for k, v in model_dict.items():
        v.load_state_dict(checkpoint['model_state_dict'][k])

    # 用最佳模型重新测试
    final_accuracy, final_f1, final_precision, final_recall, final_preds, final_probs = test(model_dict, test_loader)
    print(f"Best Model Final Performance - Accuracy: {final_accuracy:.4f}, F1: {final_f1:.4f}")
    
    
    # 训练结束后，加载最佳模型状态（确保使用的是最佳模型）
    if best_model_dict is not None:
        for key in model_dict.keys():
            model_dict[key].load_state_dict(best_model_dict[key])
        print(f"已加载最佳模型（epoch {best_epoch}, 准确率 {best_acc:.4f})")
    
    # 最终测试最佳模型
    final_accuracy, final_f1, final_precision, final_recall, final_preds, final_probs = test(model_dict, test_loader)
    print(f"最终模型性能 - 准确率: {final_accuracy:.4f}, F1: {final_f1:.4f}")
    
    end_time = datetime.datetime.now()
    print (end_time) 
    
    
if __name__ == '__main__':

    args = parse_args()
    args_dic = vars(args)
    print('args_dict', args_dic)

    main(args_dic)
    print('The Training and test is finished!')    
    
    
    
    
    
    