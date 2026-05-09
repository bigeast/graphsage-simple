import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
from collections import defaultdict
import random
import time
from sklearn.metrics import f1_score

from torch.utils.tensorboard import SummaryWriter

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super().__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(list(to_neigh), 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + [nodes[i]] for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        # mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        mask = torch.zeros(len(nodes), len(unique_nodes))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.tensor(unique_nodes_list, dtype=torch.long).cuda())
        else:
            embed_matrix = self.features(torch.tensor(unique_nodes_list, dtype=torch.long))
        to_feats = mask.mm(embed_matrix)
        return to_feats

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, aggregator,
            num_sample=10,
            base_model=None, gcn=False, cuda=False, 
            feature_transform=False): 
        super().__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(torch.zeros(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim, dtype=torch.float))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], 
                self.num_sample)
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.as_tensor(nodes).cuda())
                # self_feats = self.features(torch.tensor(nodes).cuda())
                
            else:
                self_feats = self.features(torch.as_tensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))
        return combined
        
class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super().__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.zeros(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

# 添加验证函数
def evaluate(model, nodes, labels, writer, epoch, phase="Validation"):
    """
    评估模型性能并记录到 TensorBoard
    """
    model.eval()
    with torch.no_grad():
        # 获取预测结果
        output = model.forward(nodes)
        pred = output.data.numpy().argmax(axis=1)
        true_labels = labels[nodes].squeeze()
        
        # 计算各种指标
        f1_micro = f1_score(true_labels, pred, average="micro")
        f1_macro = f1_score(true_labels, pred, average="macro")
        f1_weighted = f1_score(true_labels, pred, average="weighted")
        
        # 计算准确率
        accuracy = (pred == true_labels).mean()
        
        # 记录到 TensorBoard
        if writer is not None:
            writer.add_scalar(f'{phase}/F1_micro', f1_micro, epoch)
            writer.add_scalar(f'{phase}/F1_macro', f1_macro, epoch)
            writer.add_scalar(f'{phase}/F1_weighted', f1_weighted, epoch)
            writer.add_scalar(f'{phase}/Accuracy', accuracy, epoch)
        
        return {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'accuracy': accuracy
        }

# 主程序
if __name__ == "__main__":
    # 加载数据
    feat_data, labels, adj_lists = load_cora()
    
    dim_1 = 128
    samples_1 = 10
    dim_2 = 128
    samples_2 = 10
    num_nodes = 2708
    
    # 设置随机种子
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    
    # 创建模型
    features = nn.Embedding.from_pretrained(torch.tensor(feat_data, dtype=torch.float), freeze=True)
    agg1 = MeanAggregator(features)
    enc1 = Encoder(features, 1433, dim_1, adj_lists, agg1, num_sample=10, gcn=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t())
    enc2 = Encoder(lambda nodes : enc1(nodes).t(),
            dim_1, dim_2, adj_lists, agg2, 
            num_sample=samples_2,
            base_model=enc1,
            gcn=False)
    
    graphsage = SupervisedGraphSage(7, enc2)
    
    # 划分数据集
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])
    
    # 优化器
    optimizer = torch.optim.SGD([p for p in graphsage.parameters() if p.requires_grad], lr=0.7)
    
    # 训练配置
    batch_size = 256
    n_epochs = 100
    n_batchs = len(train) // batch_size + 1
    last_loss = 0.0
    times = []
    
    # 初始化 TensorBoard
    writer = SummaryWriter('runs/graphsage_cora')
    
    # 1. 记录模型图
    # 创建一个示例输入（需要是节点ID的tensor）
    # sample_nodes = torch.tensor(train[:10], dtype=torch.long)
    # writer.add_graph(graphsage, sample_nodes)
    # print("✓ 模型图已记录到 TensorBoard")
    
    # 训练循环
    best_val_f1 = 0.0
    early_stop_counter = 0
    early_stop_patience = 10
    
    for epoch in range(n_epochs):
        # 训练阶段
        graphsage.train()
        random.shuffle(train)
        epoch_losses = []
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch}/{n_epochs}")
        print("-" * 50)
        
        for batch_idx, batch_start in enumerate(range(0, len(train), batch_size)):
            batch_nodes = train[batch_start:min(batch_start + batch_size, len(train))]
            
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, torch.tensor(labels[batch_nodes]))
            loss.backward()
            optimizer.step()
            
            batch_time = time.time() - start_time
            times.append(batch_time)
            
            epoch_losses.append(loss.item())
            
            # 记录每个 batch 的 loss
            global_step = epoch * n_batchs + batch_idx
            writer.add_scalar('Training/Batch_Loss', loss.item(), global_step)
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{n_batchs}, Loss: {loss.item():.4f}, Time: {batch_time:.3f}s")
        
        # 计算 epoch 平均 loss
        avg_loss = np.mean(epoch_losses)
        writer.add_scalar('Training/Epoch_Loss', avg_loss, epoch)
        
        # 2. 在验证集上评估
        val_metrics = evaluate(graphsage, val, labels, writer, epoch, "Validation")
        
        # 3. 可选：在训练集上评估（用于监控过拟合）
        if epoch % 5 == 0:  # 每5个epoch评估一次训练集
            # 使用部分训练集评估以节省时间
            train_sample = random.sample(train, min(500, len(train)))
            train_metrics = evaluate(graphsage, train_sample, labels, writer, epoch, "Training")
        
        epoch_time = time.time() - epoch_start_time
        
        # 打印结果
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Val F1 (micro): {val_metrics['f1_micro']:.4f}")
        print(f"  Val F1 (macro): {val_metrics['f1_macro']:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if val_metrics['f1_micro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_micro']
            torch.save(graphsage.state_dict(), 'best_graphsage_model.pt')
            print(f"  ✓ 新的最佳模型！F1: {best_val_f1:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # 早停
        if early_stop_counter >= early_stop_patience:
            print(f"\n早停触发！验证集 F1 连续 {early_stop_patience} 个 epoch 没有提升")
            break
        
        # 学习率调整（可选）
        if epoch > 0 and avg_loss > last_loss and last_loss > 0:
            # 如果 loss 增加，降低学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.95
            print(f"  Learning rate adjusted to {optimizer.param_groups[0]['lr']:.5f}")
        
        last_loss = avg_loss
    
    # 训练完成后的最终评估
    print("\n" + "="*50)
    print("训练完成！最终评估：")
    
    # 在验证集上评估
    final_val_metrics = evaluate(graphsage, val, labels, writer, n_epochs, "Final_Validation")
    print(f"验证集 - F1 micro: {final_val_metrics['f1_micro']:.4f}")
    print(f"验证集 - F1 macro: {final_val_metrics['f1_macro']:.4f}")
    print(f"验证集 - Accuracy: {final_val_metrics['accuracy']:.4f}")
    
    # 在测试集上评估
    test_metrics = evaluate(graphsage, test, labels, writer, n_epochs, "Test")
    print(f"\n测试集 - F1 micro: {test_metrics['f1_micro']:.4f}")
    print(f"测试集 - F1 macro: {test_metrics['f1_macro']:.4f}")
    print(f"测试集 - Accuracy: {test_metrics['accuracy']:.4f}")
    
    # 记录超参数
    writer.add_hparams(
        {
            'lr': 0.3,
            'batch_size': 256,
            'dim_1': 256,
            'dim_2': 256,
            'samples_1': 5,
            'samples_2': 5,
            'n_epochs': n_epochs
        },
        {
            'hparam/best_val_f1': best_val_f1,
            'hparam/final_test_f1': test_metrics['f1_micro'],
            'hparam/final_test_accuracy': test_metrics['accuracy']
        }
    )
    
    print(f"\n平均batch时间: {np.mean(times):.4f}s")
    print(f"总训练时间: {sum(times):.2f}s")
    
    # 关闭 TensorBoard writer
    writer.close()
    print("\n✓ TensorBoard 日志已保存到 'runs/graphsage_cora'")
    print("运行以下命令查看结果：")
    print("tensorboard --logdir=runs")
