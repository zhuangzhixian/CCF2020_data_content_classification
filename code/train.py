import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
import torch
import random
import os
from torch import nn

from transformers import AdamW,  \
    get_linear_schedule_with_warmup
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from model import *
from utils import *
import logging
logging.basicConfig(level=logging.DEBUG, filename="train.log",filemode='a')

import numpy as np
from sklearn.metrics import f1_score

# MODEL_CLASSES 是一个字典，将模型类的名称映射到相应的python类，便于命令行参数传入模型类
MODEL_CLASSES = {
    'BertForClass': BertForClass,
    'BertForClass_MultiDropout': BertForClass_MultiDropout,
    'BertLastTwoCls': BertLastTwoCls,
    'BertLastTwoClsPooler': BertLastTwoClsPooler,
    'BertLastTwoEmbeddings': BertLastTwoEmbeddings,
    'BertLastTwoEmbeddingsPooler': BertLastTwoEmbeddingsPooler,
    'BertLastFourCls': BertLastFourCls,
    'BertLastFourClsPooler': BertLastFourClsPooler,
    'BertLastFourEmbeddings': BertLastFourEmbeddings,
    'BertLastFourEmbeddingsPooler': BertLastFourEmbeddingsPooler,
    'BertDynCls': BertDynCls,
    'BertDynEmbeddings': BertDynEmbeddings,
    'BertRNN': BertRNN,
    'BertCNN': BertCNN,
    'BertRCNN': BertRCNN,
    'XLNet': XLNet,
    'Electra': Electra

}

# 创建一个参数解析器，用于解析命令行参数
parser = argparse.ArgumentParser()

# add_argument()方法用于指定程序需要接受的命令参数
# modelId: 模型编号，用于区分不同的模型
parser.add_argument('--modelId', default='0', type=int)
# model: 模型类的名称，必须与MODEL_CLASSES中的键一致
parser.add_argument(
        "--model",
        default='BertLastFourCls',
        type=str,
    )
# Stratification: 是否使用分层采样，分层采样的意思是在划分训练集和验证集时，每个类别的样本数量比例与原始数据集中的比例相同
parser.add_argument(
        "--Stratification",
        default=False,
        type=bool,
    )
# model_path: 预训练模型的路径
parser.add_argument(
        "--model_path",
        default='../chinese_roberta_wwm_ext_pytorch/',
        type=str,
    )
# dropout: dropout层的丢弃概率
parser.add_argument(
        "--dropout",
        default=0,
        type=float,
    )
# MAX_LEN: 文本序列的最大长度
parser.add_argument(
        "--MAX_LEN",
        default=512,
        type=int,
    )
# epoch: 训练的轮次数，每个epoch表示使用所有训练数据训练一次
parser.add_argument(
        "--epoch",
        default=1,
        type=int,
    )
# learn_rate: 学习率
parser.add_argument(
        "--learn_rate",
        default=2e-5,
        type=float,
    )
# batch_size: 批次大小，表示每次训练模型时，输入模型的样本数
# 一个epoch包含的批次数 = 训练集样本数 / batch_size
parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
    )
# k_fold: k折交叉验证的折数
parser.add_argument(
        "--k_fold",
        default=5,
        type=int,
    )
# seed: 随机种子，用于生成随机数，保证实验的可重复性
parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
# pgd: 是否使用pgd对抗训练
parser.add_argument(
        "--pgd",
        default=False,
        type=bool,
    )
# train_path: 训练集的路径
parser.add_argument(
        "--train_path",
        default="data/labeled_data_test.csv",
        type=str,
    )
# weight_list: 类别权重列表，用于处理样本不均衡问题
parser.add_argument(
        "--weight_list",
        default=[1,1,1,1,1,1,1,1,1,1,1,1,1],
        type=list,
    )

# 使用parse_args()方法解析参数，并将解析后的参数保存到args中
# 可以通过args.参数名的方式获取参数的值
args = parser.parse_args()

class Config:
    def __init__(self):
        # 从命令行参数中获取参数，保存在类的属性中
        # 如果参数没有提供，则使用默认值
        
        # modelId: 模型编号，用于区分不同的模型
        self.modelId = args.modelId
        # model: 模型类的名称，必须与MODEL_CLASSES中的键一致
        self.model = args.model
        # Stratification: 是否使用分层采样，分层采样的意思是在划分训练集和验证集时，每个类别的样本数量比例与原始数据集中的比例相同
        self.Stratification = args.Stratification
        # model_path: 预训练模型的路径
        self.model_path = args.model_path
        # num_class: 类别数量
        self.num_class = 13
        # dropout: dropout层的丢弃概率
        self.dropout = args.dropout
        # MAX_LEN: 文本序列的最大长度
        self.MAX_LEN = args.MAX_LEN
        # epoch: 训练的轮次数，每个epoch表示使用所有训练数据训练一次
        self.epoch = args.epoch
        # learn_rate: 学习率
        self.learn_rate = args.learn_rate
        # normal_lr: 学习率
        self.normal_lr = 1e-4
        # batch_size: 批次大小，表示每次训练模型时，输入模型的样本数
        self.batch_size = args.batch_size
        # k_fold: k折交叉验证的折数
        self.k_fold = args.k_fold
        # seed: 随机种子，用于生成随机数，保证实验的可重复性
        self.seed = args.seed
        # 指定设备，如果有可用的GPU，则使用GPU，否则使用CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # pgd: 是否使用pgd对抗训练
        self.pgd = args.pgd

# config初始化为Config类的一个实例
config = Config()
# 设置随机种子，保证实验的可重复性，分别设置python、numpy和pytorch的随机种子
random.seed(config.seed) # python随机种子
np.random.seed(config.seed) # numpy随机种子
torch.manual_seed(config.seed) # pytorch随机种子，为CPU设置随机种子
torch.cuda.manual_seed_all(config.seed) # pytorch随机种子，为GPU设置随机种子

# 设置日志文件的路径
file_path = './log/'
# 创建一个日志记录器logger
logger = logging.getLogger('mylogger')
# 设置logger的全局日志级别为DEBUG
logger.setLevel(logging.DEBUG)

# 加载训练集
train = pd.read_csv('data/labeled_data_test.csv')
# 加载测试集
test = pd.read_csv('data/test_data_test.csv')
# 加载提交样例
sub = pd.read_csv('data/submit_example.csv')

# 提取训练集和测试集的文本内容，确保文本内容的数据类型为字符串
train_content = train['content'].values.astype(str)
test_content = test['content'].values.astype(str)

# 假设class_label是以逗号分隔的字符串
def parse_labels(label_str):
    """将逗号分隔的标签字符串转换为标签列表"""
    return label_str.split(',')

# 应用解析函数到每个class_label上
train_labels = train['class_label'].apply(parse_labels)
categories = ["个人基本信息", "个人身份信息", "网络身份标识信息", "个人健康生理信息", "个人教育信息", "个人工作信息", "个人财产信息", "身份鉴别信息", "个人通信信息", "个人上网信息", "个人设备信息", "个人位置信息", "个人标签信息"]

# 创建类别到索引的映射
category_to_index = {category: index for index, category in enumerate(categories)}

# 转换为多热编码
num_samples = len(train_labels)
num_categories = len(categories)
multi_hot_labels = np.zeros((num_samples, num_categories), dtype=int)

for i, labels in enumerate(train_labels):
    for label in labels:
        index = category_to_index[label]
        multi_hot_labels[i, index] = 1

# 由于测试集没有标签，这里创建一个形状为(测试集样本数, 类别数量)的全0数组，用于存放模型的预测结果
test_label = np.zeros((len(test), config.num_class))

# 初始化oof_train和oof_test，用于存放模型的交叉验证结果
oof_train = np.zeros((len(train), config.num_class), dtype=np.float32)
oof_test = np.zeros((len(test), config.num_class), dtype=np.float32)

# 使用KFold划分训练集，n_splits定义了折数，shuffle=True表示打乱数据集，random_state设置随机种子，保证结果的可重复性
kf = KFold(n_splits=config.k_fold, shuffle=True, random_state=config.seed)

# 遍历每一折，split()方法返回每一折的训练集和验证集的索引，train_index和valid_index分别表示训练集和验证集的索引
for fold, (train_index, valid_index) in enumerate(kf.split(np.arange(len(train_labels)))):
    
    print('\n\n------------fold:{}------------\n'.format(fold))
    # 根据索引划分训练数据和标签
    c = train_content[train_index]
    y = multi_hot_labels[train_index]  # 使用多热编码的标签
    
    # 根据索引划分验证数据和标签
    val_c = train_content[valid_index]
    val_y = multi_hot_labels[valid_index]  # 使用多热编码的标签
    
    # 调用data_generator()方法，准备当前折的训练集和验证集的数据生成器
    train_D = data_generator([c, y], config, shuffle=True)
    val_D = data_generator([val_c, val_y], config)
    
    # 创建模型，如果有多个GPU，则使用DataParallel将模型复制到多个GPU上
    model = MODEL_CLASSES[config.model](config).to(config.device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
        
    # 根据配置初始化对抗训练策略，可以选择是否使用PGD
    if config.pgd:
        pgd = PGD(model)
        K = 3

    # 初始化BCEWithLogitsLoss，可以通过pos_weight参数处理不平衡的标签
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(np.array(args.weight_list)).float() if args.weight_list else None)
    loss_fn = loss_fn.to(config.device)

    # 计算在所有训练轮次（epoch）中总共需要执行的训练步数（train_steps）
    # 计算方法：训练集样本数 / batch_size * epoch
    num_train_steps = int(len(train) / config.batch_size * config.epoch)
    
    # 获取模型的所有参数，用于优化器优化
    param_optimizer = list(model.named_parameters())

    # 指定不需要权重衰减的参数，即不对这些参数的权重进行L2正则化，包括偏置项bias和LayerNorm层的参数
    # 偏置项bias通常不进行权重衰减，层归一化（LayerNorm）的参数通常也不进行权重衰减
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # 如果Stratification为True，则使用分层学习率
    if config.Stratification:
        # 找到所有BERT模型的参数
        bert_params = [x for x in param_optimizer if 'bert' in x[0]]
        # 找到所有非BERT模型的参数
        normal_params = [p for n, p in param_optimizer if 'bert' not in n]
        # 对BERT模型的参数进行权重衰减，如果参数名中包含no_decay中的字符串，则不进行权重衰减，否则进行权重衰减，权重衰减系数为0.01
        # 对非BERT模型的参数不进行权重衰减，学习率设置为config.normal_lr
        optimizer_parameters = [
            {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': normal_params, 'lr': config.normal_lr},
        ]
    # 如果Stratification为False，则不使用分层采样
    else:
        # 对所有模型的参数进行权重衰减，如果参数名中包含no_decay中的字符串，则不进行权重衰减，否则进行权重衰减，权重衰减系数为0.01
        # 不采用分层学习率，所有参数的学习率都设置为config.learn_rate
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

    # 使用AdamW优化器，学习率为config.learn_rate，这种方式比传统的Adam优化器更好地处理了权重衰减
    optimizer = AdamW(optimizer_parameters, lr=config.learn_rate)
    
    # 配置学习率调度器，采用带预热的学习率调度器，预热步数为训练集样本数的一半，总共训练步数为num_train_steps
    # 预热阶段学习率逐渐增加，然后保持不变，最后学习率逐渐减小，有助于模型更快地收敛
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train) / config.batch_size / 2),
        num_training_steps=num_train_steps
    )

    # 设置F1值的初始值为0，用于记录最佳F1值
    best_f1 = 0
    
    # 根据模型编号和折数，设置模型保存路径，如果路径不存在，则创建路径
    PATH = './models/model{}/bert_{}.pth'.format(config.modelId, fold)
    save_model_path = './models/model{}'.format(config.modelId)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    # 对每个epoch进行训练，config.epoch为训练的轮次数
    for e in range(config.epoch):
        print('\n------------epoch:{}------------'.format(e))
        # 将模型设置为训练模式，启用dropout层和Batch Normalization层
        model.train()
        
        # 初始化准确率acc、训练集样本数train_len、损失总和loss_num
        acc = 0
        train_len = 0
        loss_num = 0
        
        # tqdm是一个快速，可扩展的Python进度条，可以在循环体中添加一个进度条，显示循环的进度
        # 传入前面定义的train_D，表示对训练集的数据生成器进行遍历，tq是进度条的一个实例
        tq = tqdm(train_D)

        # 在每个批次中，遍历训练集的数据生成器，input_ids、input_masks、segment_ids和labels分别表示输入的文本id、mask、segment和标签
        # 将加载的数据转换为tensor类型，并移动到指定的设备上
        for input_ids, input_masks, segment_ids, labels in tq:
            # 转换数据类型以适配BCEWithLogitsLoss
            input_ids = torch.tensor(input_ids).to(config.device)
            input_masks = torch.tensor(input_masks).to(config.device)
            segment_ids = torch.tensor(segment_ids).to(config.device)
            label_t = torch.tensor(labels, dtype=torch.float).to(config.device)

            # 模型的前向传播，得到预测结果y_pred
            y_pred = model(input_ids, input_masks, segment_ids)

            # 计算损失值
            loss = loss_fn(y_pred, label_t)
            # 计算损失值的均值
            loss = loss.mean()
            # 反向传播，计算梯度
            loss.backward()

            # 如果config.pgd为True，则使用PGD对抗训练
            if config.pgd:
                # 先备份模型的原始梯度
                pgd.backup_grad()
                # 进行K次迭代，每次迭代都在embedding上添加对抗扰动
                for t in range(K):
                    # 在embedding上添加对抗扰动, first attack时备份param.data
                    pgd.attack(is_first_attack=(t == 0))
                    # 如果不是最后一次迭代，则将梯度置零，否则恢复原始梯度
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    
                    # 模型的前向传播，得到预测结果y_pred
                    y_pred = model(input_ids, input_masks, segment_ids)
                    
                    # 计算损失值
                    loss_adv = loss_fn(y_pred, label_t)
                    # 取损失值的均值
                    loss_adv = loss_adv.mean()
                    # 反向传播，计算梯度
                    loss_adv.backward()
                # 在最后一次迭代后，恢复embedding参数到之前备份的状态
                pgd.restore()

            # 梯度下降，更新参数
            optimizer.step()
            # 使用学习率调度器，调整学习率
            scheduler.step()
            # 清除累积的梯度，为下一批数据训练做准备
            model.zero_grad()

            # 使用sigmoid激活并应用阈值判断每个类别的归属
            sigmoid = torch.nn.Sigmoid()
            # 将预测结果应用sigmoid激活函数，得到每个类别的概率
            probs = sigmoid(y_pred.detach().to("cpu"))
            # 应用0.7的阈值，大于0.7的类别为正类，小于0.7的类别为负类
            predictions = (probs > 0.7).int()

            # 比较预测结果和真实标签，计算预测正确的类别比例
            correct_predictions = (predictions == label_t.detach().to("cpu").int()).float()
            # 计算当前批次平均准确率，并累加到总准确率acc中
            acc += correct_predictions.mean().item()
            # 损失值累加
            loss_num += loss.item()
            # 训练集样本数累加
            train_len += len(labels)
            # 在进度条中显示当前折数、当前轮次、损失值和准确率
            tq.set_postfix(fold=fold, epoch=e, loss=loss_num / train_len, acc=acc / train_len)

        # 将模型设置为评估模式
        model.eval()
        # 禁用梯度计算，节约资源
        with torch.no_grad():
            # 初始化验证集的预测结果列表y_p
            predicted_labels = []
            # 初始化train_logit，用于存放验证集的预测结果
            train_logit = None
            # 遍历验证集的数据生成器，获得输入的文本id、mask、segment和标签
            for input_ids, input_masks, segment_ids, labels in tqdm(val_D):
                # 将输入的文本id、mask、segment和标签转换为tensor类型，并移动到指定的设备上
                input_ids = torch.tensor(input_ids).to(config.device)
                input_masks = torch.tensor(input_masks).to(config.device)
                segment_ids = torch.tensor(segment_ids).to(config.device)
                label_t = torch.tensor(labels).to(config.device)

                # 模型的前向传播，得到预测结果y_pred
                y_pred = model(input_ids, input_masks, segment_ids)
                # 将预测结果移动到cpu上，并转换为numpy数组，以便进行后续处理
                y_pred = y_pred.detach().to("cpu").numpy()
                
                # sigmoid激活函数处理预测结果
                sigmoid = torch.nn.Sigmoid()
                # 应用sigmoid将logits转换成概率
                probs = sigmoid(torch.tensor(y_pred))
                # 应用阈值0.7确定最终预测
                predictions = (probs >= 0.7).int().numpy()
                
                # 如果train_logit为None，则将y_pred赋值给train_logit，否则将y_pred和train_logit在竖直方向上拼接
                if train_logit is None:
                    train_logit = y_pred
                else:
                    # vstack()函数用于在竖直方向上（行方向）合并数组，这里用于合并预测结果，以便计算F1值
                    train_logit = np.vstack((train_logit, y_pred))

                predicted_labels.extend(predictions)

            # 由于F1分数的计算需要在所有数据上进行，我们在循环结束后计算
            # 注意：这里使用了多标签的真实标签val_y (即multi_hot_labels[valid_index])和预测标签predicted_labels
            f1 = f1_score(val_y, predicted_labels, average="macro")
            print("best_f1:{}  f1:{}\n".format(best_f1, f1))

            # 如果当前F1值大于最佳F1值，则更新最佳F1值，并将当前模型保存到指定路径
            if f1 >= best_f1:
                best_f1 = f1
                oof_train[valid_index] = np.array(train_logit)
                torch.save(model.module if hasattr(model, "module") else model, PATH)

    # 使用data_generator()方法，准备测试集的数据生成器
    # test_content是测试集的文本内容，test_label是测试集的标签
    test_D = data_generator([test_content, test_label], config)
    # 加载保存的模型，该模型是在验证集上F1值最高的模型
    model = torch.load(PATH).to(config.device)
    # 设置为评估模式，禁用梯度计算
    model.eval()
    
    # 如果有多个GPU，则使用DataParallel将模型复制到多个GPU上
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
        
    # 禁用梯度计算，节约资源
    with torch.no_grad():
        # 初始化列表，用于存放测试集的预测概率
        pred_probs = []

        # 遍历测试集的数据生成器，获得输入的文本id、mask、segment和标签
        for input_ids, input_masks, segment_ids, labels in tqdm(test_D):
            input_ids = torch.tensor(input_ids).to(config.device)
            input_masks = torch.tensor(input_masks).to(config.device)
            segment_ids = torch.tensor(segment_ids).to(config.device)

            # 模型的前向传播，得到预测结果y_pred
            y_pred = model(input_ids, input_masks, segment_ids)
            # 应用sigmoid激活函数将logits转换为预测概率
            probs = torch.sigmoid(y_pred).to('cpu').numpy()
            # 将当前批次的预测概率添加到列表中
            pred_probs.extend(probs)

    # 将列表转换为NumPy数组，以便进行后续处理
    pred_probs = np.array(pred_probs)
    
    # 释放显存，删除模型，清空显存
    optimizer.zero_grad()
    del model
    torch.cuda.empty_cache()

    # 保存测试集的预测概率到oof_test中，适用于多标签分类
    oof_test = pred_probs

# oof_test原本是测试集的预测结果的累加，这里将oof_test除以折数，得到平均预测结果
oof_test /= config.k_fold

# 检查结果保存路径是否存在，如果不存在，则创建路径
save_result_path = './result'
if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)

# 假设设置阈值为0.7
threshold = 0.7
# 初始化一个空列表用于存储每个样本的标签列表
predicted_labels = []

for i in range(oof_test.shape[0]):
    # 获取当前样本的预测结果
    sample_pred = oof_test[i]
    # 根据阈值判断每个类别是否为正类
    labels = [categories[index] for index, value in enumerate(sample_pred) if value >= threshold]
    predicted_labels.append(labels)

# 将预测的标签列表转换为字符串格式，以逗号分隔，用于提交
sub['class_label'] = [','.join(label_list) for label_list in predicted_labels]

# 将预测结果保存到csv文件中，这个结果只包含了预测的类别名称
sub.to_csv("./result/result{}_original.csv".format(config.modelId), index=False)

