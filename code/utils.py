import torch
import numpy as np
import os
from Config import *
import torch

# 将原始文本转换为模型输入的格式，并返回输入id、mask、segment和标签，将其组合成一个批次的数据并返回
class data_generator:
    # 初始化方法，data是一个元组，包含了文本和标签，config是配置对象，包含模型训练和数据处理的各种配置，shuffle决定了是否在每个epoch开始时打乱数据
    def __init__(self, data, config, shuffle=False):
        # data是一个元组，包含了文本和标签
        self.data = data
        # batch_size是批量大小
        self.batch_size = config.batch_size
        # max_length是文本最大长度
        self.max_length = config.MAX_LEN
        # shuffle是是否打乱数据
        self.shuffle = shuffle
        
        # 确定词汇表文件名，BERT模型的词汇表文件名是vocab.txt，XLNet模型的词汇表文件名是spiece.model
        vocab = 'vocab.txt' if os.path.exists(config.model_path + 'vocab.txt') else 'spiece.model'
        # 加载对应的分词器
        self.tokenizer = TOKENIZERS[config.model].from_pretrained(config.model_path + vocab)

        # 计算每个epoch的批次数
        self.steps = len(self.data[0]) // self.batch_size
        # 如果数据不能整除批量大小，则批次数加1
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        # 返回每个epoch的批次数
        return self.steps

    def __iter__(self):
        # 将数据分为文本和标签，self.data[0]是文本，self.data[1]是标签
        c, y = self.data
        # 创建一个索引列表，包含了数据的索引
        idxs = list(range(len(c)))
        # 如果shuffle为True，则打乱索引列表
        if self.shuffle:
            np.random.shuffle(idxs)
        # 初始化四个列表，分别存储输入id、mask、segment、标签
        input_ids, input_masks, segment_ids, labels = [], [], [], []

        # 遍历索引列表，i是当前处理的数据的索引，index是该索引在索引列表中的索引
        for index, i in enumerate(idxs):
            # 获取当前数据的文本
            text = c[i]
            # 如果文本长度大于510，则截取前255和后255，因为BERT的最大长度是512，要留两个位置给[CLS]和[SEP]
            if len(text) > 510:
                text = text[:255] + text[-255:]
                
            # 使用分词器对文本进行分词，得到输入id
            input_id = self.tokenizer.encode(text, max_length=self.max_length, truncation='longest_first')
            # 创建mask列表，长度和输入id相同
            input_mask = [1] * len(input_id)
            # 创建segment列表，长度和输入id相同
            segment_id = [0] * len(input_id)
            # 计算需要填充的长度，即最大长度减去当前输入id的长度
            padding_length = self.max_length - len(input_id)
            
            # 对输入id、mask、segment进行填充，确保它们的长度都是最大长度
            input_id += ([0] * padding_length)
            input_mask += ([0] * padding_length)
            segment_id += ([0] * padding_length)

            # 将输入id、mask、segment和标签添加到对应的列表中
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            labels.append(y[i])
            
            # 如果当前列表的长度等于批量大小或者已经遍历完索引列表，则返回一个批次的数据
            if len(input_ids) == self.batch_size or i == idxs[-1]:
                # 使用yield关键字返回一个批次的数据，返回的数据是一个元组，包含了输入id、mask、segment和标签
                yield input_ids, input_masks, segment_ids, labels
                # 重置四个列表，准备处理下一个批次的数据
                input_ids, input_masks, segment_ids, labels = [], [], [], []

# 投影梯度下降法，用于对抗训练，即在模型训练过程中，迭代地对embedding参数进行对抗训练
class PGD():
    # 接收一个模型对象，即要接受对抗训练的模型
    def __init__(self, model):
        # 将模型对象保存到self.model中
        self.model = model
        # 初始化emb_backup和grad_backup两个字典，用于保存embedding参数和梯度
        self.emb_backup = {}
        self.grad_backup = {}

    # 攻击方法，epsilon是扰动大小，alpha是扰动步长，emb_name是embedding参数名，is_first_attack表示是否是第一次攻击（决定是否备份embedding参数）
    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False):
        # 遍历模型的所有参数，只对需要梯度且名字中包含emb_name的参数进行对抗训练
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # 如果是第一次攻击，则备份embedding参数，以便后续恢复
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                # 计算梯度的L2范数
                norm = torch.norm(param.grad)
                # 如果梯度的L2范数不为0且不是nan，则计算对抗训练的扰动
                if norm != 0 and not torch.isnan(norm):
                    # 扰动向量r_at的计算公式为alpha * param.grad / norm，即沿着梯度方向，大小由alpha和梯度的L2范数共同决定
                    r_at = alpha * param.grad / norm
                    # 将embedding参数加上扰动向量r_at，得到对抗训练后的embedding参数
                    param.data.add_(r_at)
                    # 调用project方法，确保对抗训练后的embedding参数不会超过epsilon的范围
                    param.data = self.project(name, param.data, epsilon)

    # 恢复方法，用于在对抗训练完成后恢复embedding参数
    def restore(self, emb_name='word_embeddings'):
        # 遍历模型的所有参数，只对需要梯度且名字中包含emb_name的参数进行对抗训练
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # 对embedding参数进行恢复，即将其值恢复为备份的值
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        # 清空emb_backup字典，即删除所有备份的embedding参数
        self.emb_backup = {}

    # 投影方法，用于确保对抗训练后的embedding参数不会超过epsilon的范围
    def project(self, param_name, param_data, epsilon):
        # 计算对抗训练后的embedding参数和备份的embedding参数的差值，即扰动向量r
        r = param_data - self.emb_backup[param_name]
        # 如果r的L2范数大于epsilon，则将r缩放到L2范数为epsilon
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        # 将缩放后的扰动向量r加到备份的embedding参数上，得到对抗训练后的embedding参数
        return self.emb_backup[param_name] + r

    # 在对抗训练开始前备份模型的梯度
    def backup_grad(self):
        # 遍历模型的所有参数，只对需要梯度的参数进行备份
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 将参数的梯度保存到grad_backup字典中，这里使用了clone方法，即复制梯度，避免直接引用梯度，因为梯度会在反向传播过程中被修改
                self.grad_backup[name] = param.grad.clone()

    # 在对抗训练完成后恢复模型的梯度
    def restore_grad(self):
        # 遍历模型的所有参数，只对需要梯度的参数进行恢复
        for name, param in self.model.named_parameters():
            # 如果梯度备份字典中有该参数的梯度，则将备份的梯度恢复到该参数上
            if param.requires_grad:
                param.grad = self.grad_backup[name]
                
""" 
# 快速梯度方法，用于对抗训练，即在模型训练过程中，只进行单次对embedding参数的对抗训练
class FGM():
    # 接收一个模型对象，即要接受对抗训练的模型
    def __init__(self, model):
        # 将模型对象保存到self.model中
        self.model = model
        # 初始化backup字典，用于保存embedding参数
        self.backup = {}

    # 攻击方法，epsilon是扰动大小，emb_name是embedding参数名
    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # 遍历模型的所有参数，只对需要梯度且名字中包含emb_name的参数进行对抗训练
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # 备份embedding参数，以便后续恢复
                self.backup[name] = param.data.clone()
                # 计算梯度的L2范数
                norm = torch.norm(param.grad)
                # 如果梯度的L2范数不为0，则计算对抗训练的扰动
                if norm != 0:
                    # 扰动向量r_at的计算公式为epsilon * param.grad / norm，即沿着梯度方向，大小由epsilon和梯度的L2范数共同决定
                    r_at = epsilon * param.grad / norm
                    # 将embedding参数加上扰动向量r_at，得到对抗训练后的embedding参数
                    param.data.add_(r_at)

    # 恢复方法，用于在对抗训练完成后恢复embedding参数
    def restore(self, emb_name='word_embeddings'):
        # 遍历模型的所有参数，只对需要梯度且名字中包含emb_name的参数进行对抗训练
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # 对embedding参数进行恢复，即将其值恢复为备份的值
                assert name in self.backup
                param.data = self.backup[name]
        # 清空backup字典，即删除所有备份的embedding参数
        self.backup = {}


# Focal Loss，用于解决样本不平衡问题，即难样本的权重更大，支持平滑标签交叉熵
# 该损失函数的计算公式为：Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
# 该损失函数适用于多分类问题，num_class是类别数量，alpha是平衡因子，gamma是调节系数，smooth是平滑参数，size_average指示是否对每个样本的损失求平均
class FocalLoss(nn.Module):

    def __init__(self, num_class, alpha=None, gamma=2,
                smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        # num_class是类别数量
        self.num_class = num_class
        # alpha是平衡因子，用于调整不同类别的权重，alpha是一个列表，长度等于类别数量，每个元素是一个权重
        self.alpha = alpha
        # gamma是调节系数，用于调整难易样本的权重，gamma越大，难样本的权重越大
        self.gamma = gamma
        # smooth是平滑参数，用于平滑交叉熵损失，防止模型对其预测过于自信
        self.smooth = smooth
        # size_average指示是否对每个样本的损失求平均
        self.size_average = size_average

        # 如果alpha是None，则初始化为全1的张量，表示所有类别的权重都是1
        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        # 如果alpha是一个列表，则将其转换为张量，表示每个类别的权重
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        # 如果alpha不是一个列表或ndarray，则抛出异常，表示不支持的alpha类型
        else:
            raise TypeError('Not support alpha type')
        
        # 如果smooth不是None，则检查其值是否在[0,1]之间，否则抛出异常
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    # 前向传播方法，input是模型的输出，target是真实标签
    def forward(self, input, target):
        # 使用softmax函数将模型的输出转换为概率
        logit = F.softmax(input, dim=1)

        # 如果logit的维度大于2，则将其转换为二维张量，目的是方便后续计算损失
        if logit.dim() > 2:
            # 使用view方法将logit的维度转换为三维，第一个维度是batch_size，第二个维度是类别数量，将其他维度合并
            logit = logit.view(logit.size(0), logit.size(1), -1)
            # 使用permute方法将维度转换为(batch_size, -1, num_class)
            logit = logit.permute(0, 2, 1).contiguous()
            # 使用view方法将维度转换为二维，第一个维度是(batch_size * -1)，第二个维度是num_class
            logit = logit.view(-1, logit.size(-1))
        # 将真实标签转换为二维张量，第一个维度是(batch_size * -1)，第二个维度是1
        target = target.view(-1, 1)

        # epsilon是一个极小值，用于防止log函数的参数为0
        epsilon = 1e-10
        
        # 确保alpha和logit在同一设备上，如果不在同一设备上，则将alpha转换为和logit相同的设备
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        # 将真实标签转换为长整型，并确保其在和logit相同的设备上
        idx = target.cpu().long()
        # 创建一个全0的张量，维度是(target.size(0), num_class)，用于存储one-hot编码后的标签
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        
        # 使用scatter_方法将one_hot_key中的某些位置设为1，这些位置由idx指定
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        
        # 确保one_hot_key在和logit相同的设备上，如果不在同一设备上，则将one_hot_key转换为和logit相同的设备
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        # 如果smooth不是None，则对one_hot_key进行平滑处理，防止模型对其预测过于自信
        if self.smooth:
            # 将one_hot_key中的0替换为self.smooth，将1替换为1-self.smooth
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
            
        # 将one_hot_key和logit相乘，得到pt，即预测概率
        pt = (one_hot_key * logit).sum(1) + epsilon
        
        # 计算pt的对数
        logpt = pt.log()

        # 获取gamma，即调节系数
        gamma = self.gamma

        # 根据目标索引idx，从alpha中获取对应的权重
        alpha = alpha[idx]
        
        # 根据Focal Loss的公式计算损失，通过（1 - pt）的gamma次方来调整损失，即增加难样本的权重
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        # 根据size_average的值，决定是对每个样本的损失求平均还是直接求和
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
            
        return loss 
    """