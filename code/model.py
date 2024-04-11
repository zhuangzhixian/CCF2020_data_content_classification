from torch import nn
from transformers import  \
    XLNetModel, ElectraModel
import torch
import os
from torch import nn
import torch.nn.functional as F
from transformers.activations import get_activation

from Config import *

# 直接利用BERT的输出进行分类，将CLS的输出和pooler的输出拼接在一起，然后进行分类
class BertForClass(nn.Module):
    def __init__(self, config):
        super(BertForClass, self).__init__()
        # 设置模型需要预测的类别数，从传入的config中获取
        self.n_classes = config.num_class
        # 读取模型的配置文件，如果模型路径下有bert_config.json文件，则读取该文件，否则读取config.json文件
        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        # 加载BERT模型的配置，CONFIGS是一个映射，根据传入的模型名字，获取对应的配置类
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
                                                                 output_hidden_states=True) # output_hidden_states=True表示输出所有隐藏层的状态
        # 加载BERT模型，MODELS是一个映射，根据传入的模型名字，获取对应的模型类
        self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
        # 根据config中的dropout参数，判断是否使用dropout
        self.isDropout = True if 0 < config.dropout < 1 else False
        # 创建了一个dropout层，如果使用dropout，则在模型的输出后加入dropout层
        self.dropout = nn.Dropout(p=config.dropout)
        # 创建了一个线性层，将BERT的输出拼接在一起，然后进行分类
        # 因为要拼接，所以线性层的输入维度是BERT的输出维度的两倍，输出维度是类别数
        self.classifier = nn.Linear(self.bert_config.hidden_size * 2, self.n_classes)

    # 前向传播函数，input_ids是输入序列的编码，input_masks是序列的注意力掩码，segment_ids是对于双序列任务的序列分段标识
    def forward(self, input_ids, input_masks, segment_ids):
        # 调用BERT模型，进行前向传播
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        # 获取序列输出
        sequence_output = outputs.last_hidden_state
        # 获取池化后的输出
        pooler_output = outputs.pooler_output
        # 计算序列输出的平均值，dim=1表示在第二个维度上求平均值，即对序列长度求平均
        seq_avg = torch.mean(sequence_output, dim=1)
        # 将序列输出的平均值和pooler的输出拼接在一起，作为分类的输入
        concat_out = torch.cat((seq_avg, pooler_output), dim=1)
        # 如果使用dropout，则对拼接的输出进行dropout
        if self.isDropout:
            concat_out = self.dropout(concat_out)
        # 将处理后的数据通过分类器，得到最终分类结果，即每个类别的得分
        logit = self.classifier(concat_out)
        
        return logit

# 直接利用BERT的输出进行分类，将CLS的输出和pooler的输出拼接在一起，然后进行分类
# 引入了多个dropout层，然后将多个dropout层的输出通过相同的分类器进行分类，最后将多个分类结果求平均
class BertForClass_MultiDropout(nn.Module):
    def __init__(self, config):
        super(BertForClass_MultiDropout, self).__init__()
        # 设置模型需要预测的类别数，从传入的config中获取
        self.n_classes = config.num_class
        # 读取模型的配置文件，如果模型路径下有bert_config.json文件，则读取该文件，否则读取config.json文件
        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        # 加载BERT模型的配置，CONFIGS是一个映射，根据传入的模型名字，获取对应的配置类
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
                                                                 output_hidden_states=True) # output_hidden_states=True表示输出所有隐藏层的状态
        # 加载BERT模型，MODELS是一个映射，根据传入的模型名字，获取对应的模型类
        self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
        # 定义多个dropout层，数量为5
        self.multi_drop = 5
        # 包含了5个dropout层的模块列表，每个层的dropout概率都是config中的dropout参数
        self.multi_dropouts = nn.ModuleList([nn.Dropout(config.dropout) for _ in range(self.multi_drop)])
        # 创建了一个线性层，将BERT的输出拼接在一起，然后进行分类
        # 因为要拼接，所以线性层的输入维度是BERT的输出维度的两倍，输出维度是类别数
        self.classifier = nn.Linear(self.bert_config.hidden_size * 2, self.n_classes)

    # 前向传播函数，input_ids是输入序列的编码，input_masks是序列的注意力掩码，segment_ids是对于双序列任务的序列分段标识
    def forward(self, input_ids, input_masks, segment_ids):
        # 调用BERT模型，进行前向传播
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        # 获取序列输出
        sequence_output = outputs.last_hidden_state
        # 获取池化后的输出
        pooler_output = outputs.pooler_output
        # 计算序列输出的平均值，dim=1表示在第二个维度上求平均值，即对序列长度求平均
        seq_avg = torch.mean(sequence_output, dim=1)
        # 将序列输出的平均值和pooler的输出拼接在一起，作为分类的输入
        concat_out = torch.cat((seq_avg, pooler_output), dim=1)
        # 遍历多个dropout层，将拼接的输出通过每个dropout层
        for j, dropout in enumerate(self.multi_dropouts):
            # 对于第一个dropout层，直接通过分类器得到分类结果，要除以multi_drop，因为最后要求平均
            if j == 0:
                logit = self.classifier(dropout(concat_out)) / self.multi_drop
            # 对于后面的dropout层，将分类结果除以multi_drop，然后累加，相当于求平均
            else:
                logit += self.classifier(dropout(concat_out)) / self.multi_drop

        return logit

# 直接利用BERT的输出进行分类，将最后一个隐藏层的CLS输出和倒数第二个隐藏层的CLS输出拼接在一起，然后进行分类
class BertLastTwoCls(nn.Module):
    def __init__(self, config):
        super(BertLastTwoCls, self).__init__()
        # 设置模型需要预测的类别数，从传入的config中获取
        self.n_classes = config.num_class
        # 读取模型的配置文件，如果模型路径下有bert_config.json文件，则读取该文件，否则读取config.json文件
        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
                                                                 output_hidden_states=True) # output_hidden_states=True表示输出所有隐藏层的状态
        # 加载BERT模型，MODELS是一个映射，根据传入的模型名字，获取对应的模型类
        self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
        # 根据config中的dropout参数，判断是否使用dropout
        self.isDropout = True if 0 < config.dropout < 1 else False
        # 创建了一个dropout层，如果使用dropout，则在模型的输出后加入dropout层
        self.dropout = nn.Dropout(p=config.dropout)
        # 创建了一个线性层，将BERT的输出拼接在一起，然后进行分类
        # 因为要拼接，所以线性层的输入维度是BERT的输出维度的两倍，输出维度是类别数
        self.classifier = nn.Linear(self.bert_config.hidden_size * 2, config.num_class)

    # 前向传播函数，input_ids是输入序列的编码，input_masks是序列的注意力掩码，segment_ids是对于双序列任务的序列分段标识
    def forward(self, input_ids, input_masks, segment_ids):
        # 调用BERT模型，进行前向传播
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        # 获取隐藏状态
        hidden_states = outputs.hidden_states
        # 将最后两个隐藏层的输出拼接在一起
        output = torch.cat(
            (hidden_states[-1][:, 0], hidden_states[-2][:, 0]), dim=1)
        # 如果使用dropout，则对拼接的输出进行dropout
        if self.isDropout:
            output = self.dropout(output)
        # 将处理后的数据通过分类器，得到最终分类结果，即每个类别的得分
        logit = self.classifier(output)

        return logit

# 直接利用BERT的输出进行分类，将最后一个隐藏层的CLS输出、倒数第二个隐藏层的CLS输出和pooler的输出拼接在一起，然后进行分类
class BertLastTwoClsPooler(nn.Module):
    def __init__(self, config):
        super(BertLastTwoClsPooler, self).__init__()
        # 设置模型需要预测的类别数，从传入的config中获取
        self.n_classes = config.num_class
        # 读取模型的配置文件，如果模型路径下有bert_config.json文件，则读取该文件，否则读取config.json文件
        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
                                                                 output_hidden_states=True) # output_hidden_states=True表示输出所有隐藏层的状态
        # 加载BERT模型，MODELS是一个映射，根据传入的模型名字，获取对应的模型类
        self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
        # 根据config中的dropout参数，判断是否使用dropout
        self.isDropout = True if 0 < config.dropout < 1 else False
        # 创建了一个dropout层，如果使用dropout，则在模型的输出后加入dropout层
        self.dropout = nn.Dropout(p=config.dropout)
        # 创建了一个线性层，将BERT的输出拼接在一起，然后进行分类
        # 因为要拼接，所以线性层的输入维度是BERT的输出维度的三倍，输出维度是类别数
        self.classifier = nn.Linear(self.bert_config.hidden_size * 3, config.num_class)

    # 前向传播函数，input_ids是输入序列的编码，input_masks是序列的注意力掩码，segment_ids是对于双序列任务的序列分段标识
    def forward(self, input_ids, input_masks, segment_ids):
        # 调用BERT模型，进行前向传播
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        # 获取池化后的输出
        pooler_output = outputs.pooler_output
        # 获取隐藏状态
        hidden_states = outputs.hidden_states
        # 将最后两个隐藏层的输出和pooler的输出拼接在一起
        output = torch.cat(
            (pooler_output, hidden_states[-1][:, 0], hidden_states[-2][:, 0]), dim=1)
        # 如果使用dropout，则对拼接的输出进行dropout
        if self.isDropout:
            output = self.dropout(output)
        # 将处理后的数据通过分类器，得到最终分类结果，即每个类别的得分
        logit = self.classifier(output)

        return logit

# 直接利用BERT的输出进行分类，将最后两个隐藏层的平均值拼接在一起，然后进行分类
class BertLastTwoEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertLastTwoEmbeddings, self).__init__()
        # 设置模型需要预测的类别数，从传入的config中获取
        self.n_classes = config.num_class
        # 读取模型的配置文件，如果模型路径下有bert_config.json文件，则读取该文件，否则读取config.json文件
        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
                                                                 output_hidden_states=True) # output_hidden_states=True表示输出所有隐藏层的状态
        # 加载BERT模型，MODELS是一个映射，根据传入的模型名字，获取对应的模型类
        self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
        # 根据config中的dropout参数，判断是否使用dropout
        self.isDropout = True if 0 < config.dropout < 1 else False
        # 创建了一个dropout层，如果使用dropout，则在模型的输出后加入dropout层
        self.dropout = nn.Dropout(p=config.dropout)
        # 创建了一个线性层，将BERT的输出拼接在一起，然后进行分类
        # 因为要拼接，所以线性层的输入维度是BERT的输出维度的两倍，输出维度是类别数
        self.classifier = nn.Linear(self.bert_config.hidden_size * 2, config.num_class)

    # 前向传播函数，input_ids是输入序列的编码，input_masks是序列的注意力掩码，segment_ids是对于双序列任务的序列分段标识
    def forward(self, input_ids, input_masks, segment_ids):
        # 调用BERT模型，进行前向传播
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        # 获取隐藏状态
        hidden_states = outputs.hidden_states
        
        # 计算最后两个隐藏层的平均值
        hidden_states1 = torch.mean(hidden_states[-1], dim=1)
        hidden_states2 = torch.mean(hidden_states[-2], dim=1)
        
        # 将最后两个隐藏层的平均值拼接在一起
        output = torch.cat(
            (hidden_states1, hidden_states2), dim=1)
        # 如果使用dropout，则对拼接的输出进行dropout
        if self.isDropout:
            output = self.dropout(output)
        # 将处理后的数据通过分类器，得到最终分类结果，即每个类别的得分
        logit = self.classifier(output)

        return logit

# 直接利用BERT的输出进行分类，将最后两个隐藏层的平均值和pooler的输出拼接在一起，然后进行分类
class BertLastTwoEmbeddingsPooler(nn.Module):
    def __init__(self, config):
        super(BertLastTwoEmbeddingsPooler, self).__init__()
        # 设置模型需要预测的类别数，从传入的config中获取
        self.n_classes = config.num_class
        # 读取模型的配置文件，如果模型路径下有bert_config.json文件，则读取该文件，否则读取config.json文件
        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
                                                                 output_hidden_states=True) # output_hidden_states=True表示输出所有隐藏层的状态
        # 加载BERT模型，MODELS是一个映射，根据传入的模型名字，获取对应的模型类
        self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
        # 根据config中的dropout参数，判断是否使用dropout
        self.isDropout = True if 0 < config.dropout < 1 else False
        # 创建了一个dropout层，如果使用dropout，则在模型的输出后加入dropout层
        self.dropout = nn.Dropout(p=config.dropout)
        # 创建了一个线性层，将BERT的输出拼接在一起，然后进行分类
        # 因为要拼接，所以线性层的输入维度是BERT的输出维度的三倍，输出维度是类别数
        self.classifier = nn.Linear(self.bert_config.hidden_size * 3, config.num_class)

    # 前向传播函数，input_ids是输入序列的编码，input_masks是序列的注意力掩码，segment_ids是对于双序列任务的序列分段标识
    def forward(self, input_ids, input_masks, segment_ids):
        # 调用BERT模型，进行前向传播
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        # 获取池化后的输出
        pooler_output = outputs.pooler_output
        
        # 获取隐藏状态
        hidden_states = outputs.hidden_states
        
        # 计算最后两个隐藏层的平均值
        hidden_states1 = torch.mean(hidden_states[-1], dim=1)
        hidden_states2 = torch.mean(hidden_states[-2], dim=1)

        # 将最后两个隐藏层的平均值和pooler的输出拼接在一起
        output = torch.cat(
            (pooler_output, hidden_states1, hidden_states2), dim=1)
        # 如果使用dropout，则对拼接的输出进行dropout
        if self.isDropout:
            output = self.dropout(output)
        # 将处理后的数据通过分类器，得到最终分类结果，即每个类别的得分 
        logit = self.classifier(output)

        return logit

# 直接利用BERT的输出进行分类，将最后四个隐藏层的CLS输出拼接在一起，然后进行分类
class BertLastFourCls(nn.Module):
    def __init__(self, config):
        super(BertLastFourCls, self).__init__()
        # 设置模型需要预测的类别数，从传入的config中获取
        self.n_classes = config.num_class
        # 读取模型的配置文件，如果模型路径下有bert_config.json文件，则读取该文件，否则读取config.json文件
        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
                                                                 output_hidden_states=True) # output_hidden_states=True表示输出所有隐藏层的状态
        # 加载BERT模型，MODELS是一个映射，根据传入的模型名字，获取对应的模型类
        self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
        # 根据config中的dropout参数，判断是否使用dropout
        self.isDropout = True if 0 < config.dropout < 1 else False
        # 创建了一个dropout层，如果使用dropout，则在模型的输出后加入dropout层
        self.dropout = nn.Dropout(p=config.dropout)
        # 创建了一个线性层，将BERT的输出拼接在一起，然后进行分类
        # 因为要拼接，所以线性层的输入维度是BERT的输出维度的四倍，输出维度是类别数
        self.classifier = nn.Linear(self.bert_config.hidden_size * 4, config.num_class)

    # 前向传播函数，input_ids是输入序列的编码，input_masks是序列的注意力掩码，segment_ids是对于双序列任务的序列分段标识
    def forward(self, input_ids, input_masks, segment_ids):
        # 调用BERT模型，进行前向传播
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        # 获取隐藏状态
        hidden_states = outputs.hidden_states
        # 将最后四个隐藏层的CLS输出拼接在一起
        output = torch.cat(
            (hidden_states[-1][:, 0], hidden_states[-2][:, 0], hidden_states[-3][:, 0], hidden_states[-4][:, 0]), dim=1)
        # 如果使用dropout，则对拼接的输出进行dropout
        if self.isDropout:
            output = self.dropout(output)
        # 将处理后的数据通过分类器，得到最终分类结果，即每个类别的得分
        logit = self.classifier(output)

        return logit

# 直接利用BERT的输出进行分类，将最后四个隐藏层的CLS输出和pooler的输出拼接在一起，然后进行分类
class BertLastFourClsPooler(nn.Module):
    def __init__(self, config):
        super(BertLastFourClsPooler, self).__init__()
        # 设置模型需要预测的类别数，从传入的config中获取
        self.n_classes = config.num_class
        # 读取模型的配置文件，如果模型路径下有bert_config.json文件，则读取该文件，否则读取config.json文件
        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
                                                                 output_hidden_states=True) # output_hidden_states=True表示输出所有隐藏层的状态
        # 加载BERT模型，MODELS是一个映射，根据传入的模型名字，获取对应的模型类
        self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
        # 根据config中的dropout参数，判断是否使用dropout
        self.isDropout = True if 0 < config.dropout < 1 else False
        # 创建了一个dropout层，如果使用dropout，则在模型的输出后加入dropout层
        self.dropout = nn.Dropout(p=config.dropout)
        # 创建了一个线性层，将BERT的输出拼接在一起，然后进行分类
        # 因为要拼接，所以线性层的输入维度是BERT的输出维度的五倍，输出维度是类别数
        self.classifier = nn.Linear(self.bert_config.hidden_size * 5, config.num_class)

    # 前向传播函数，input_ids是输入序列的编码，input_masks是序列的注意力掩码，segment_ids是对于双序列任务的序列分段标识
    def forward(self, input_ids, input_masks, segment_ids):
        # 调用BERT模型，进行前向传播
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        # 获取池化后的输出
        pooler_output = outputs.pooler_output
        # 获取隐藏状态
        hidden_states = outputs.hidden_states
        # 将最后四个隐藏层的CLS输出和pooler的输出拼接在一起
        output = torch.cat(
            (pooler_output, hidden_states[-1][:, 0], hidden_states[-2][:, 0], hidden_states[-3][:, 0], hidden_states[-4][:, 0]), dim=1)
        # 如果使用dropout，则对拼接的输出进行dropout
        if self.isDropout:
            output = self.dropout(output)
        # 将处理后的数据通过分类器，得到最终分类结果，即每个类别的得分
        logit = self.classifier(output)

        return logit

# 直接利用BERT的输出进行分类，将最后四个隐藏层的平均值拼接在一起，然后进行分类
class BertLastFourEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertLastFourEmbeddings, self).__init__()
        # 设置模型需要预测的类别数，从传入的config中获取
        self.n_classes = config.num_class
        # 读取模型的配置文件，如果模型路径下有bert_config.json文件，则读取该文件，否则读取config.json文件
        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
                                                                 output_hidden_states=True) # output_hidden_states=True表示输出所有隐藏层的状态
        # 加载BERT模型，MODELS是一个映射，根据传入的模型名字，获取对应的模型类
        self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
        # 根据config中的dropout参数，判断是否使用dropout
        self.isDropout = True if 0 < config.dropout < 1 else False
        # 创建了一个dropout层，如果使用dropout，则在模型的输出后加入dropout层
        self.dropout = nn.Dropout(p=config.dropout)
        # 创建了一个线性层，将BERT的输出拼接在一起，然后进行分类
        # 因为要拼接，所以线性层的输入维度是BERT的输出维度的四倍，输出维度是类别数
        self.classifier = nn.Linear(self.bert_config.hidden_size * 4, config.num_class)

    # 前向传播函数，input_ids是输入序列的编码，input_masks是序列的注意力掩码，segment_ids是对于双序列任务的序列分段标识
    def forward(self, input_ids, input_masks, segment_ids):
       # 调用BERT模型，进行前向传播
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        # 获取隐藏状态
        hidden_states = outputs.hidden_states
        
        # 计算最后四个隐藏层的平均值
        hidden_states1 = torch.mean(hidden_states[-1], dim=1)
        hidden_states2 = torch.mean(hidden_states[-2], dim=1)
        hidden_states3 = torch.mean(hidden_states[-3], dim=1)
        hidden_states4 = torch.mean(hidden_states[-4], dim=1)
        
        # 将最后四个隐藏层的平均值拼接在一起
        output = torch.cat(
            (hidden_states1, hidden_states2, hidden_states3, hidden_states4), dim=1)
        # 如果使用dropout，则对拼接的输出进行dropout
        if self.isDropout:
            output = self.dropout(output)
        # 将处理后的数据通过分类器，得到最终分类结果，即每个类别的得分
        logit = self.classifier(output)

        return logit

# 直接利用BERT的输出进行分类，将最后四个隐藏层的平均值和pooler的输出拼接在一起，然后进行分类
class BertLastFourEmbeddingsPooler(nn.Module):
    def __init__(self, config):
        super(BertLastFourEmbeddingsPooler, self).__init__()
        # 设置模型需要预测的类别数，从传入的config中获取
        self.n_classes = config.num_class
        # 读取模型的配置文件，如果模型路径下有bert_config.json文件，则读取该文件，否则读取config.json文件
        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
                                                                 output_hidden_states=True) # output_hidden_states=True表示输出所有隐藏层的状态
        # 加载BERT模型，MODELS是一个映射，根据传入的模型名字，获取对应的模型类
        self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
        # 根据config中的dropout参数，判断是否使用dropout
        self.isDropout = True if 0 < config.dropout < 1 else False
        # 创建了一个dropout层，如果使用dropout，则在模型的输出后加入dropout层
        self.dropout = nn.Dropout(p=config.dropout)
        # 创建了一个线性层，将BERT的输出拼接在一起，然后进行分类
        # 因为要拼接，所以线性层的输入维度是BERT的输出维度的五倍，输出维度是类别数
        self.classifier = nn.Linear(self.bert_config.hidden_size * 5, config.num_class)

    # 前向传播函数，input_ids是输入序列的编码，input_masks是序列的注意力掩码，segment_ids是对于双序列任务的序列分段标识
    def forward(self, input_ids, input_masks, segment_ids):
        # 调用BERT模型，进行前向传播
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        # 获取池化后的输出
        pooler_output = outputs.pooler_output
        # 获取隐藏状态
        hidden_states = outputs.hidden_states
        
        # 计算最后四个隐藏层的平均值
        hidden_states1 = torch.mean(hidden_states[-1], dim=1)
        hidden_states2 = torch.mean(hidden_states[-2], dim=1)
        hidden_states3 = torch.mean(hidden_states[-3], dim=1)
        hidden_states4 = torch.mean(hidden_states[-4], dim=1)
        
        # 将最后四个隐藏层的平均值和pooler的输出拼接在一起
        output = torch.cat(
            (pooler_output, hidden_states1, hidden_states2, hidden_states3, hidden_states4), dim=1)
        # 如果使用dropout，则对拼接的输出进行dropout
        if self.isDropout:
            output = self.dropout(output)
        # 将处理后的数据通过分类器，得到最终分类结果，即每个类别的得分
        logit = self.classifier(output)

        return logit

# 直接利用BERT的输出进行分类，计算每个隐藏层CLS的加权输出，然后将所有加权输出求和，得到最终的文本表示，并进行分类
class BertDynCls(nn.Module):
    def __init__(self, config):
        super(BertDynCls, self).__init__()
        # 设置模型需要预测的类别数，从传入的config中获取
        self.n_classes = config.num_class
        # 读取模型的配置文件，如果模型路径下有bert_config.json文件，则读取该文件，否则读取config.json文件
        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
                                                                 output_hidden_states=True) # output_hidden_states=True表示输出所有隐藏层的状态
        # 加载BERT模型，MODELS是一个映射，根据传入的模型名字，获取对应的模型类
        self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
        # 创建了一个线性层，用于计算每个隐藏层状态的动态权重
        self.dynWeight = nn.Linear(self.bert_config.hidden_size, 1)
        # 创建了一个线性层，用于对特征进行降维
        self.dense = nn.Linear(self.bert_config.hidden_size, 512)
        # 根据config中的dropout参数，判断是否使用dropout
        self.isDropout = True if 0 < config.dropout < 1 else False
        # 创建了一个dropout层，如果使用dropout，则在模型的输出后加入dropout层
        self.dropout = nn.Dropout(p=config.dropout)
        # 创建了一个线性层，将BERT的输出拼接在一起，然后进行分类
        self.classifier = nn.Linear(512, config.num_class)

    # 前向传播函数，input_ids是输入序列的编码，input_masks是序列的注意力掩码，segment_ids是对于双序列任务的序列分段标识
    def forward(self, input_ids, input_masks, segment_ids):
        # 调用BERT模型，进行前向传播
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        # 获取隐藏状态
        hidden_states = outputs.hidden_states

        # 构建一个列表，用于存储每个隐藏层的加权输出
        weighted_hidden_states = []

        # 遍历每个隐藏层输出
        for hidden in hidden_states[1:]:  # 从1开始是因为第一个隐藏层是输入层，不需要加权
            # 计算每个隐藏层输出的动态权重
            weight = torch.sigmoid(self.dynWeight(hidden[:, 0, :]))
            # 应用权重到整个隐藏层
            weighted_hidden = hidden * weight.unsqueeze(-1)
            weighted_hidden_states.append(weighted_hidden)
        
        # 将所有加权的隐藏层状态求和，形成最终的文本表示
        weighted_sum = torch.sum(torch.stack(weighted_hidden_states), dim=0)
        
        # 计算加权和的平均值作为文本的表示
        avg_weighted_sum = torch.mean(weighted_sum, dim=1)
        
        # 如果使用dropout，则对加权平均后的文本表示进行dropout
        if self.isDropout:
            avg_weighted_sum = self.dropout(avg_weighted_sum)
        
        # 通过降维层
        dense_output = self.dense(avg_weighted_sum)
        # 将处理后的数据通过分类器，得到最终分类结果，即每个类别的得分
        logit = self.classifier(dense_output)

        return logit

# 直接利用BERT的输出进行分类，计算每个隐藏层的平均值的加权输出，然后将所有加权输出求和，得到最终的文本表示，并进行分类
class BertDynEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertDynEmbeddings, self).__init__()
        # 设置模型需要预测的类别数，从传入的config中获取
        self.n_classes = config.num_class
        # 读取模型的配置文件，如果模型路径下有bert_config.json文件，则读取该文件，否则读取config.json文件
        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
                                                                 output_hidden_states=True) # output_hidden_states=True表示输出所有隐藏层的状态
        # 加载BERT模型，MODELS是一个映射，根据传入的模型名字，获取对应的模型类
        self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
        # 创建了一个线性层，用于计算每个隐藏层状态的动态权重
        self.dynWeight = nn.Linear(self.bert_config.hidden_size, 1)
        # 创建了一个线性层，用于对特征进行降维
        self.dense = nn.Linear(self.bert_config.hidden_size, 512)
        # 根据config中的dropout参数，判断是否使用dropout
        self.isDropout = True if 0 < config.dropout < 1 else False
        # 创建了一个dropout层，如果使用dropout，则在模型的输出后加入dropout层
        self.dropout = nn.Dropout(p=config.dropout)
        # 创建了一个线性层，将BERT的输出拼接在一起，然后进行分类
        self.classifier = nn.Linear(512, config.num_class)

    # 前向传播函数，input_ids是输入序列的编码，input_masks是序列的注意力掩码，segment_ids是对于双序列任务的序列分段标识
    def forward(self, input_ids, input_masks, segment_ids):
        # 调用BERT模型，进行前向传播
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        # 获取隐藏状态
        hidden_states = outputs.hidden_states

        # 构建一个列表，用于存储每个隐藏层的加权输出
        weighted_hidden_states = []
        
        # 遍历每个隐藏层输出
        for hidden in hidden_states[1:]:  # 从1开始是因为第一个隐藏层是输入层，不需要加权
            # 计算每个隐藏层输出的平均值
            hid_avg = torch.mean(hidden, dim=1)
            # 计算每个隐藏层输出的动态权重
            weight = torch.sigmoid(self.dynWeight(hid_avg))
            # 应用权重到整个隐藏层
            weighted_hidden = weight * hid_avg
            weighted_hidden_states.append(weighted_hidden)

        # 将所有加权的隐藏层状态求和，形成最终的文本表示
        combined_hidden = torch.sum(torch.stack(weighted_hidden_states, dim=0), dim=0)

        # 如果使用dropout，则对加权平均后的文本表示进行dropout
        if self.isDropout:
            combined_hidden = self.dropout(combined_hidden)

        # 通过降维层
        dense_output = self.dense(combined_hidden)
        # 将处理后的数据通过分类器，得到最终分类结果，即每个类别的得分
        logit = self.classifier(dense_output)

        return logit

# 使用BERT提取特征，然后输入到RNN中进行分类，RNN的输出作为最终的分类结果
# 综合考虑了BERT的特征提取和RNN的序列建模能力
# 结合了RNN的序列输出output和最后一个时间步的隐藏状态hidden，将其拼接在一起，然后通过线性层进行分类
# 最后一个时间步的信息提供了针对序列最后部分的上下文信息，而隐藏状态hidden提供了整个序列的信息
class BertRNN(nn.Module):
    def __init__(self, config):
        super(BertRNN, self).__init__()
        # 设置RNN的类型，可以是lstm、gru或rnn，gru是一种门控循环神经网络
        self.rnn_type = "gru"
        # 设置是否使用双向RNN
        self.bidirectional = True
        # 设置RNN的隐藏层维度
        self.hidden_dim = 256
        # 设置RNN的层数
        self.n_layers = 2
        # 设置是否批量优先，意思是输入数据的第一个维度是batch_size
        self.batch_first = True
        # 设置dropout的概率
        self.drop_out = 0.1
        # 设置模型需要预测的类别数，从传入的config中获取
        self.n_classes = config.num_class
        # 读取模型的配置文件，如果模型路径下有bert_config.json文件，则读取该文件，否则读取config.json文件
        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
                                                                 output_hidden_states=True) # output_hidden_states=True表示输出所有隐藏层的状态
        # 加载BERT模型，MODELS是一个映射，根据传入的模型名字，获取对应的模型类
        self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
        # 定义是单向还是双向RNN，单向为1，双向为2
        self.num_directions = 1 if not self.bidirectional else 2

        # 根据RNN的类型，初始化RNN层
        if self.rnn_type == 'lstm':
            # 接受BERT的隐藏层大小作为输入维度，隐藏层维度为hidden_dim，层数为n_layers，是否双向为bidirectional，是否批量优先为batch_first，dropout为drop_out
            self.rnn = nn.LSTM(self.bert_config.to_dict()['hidden_size'],
                               hidden_size=self.hidden_dim,
                               num_layers=self.n_layers,
                               bidirectional=self.bidirectional,
                               batch_first=self.batch_first,
                               dropout=self.drop_out)
        # 如果RNN的类型是gru，类似lstm
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(self.bert_config.to_dict()['hidden_size'],
                              hidden_size=self.hidden_dim,
                              num_layers=self.n_layers,
                              bidirectional=self.bidirectional,
                              batch_first=self.batch_first,
                              dropout=self.drop_out)
        # 否则，RNN的类型是rnn，也类似
        else:
            self.rnn = nn.RNN(self.bert_config.to_dict()['hidden_size'],
                              hidden_size=self.hidden_dim,
                              num_layers=self.n_layers,
                              bidirectional=self.bidirectional,
                              batch_first=self.batch_first,
                              dropout=self.drop_out)
            
        # 创建了一个dropout层
        self.dropout = nn.Dropout(self.drop_out)
        # 创建了一个线性层，将RNN的输出转换为分类结果
        # 输入维度是RNN的隐藏层维度乘以单向或双向，输出维度是类别数
        self.fc_rnn = nn.Linear(self.hidden_dim * self.num_directions, config.num_class)

    # 前向传播函数，input_ids是输入序列的编码，input_masks是序列的注意力掩码，segment_ids是对于双序列任务的序列分段标识
    def forward(self, input_ids, input_masks, segment_ids):
        # 调用BERT模型，进行前向传播
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        # 获取序列输出
        sequence_output = outputs.last_hidden_state
        
        # flatten_parameters()是为了让RNN的参数变为连续的，提高效率
        self.rnn.flatten_parameters()
        
        # 根据RNN的类型，将BERT的输出作为RNN的输入
        if self.rnn_type in ['rnn', 'gru']:
            # 得到每个时间步的输出和最后一个时间步的隐藏状态
            output, hidden = self.rnn(sequence_output)
        else:
            # 得到每个时间步的输出和最后一个时间步的隐藏状态和细胞状态
            output, (hidden, cell) = self.rnn(sequence_output)

        # 特殊处理双向RNN的隐藏状态
        if self.bidirectional:
            if self.rnn_type == 'lstm':
                hidden_forward = hidden[0][-2, :, :]  # LSTM前向的最后一层
                hidden_backward = hidden[0][-1, :, :]  # LSTM后向的最后一层
            else:
                hidden_forward = hidden[-2, :, :]  # 非LSTM前向的最后一层
                hidden_backward = hidden[-1, :, :]  # 非LSTM后向的最后一层
            
            # 将前向和后向的最后一层状态合并，计算平均值作为最终的隐藏状态
            hidden = (hidden_forward + hidden_backward) / 2
        else:
            # 如果是单向RNN，直接取最后一层的隐藏状态
            hidden = hidden[-1, :, :]

        # 调整序列输出的维度以匹配隐藏状态的维度
        # 仅保留序列的最后一个时间步的输出
        output = output[:, -1, :]

        # 合并处理后的隐藏状态和序列输出，并通过dropout
        fc_input = self.dropout(torch.cat((output, hidden), dim=1))

        # 将处理后的数据通过分类器，得到最终分类结果，即每个类别的得分
        out = self.fc_rnn(fc_input)

        return out

# 使用BERT提取特征，然后输入到CNN中进行分类，CNN的输出作为最终的分类结果
# 综合考虑了BERT的特征提取和CNN的局部特征提取能力
# 使用了多种大小的卷积核，分别提取不同范围的局部特征，然后将所有卷积核的输出拼接在一起，通过线性层进行分类
class BertCNN(nn.Module):
    def __init__(self, config):
        super(BertCNN, self).__init__()
        # 设置卷积核的数量
        self.num_filters = 100
        # 读取模型的配置文件，如果模型路径下有bert_config.json文件，则读取该文件，否则读取config.json文件
        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
                                                                 output_hidden_states=True) # output_hidden_states=True表示输出所有隐藏层的状态
        # hidden_size是BERT的输出维度
        self.hidden_size = self.bert_config.to_dict()['hidden_size']
        # 设置卷积核的大小，使用3、4、5三种大小的卷积核
        self.filter_sizes = {3, 4, 5}
        # dropout的概率
        self.drop_out = 0.5
        # 创建一个卷积层列表，用于存储三种不同大小的卷积核
        # 每个卷积核的输入通道数是1，因为文本特征是一维的，没有深度
        # 每个卷积核的输出通道数是num_filters，卷积核大小是（k，hidden_size）
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.hidden_size)) for k in self.filter_sizes])

        # 加载BERT模型，MODELS是一个映射，根据传入的模型名字，获取对应的模型类
        self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
        # 创建了一个dropout层
        self.dropout = nn.Dropout(self.drop_out)
        # 创建了一个线性层，将卷积层的输出拼接在一起，然后进行分类
        self.fc_cnn = nn.Linear(self.num_filters * len(self.filter_sizes), config.num_class)

    # 卷积和池化操作
    def conv_and_pool(self, x, conv):
        # 对输入进行卷积操作，然后使用ReLU激活函数
        x = F.relu(conv(x)).squeeze(3)
        # 对卷积后的结果进行最大池化操作
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    # 前向传播函数，input_ids是输入序列的编码，input_masks是序列的注意力掩码，segment_ids是对于双序列任务的序列分段标识
    def forward(self, input_ids, input_masks, segment_ids):
        # 调用BERT模型，进行前向传播
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        # 获取序列输出
        sequence_output = outputs.last_hidden_state
        # 先对BERT的输出进行dropout，以减少过拟合
        sequence_output = self.dropout(sequence_output)
        # 增加一个维度，因为卷积层的输入需要是四维的
        out = sequence_output.unsqueeze(1)
        # 对每个卷积核进行卷积和池化操作，将所有卷积核的结果拼接在一起
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # 再次进行dropout
        out = self.dropout(out)
        # 将处理后的数据通过分类器，得到最终分类结果，即每个类别的得分
        out = self.fc_cnn(out)
        return out

# 使用BERT提取特征，提取后的特征输入到RNN中，再通过CNN进行池化，最后通过线性层进行分类
class BertRCNN(nn.Module):
    def __init__(self, config):
        super(BertRCNN, self).__init__()
        # 设置RNN的类型，可以是lstm、gru或rnn，这里使用lstm
        self.rnn_type = "lstm"
        # 设置是否使用双向RNN
        self.bidirectional = True
        # 设置RNN的隐藏层维度
        self.hidden_dim = 256
        # 设置RNN的层数
        self.n_layers = 2
        # 设置是否批量优先，意思是输入数据的第一个维度是batch_size
        self.batch_first = True
        # 设置dropout的概率
        self.drop_out = 0.5
        # 设置模型需要预测的类别数，从传入的config中获取
        self.n_classes = config.num_class
        # 读取模型的配置文件，如果模型路径下有bert_config.json文件，则读取该文件，否则读取config.json文件
        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json, 
                                                                 output_hidden_states=True) # output_hidden_states=True表示输出所有隐藏层的状态
        # 加载BERT模型，MODELS是一个映射，根据传入的模型名字，获取对应的模型类
        self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
        
        # 根据RNN类型，初始化RNN层
        if self.rnn_type == 'lstm':
            # 接受BERT的隐藏层大小作为输入维度，隐藏层维度为hidden_dim，层数为n_layers，是否双向为bidirectional，是否批量优先为batch_first，dropout为drop_out
            self.rnn = nn.LSTM(self.bert_config.to_dict()['hidden_size'],
                               self.hidden_dim,
                               num_layers=self.n_layers,
                               bidirectional=self.bidirectional,
                               batch_first=self.batch_first,
                               dropout=self.drop_out)
        # 如果RNN的类型是gru，类似lstm
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(self.bert_config.to_dict()['hidden_size'],
                              hidden_size=self.hidden_dim,
                              num_layers=self.n_layers,
                              bidirectional=self.bidirectional,
                              batch_first=self.batch_first,
                              dropout=self.drop_out)
        # 否则，RNN的类型是rnn，也类似
        else:
            self.rnn = nn.RNN(self.bert_config.to_dict()['hidden_size'],
                              hidden_size=self.hidden_dim,
                              num_layers=self.n_layers,
                              bidirectional=self.bidirectional,
                              batch_first=self.batch_first,
                              dropout=self.drop_out)
            
        # 创建了一个dropout层
        self.dropout = nn.Dropout(self.drop_out)
        # 创建了一个线性层，将RNN的输出转换为分类结果，这里是双向RNN，所以输入维度是hidden_dim * 2，输出维度是类别数
        self.fc = nn.Linear(self.hidden_dim * 2, self.n_classes)
        

    def forward(self, input_ids, input_masks, segment_ids):
        # 调用BERT模型，进行前向传播
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        # 获取序列输出
        sequence_output = outputs.last_hidden_state
        # 获取池化后的输出
        pooler_output = outputs.pooler_output
        
        # sequence_output的维度是（batch_size, sequence_length, hidden_size）
        # 我们取出第二个维度的长度，即序列的长度
        sentence_len = sequence_output.shape[1]
        
        # pooler_output的维度是（batch_size, hidden_size）
        # 将pooler_output的维度调整为（batch_size, 1, hidden_size），然后复制到和sequence_output相同的维度，即（batch_size, sequence_length, hidden_size）
        pooler_output = pooler_output.unsqueeze(dim=1).repeat(1, sentence_len, 1)
        # 将BERT的输出和pooler的输出相加，得到最终的句子表示，维度是（batch_size, sequence_length, hidden_size）
        bert_sentence = sequence_output + pooler_output

        # flatten_parameters()是为了让RNN的参数变为连续的，提高效率
        self.rnn.flatten_parameters()
        
        # 根据RNN的类型，将BERT的输出作为RNN的输入
        if self.rnn_type in ['rnn', 'gru']:
            # 得到每个时间步的输出和最后一个时间步的隐藏状态
            output, hidden = self.rnn(bert_sentence)
        else:
            # 得到每个时间步的输出和最后一个时间步的隐藏状态和细胞状态
            output, (hidden, cell) = self.rnn(bert_sentence)

        # output的维度是（batch_size, sequence_length, hidden_dim * 2）
        # 将output通过relu激活函数
        output = torch.relu(output)
        
        # 将输出进行转置，即（batch_size, sequence_length, hidden_dim * 2） -> （batch_size, hidden_dim * 2, sequence_length）
        out = torch.transpose(output, 1, 2)

        # 对转置后的输出进行最大池化操作，得到最大值
        # 最大池化操作是在第三个维度上进行的，即对每个特征维度上的所有时间步进行最大池化
        # 最大池化后的维度是（batch_size, hidden_dim * 2, 1），squeeze(2)是为了去掉最后一个维度，变为（batch_size, hidden_dim * 2）
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        
        # 通过dropout层
        out = self.dropout(out)
        
        # 将处理后的数据通过分类器，得到最终分类结果，即每个类别的得分
        out = self.fc(out)

        return out

# 直接利用XLNet的输出进行分类，将序列输出沿序列长度的方向求和，得到整个序列的表示，然后通过线性层进行分类
class XLNet(nn.Module):
    def __init__(self, config):
        super(XLNet, self).__init__()
        # 加载XLNet模型
        self.xlnet = XLNetModel.from_pretrained(config.model_path)
        # 根据config中的dropout参数，判断是否使用dropout
        self.isDropout = True if 0 < config.dropout < 1 else False
        # 创建了一个dropout层，如果使用dropout，则在模型的输出后加入dropout层
        self.dropout = nn.Dropout(p=config.dropout)
        # 创建了一个线性层，将XLNet的输出转换为分类结果
        self.fc = nn.Linear(self.xlnet.d_model, config.num_class)

    # 前向传播函数，input_ids是输入序列的编码，input_masks是序列的注意力掩码，segment_ids是对于双序列任务的序列分段标识
    def forward(self, input_ids, input_masks, segment_ids):
        # 调用XLNet模型，获取序列的输出
        sequence_output = self.xlnet(input_ids=input_ids, token_type_ids=segment_ids,
                                     attention_mask=input_masks)
        # 序列的输出是一个元组，第一个元素是每个token的深层表示
        # 将序列的输出沿序列长度的方向求和，得到整个序列的表示
        sequence_output = torch.sum(sequence_output[0], dim=1)
        
        # 如果使用dropout，则对序列的表示进行dropout
        if self.isDropout:
            sequence_output = self.dropout(sequence_output)
            
        # 将处理后的数据通过分类器，得到最终分类结果，即每个类别的得分
        out = self.fc(sequence_output)
        return out

# 在Electra模型的基础上，添加一个分类头，用于进行文本分类
class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 创建了一个线性层，输入输出维度都是hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建了一个dropout层，丢弃概率为hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建了一个线性层，输入维度是hidden_size，输出维度是类别数
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    # 前向传播函数，features是从Electra模型中提取的特征
    def forward(self, features, **kwargs):
        # 从特征中提取第一个token的表示，类似于BERT的[CLS]表示
        x = features[:, 0, :]
        # 通过dropout层
        x = self.dropout(x)
        # 通过第一个线性层
        x = self.dense(x)
        # 使用gelu激活函数，这个函数是把tanh函数的输入缩放到0均值和单位方差的函数
        x = get_activation("gelu")(x)
        # 再次通过dropout层
        x = self.dropout(x)
        # 通过第二个线性层，得到最终的分类结果
        x = self.out_proj(x)
        return x

# 直接利用Electra的输出进行分类，将序列输出通过分类头，得到最终的分类结果，也就是通过若干个全连接层进行分类
class Electra(nn.Module):

    def __init__(self, config):
        super(Electra, self).__init__()
        # 加载Electra模型
        self.electra = ElectraModel.from_pretrained(config.model_path)
        # 读取模型的配置文件，如果模型路径下有bert_config.json文件，则读取该文件，否则读取config.json文件
        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        # 加载Electra模型的配置
        self.electra_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json)
        # 设置模型的类别数
        self.electra_config.num_labels = config.num_class
        # 调用ElectraClassificationHead，创建一个分类头
        self.fc = ElectraClassificationHead(self.electra_config)

    def forward(self, input_ids, input_masks, segment_ids):
        # 调用Electra模型，获取模型的输出，输出是一个元组，第一个元素是序列的表示，类似于BERT的[CLS]表示
        discriminator_hidden_states = self.electra(input_ids=input_ids, token_type_ids=segment_ids,
                                     attention_mask=input_masks)
        
        # 第一个元素是序列的表示，将其输入到分类头中，得到最终的分类结果
        sequence_output = discriminator_hidden_states[0]
        
        # 输入到分类头中，得到最终的分类结果
        out = self.fc(sequence_output)
        
        return out
