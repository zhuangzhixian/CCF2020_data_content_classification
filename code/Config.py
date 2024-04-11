from transformers import BertTokenizer, BertModel, BertConfig, \
    XLNetModel, XLNetTokenizer, XLNetConfig, ElectraModel, ElectraConfig, ElectraTokenizer

MODELS = {
    'BertForClass':  BertModel,
    'BertForClass_MultiDropout':  BertModel,
   'BertLastTwoCls':  BertModel,
   'BertLastTwoClsPooler':  BertModel,
    'BertLastTwoEmbeddings': BertModel,
    'BertLastTwoEmbeddingsPooler': BertModel,
    'BertLastFourCls': BertModel,
    'BertLastFourClsPooler':  BertModel,
    'BertLastFourEmbeddings':  BertModel,
   'BertLastFourEmbeddingsPooler':  BertModel,
   'BertDynCls':  BertModel,
    'BertDynEmbeddings': BertModel,
    'BertRNN': BertModel,
    'BertCNN': XLNetModel,
    'BertRCNN':  BertModel,
    'XLNet': XLNetModel,
    'Electra': ElectraModel
    }

TOKENIZERS = {
    'BertForClass': BertTokenizer,
    'BertForClass_MultiDropout': BertTokenizer,
    'BertLastTwoCls': BertTokenizer,
    'BertLastTwoClsPooler': BertTokenizer,
    'BertLastTwoEmbeddings': BertTokenizer,
    'BertLastTwoEmbeddingsPooler': BertTokenizer,
    'BertLastFourCls': BertTokenizer,
    'BertLastFourClsPooler': BertTokenizer,
    'BertLastFourEmbeddings': BertTokenizer,
    'BertLastFourEmbeddingsPooler': BertTokenizer,
    'BertDynCls': BertTokenizer,
    'BertDynEmbeddings': BertTokenizer,
    'BertRNN': BertTokenizer,
    'BertCNN': BertTokenizer,
    'BertRCNN': BertTokenizer,
    'XLNet': XLNetTokenizer,
    'Electra': ElectraTokenizer
    }

CONFIGS = {
    'BertForClass': BertConfig,
    'BertForClass_MultiDropout': BertConfig,
    'BertLastTwoCls': BertConfig,
    'BertLastTwoClsPooler': BertConfig,
    'BertLastTwoEmbeddings': BertConfig,
    'BertLastTwoEmbeddingsPooler': BertConfig,
    'BertLastFourCls': BertConfig,
    'BertLastFourClsPooler': BertConfig,
    'BertLastFourEmbeddings': BertConfig,
    'BertLastFourEmbeddingsPooler': BertConfig,
    'BertDynCls': BertConfig,
    'BertDynEmbeddings': BertConfig,
    'BertRNN': BertConfig,
    'BertCNN': BertConfig,
    'BertRCNN': BertConfig,
    'XLNet': XLNetConfig,
    'Electra': ElectraConfig
    }