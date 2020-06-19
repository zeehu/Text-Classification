#coding: utf-8
#Created Time: 2020-06-16 16:19:01
import transformers
MODEL_PATH = r"/Users/zeehu/Work/Text-Classification/bert-base-chinese/"
# a.通过词典导入分词器
tokenizer = \
transformers.BertTokenizer.from_pretrained(r"/Users/zeehu/Work/Text-Classification/bert-base-chinese/bert-base-chinese-vocab.txt") 
# b. 导入配置文件
model_config = transformers.BertConfig.from_pretrained(MODEL_PATH)
# 修改配置
model_config.output_hidden_states = True
model_config.output_attentions = True
# 通过配置和路径导入模型
model = transformers.BertModel.from_pretrained(MODEL_PATH,config = model_config)

# encode仅返回input_ids
print(tokenizer.encode("北京天安门"))
# encode_plus返回所有编码信息
print(tokenizer.encode_plus("北京大学", "北大"))


