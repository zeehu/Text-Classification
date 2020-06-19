#coding: utf-8
#Created Time: 2020-06-16 21:21:15
#data_process

def process_data(source, target, label, tokenizer, max_len):
    len_s = len(source)
    len_t = len(target)
    
    return tokenizer.encode_plus(source, target)
    
class Dataset():
    def __init__(self, tokenizer, source, target, label):
        self.tokenizer = tokenizer
        self.source = source
        self.target = target
        self.label = label
        self.max_len = 128

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        data = process_data(
                self.source[item],
                self.target[item],
                self.label[item],
                self.tokenizer,
                self.max_len)
        return data
