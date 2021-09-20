# Import Libaries
# INSTALL REQUIRED PACKAGES

import numpy as np
import pandas as pd
import json
from tensorflow import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder



class Title_Prediction():


    def __init__(self, path_weights):
        super(Title_Prediction, self).__init__()

        dict_path ='/content/drive/MyDrive/Github/NLP_Learning_by_Selective_Data/pretrained_model/chinese_L-12_H-768_A-12/vocab.txt'
        config_path = '/content/drive/MyDrive/Github/NLP_Learning_by_Selective_Data/pretrained_model/chinese_L-12_H-768_A-12/bert_config.json'
        checkpoint_path = '/content/drive/MyDrive/Github/NLP_Learning_by_Selective_Data/pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt'
        maxlen = 256
        topk = 1

        token_dict, keep_tokens = load_vocab(
        dict_path=dict_path,
        simplified=True,
        startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
        )
        self.tokenizer = Tokenizer(token_dict, do_lower_case=True)

        # the Star Chaser Model trained with movie dataset only
        self.model = build_transformer_model(
            config_path,
            checkpoint_path,
            application='unilm',
            keep_tokens=keep_tokens,  # include only tokens in keep tokens
        )
        self.model.load_weights(path_weights)
        #self.model = Model(model.inputs, output)
        #self.model.compile(optimizer=Adam(1e-5))
        self.autotitle = self.AutoTitle(start_id=None, end_id=self.tokenizer._token_end_id, maxlen=32)

    def createInner(self):
        return Title_Prediction.AutoTitle(self)

    def predict(self, contents, topk = 1):
        ls = []
        for content in contents:
            ls.append(''.join(self.autotitle.generate(content, topk)).lower())
        return ls

    class AutoTitle(AutoRegressiveDecoder):
        @AutoRegressiveDecoder.wraps(default_rtype='probas')

        def __init__(self, Title_Prediction):
            self.Title_Prediction = Title_Prediction

        def predict(self, inputs, output_ids, states):
            token_ids, segment_ids = inputs
            token_ids = np.concatenate([token_ids, output_ids], 1)
            segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
            #####################
            return self.Title_Prediction.model.predict([token_ids, segment_ids])[:, -1]
            #####################

        def generate(self, text, tokenizer, topk=1):
            max_c_len = 256 - self.maxlen
            token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
            output_ids = self.beam_search([token_ids, segment_ids],
                                          topk=topk)  # beam search
            return tokenizer.decode(output_ids)
            




