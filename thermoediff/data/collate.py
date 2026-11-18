import numpy as np
import torch
from transformers import AutoTokenizer

class HuggingfaceCollaterFunc:
    @staticmethod
    def _as_vec(x):
        if isinstance(x, np.ndarray):
            return x.astype(np.float32, copy=False)
        if isinstance(x, (list, tuple)):
            return np.asarray(x, dtype=np.float32)
        if isinstance(x, str):
            s = x.strip().strip("[]")
            sep = "," if "," in s else " "
            return np.fromstring(s, sep=sep, dtype=np.float32)
        return np.asarray(x, dtype=np.float32)

    def __init__(self, bertDir, seqMaxLen=512):
        self.tokenizer = AutoTokenizer.from_pretrained(
            bertDir, trust_remote_code=True, do_lower_case=False
        )
        self.seqMaxLen = seqMaxLen

    def __call__(self, data):
        if 'seqA' in data[0]:
            tokenizedSeqAArr = self.tokenizer(
                [i['seqA'] for i in data],
                return_tensors='pt', max_length=self.seqMaxLen,
                padding='longest', truncation=True, return_token_type_ids=False
            )
            tokenizedSeqBArr = self.tokenizer(
                [i['seqB'] for i in data],
                return_tensors='pt', max_length=self.seqMaxLen,
                padding='longest', truncation=True, return_token_type_ids=False
            )
            mat = np.stack([self._as_vec(i["y"]) for i in data], axis=0)
            tokenizedLabArr = torch.from_numpy(mat).float()
            return {
                'tokenizedSeqAArr': tokenizedSeqAArr,
                'tokenizedSeqBArr': tokenizedSeqBArr,
                'tokenizedLabArr': tokenizedLabArr
            }
        else:
            tokenizedSeqArr = self.tokenizer(
                [i['seq'] for i in data],
                return_tensors='pt', max_length=self.seqMaxLen,
                padding='longest', truncation=True, return_token_type_ids=False
            )
            mat = np.stack([self._as_vec(i["y"]) for i in data], axis=0)
            tokenizedLabArr = torch.from_numpy(mat).float()
            return {'tokenizedSeqArr': tokenizedSeqArr, 'tokenizedLabArr': tokenizedLabArr}
