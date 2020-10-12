from typing import Union


class Vocab(object):
    def __init__(self,
                 vocab_path: str,
                 unk_token: str = '<unk>',
                 bos_token: str = '<s>',
                 eos_token: str = '</s>',
                 pad_token: str = '<pad>',
                 cls_token: str = '[CLS]',
                 sep_token: str = '[SEP]'):
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token

        self.additional_tokens = [bos_token,
                                  eos_token, pad_token, cls_token, sep_token]

        with open(vocab_path, 'r', encoding='utf-8') as fp:
            # The additional tokens would be inserted before the words.
            self.words = fp.read().split()[::2]
            self.vocab = {word: i for i, word in enumerate(self.words)}

    def __getitem__(self, idx_or_token: Union[int, str]) -> Union[str, int]:
        try:
            if isinstance(idx_or_token, str):
                return self.vocab[idx_or_token]
            else:
                return self.words[idx_or_token]
        except:
            return self.vocab[self.unk_token]

    def __contains__(self, token: str) -> bool:
        return token in self.words

    def __len__(self) -> int:
        # Note that vocabulary size must be a multiple of 8 although the actual
        # number of words is less than it.
        return (len(self.words) + 7) // 8 * 8

    @property
    def unk_idx(self) -> int:
        return self.vocab[self.unk_token]

    @property
    def bos_idx(self) -> int:
        return self.vocab[self.bos_token]

    @property
    def eos_idx(self) -> int:
        return self.vocab[self.eos_token]

    @property
    def pad_idx(self) -> int:
        # return self.vocab[self.pad_token]
        return -100

    @property
    def cls_idx(self) -> int:
        return self.vocab[self.cls_token]

    @property
    def sep_idx(self) -> int:
        return self.vocab[self.sep_token]
