import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Config
from hfgpt2.utils import fusing
from hfgpt2.training import TrainingSpec, TrainConfig, Trainer
from hfgpt2.data import Dataset, Vocab, TokenizedCorpus

from typing import Tuple, Iterator, Dict


class GPT2TrainingSpec(TrainingSpec):
    def __init__(self, train_corpus: str, eval_corpus: str, vocab_path: str, seq_len: int,
                 vocab_size: int, n_positions: int, n_ctx: int,
                 n_embd: int, n_layer: int, n_head: int,
                 resid_pdrop: float, embd_pdrop: float, attn_pdrop: float,
                 layer_norm_epsilon: float, initializer_range: float,
                 base_lr: float, wd_rate: float,
                 total_steps: int, eval_steps: int, save_steps: int):
        self.train_corpus = train_corpus
        self.eval_corpus = eval_corpus
        self.vocab_path = vocab_path

        self.seq_len = seq_len

        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head

        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop

        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

        self.base_lr = base_lr
        self.wd_rate = wd_rate

        self.total_steps = total_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps

    def initialize(self):
        self.vocab = Vocab(vocab_path=self.vocab_path)
        self.config = GPT2Config(self.vocab_size, n_positions=self.n_positions,
                                 n_ctx=self.n_ctx, n_embd=self.n_embd, n_layer=self.n_layer,
                                 n_head=self.n_head, resid_pdrop=self.resid_pdrop, embd_pdrop=self.embd_pdrop,
                                 attn_pdrop=self.attn_pdrop, layer_norm_epsilon=self.layer_norm_epsilon,
                                 initializer_range=self.initializer_range)

    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        train_dataset = TokenizedCorpus(corpus_path=self.train_corpus,
                                        vocab=self.vocab,
                                        seq_len=self.n_ctx)
        eval_dataset = TokenizedCorpus(corpus_path=self.eval_corpus,
                                       vocab=self.vocab,
                                       seq_len=self.n_ctx)
        return train_dataset, eval_dataset

    def construct_model(self) -> nn.Module:
        return GPT2LMHeadModel(config=self.config)

    def create_optimizer(self, params: Iterator[nn.Parameter]
                         ) -> Tuple[optim.Optimizer,
                                    optim.lr_scheduler._LRScheduler]:
        optimizer = fusing.Adam(
            params, lr=self.base_lr, weight_decay=self.wd_rate)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1 - step / self.total_steps)
        return optimizer, scheduler

    def train_objective(self, data: Dict[str, torch.Tensor], model: nn.Module
                        ) -> Dict[str, torch.Tensor]:
        output = model(data['input'], labels=data['input'])
        loss = output[0]
        return {'loss': loss}

    def eval_objective(self, data: Dict[str, torch.Tensor], model: nn.Module
                       ) -> Dict[str, torch.Tensor]:
        output = model(data['input'], labels=data['input'])
        loss = output[0]
        return {'loss': loss}


def train_gpt2_model(args: argparse.Namespace):
    spec = GPT2TrainingSpec(
        train_corpus=args.train_corpus, eval_corpus=args.eval_corpus, vocab_path=args.vocab_path, seq_len=args.seq_len,
        vocab_size=args.vocab_size, n_positions=args.dims, n_ctx=args.dims, n_embd=args.dims,
        n_layer=args.layers, n_head=args.heads, resid_pdrop=args.dropout, embd_pdrop=args.dropout,
        attn_pdrop=args.dropout, layer_norm_epsilon=args.layer_norm_epsilon,
        initializer_range=args.initializer_range, base_lr=args.base_lr, wd_rate=args.wd_rate,
        total_steps=args.total_steps, eval_steps=args.eval_steps, save_steps=args.save_steps)
    config = TrainConfig(
        batch_train=args.batch_train, batch_eval=args.batch_eval,
        total_steps=args.total_steps, eval_steps=args.eval_steps,
        save_steps=args.save_steps, save_model_path=args.save_model_path,
        save_checkpoint_path=args.save_checkpoint_path,
        description='Train GPT-2 model',
        log_format='train/loss: {train_loss:.4f}, eval/loss: {eval_loss:.4f}',
        use_amp=args.use_amp, gpus=args.gpus)

    Trainer(spec, config).train(from_checkpoint=args.from_checkpoint,
                                from_pretrained=args.from_pretrained)


def add_subparsers(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser('train', help='train GPT2 Model')

    group = parser.add_argument_group('Corpus and Vocabulary')
    group.add_argument('--train_corpus', required=True,
                       help='training corpus file path')
    group.add_argument('--eval_corpus', required=True,
                       help='evaluation corpus file path')
    group.add_argument('--vocab_path', required=True,
                       help='vocabulary file path')

    group = parser.add_argument_group('Model configurations')
    group.add_argument('--seq_len', default=512, type=int,
                       help='maximum sequence length')
    group.add_argument('--vocab_size', default=50000,
                       type=int, help='size of vocabulary')
    group.add_argument('--layers', default=24, type=int,
                       help='number of transformer layers')
    group.add_argument('--heads', default=16, type=int,
                       help='number of multi-heads in attention layer')
    group.add_argument('--dims', default=1024, type=int,
                       help='dimension of representation in each layer')
    group.add_argument('--dropout', default=0.1, type=float,
                       help='probability that each element is dropped')
    group.add_argument('--layer_norm_epsilon', default=1e-5,
                       type=float, help='layer norm epsilon value')
    group.add_argument('--initializer_range', default=0.02,
                       type=float, help='initilaizer range value')

    group = parser.add_argument_group('Training and evaluation')
    group.add_argument('--batch_train', default=64, type=int,
                       help='number of training batch size')
    group.add_argument('--batch_eval', default=64, type=int,
                       help='number of evaluation batch size')
    group.add_argument('--base_lr', default=1e-4, type=float,
                       help='default learning rate')
    group.add_argument('--wd_rate', default=1e-2, type=float,
                       help='weight decay rate')
    group.add_argument('--total_steps', default=1000000, type=int,
                       help='number of total training steps')
    group.add_argument('--eval_steps', default=500, type=int,
                       help='period to evaluate model and record metrics')
    group.add_argument('--save_steps', default=1000, type=int,
                       help='period to save training state to checkpoint')

    group = parser.add_argument_group('Saving and restoring')
    group.add_argument('--save_model_path', default='model.pth',
                       help='save trained model weights to the file')
    group.add_argument('--save_checkpoint_path', default='checkpoint.pth',
                       help='save training state to the checkpoint file')
    group.add_argument('--from_checkpoint', default=None,
                       help='load last training state from checkpoint file')
    group.add_argument('--from_pretrained', default=None,
                       help='initialize parameters from pretrained model')

    group = parser.add_argument_group('Extensions')
    group.add_argument('--use_amp', action='store_true',
                       help='use automatic mixed-precision in training')
    group.add_argument('--gpus', default=None, type=int,
                       help='number of gpu devices to use in training')

    parser.set_defaults(func=train_gpt2_model)
