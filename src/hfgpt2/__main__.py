import argparse
from hfgpt2 import train_model, visualize_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='gpt2',
        description='Pytorch Training Code of Huggingface GPT2'
    )
    subparsers = parser.add_subparsers(dest='subcommands')

    train_model.add_subparsers(subparsers)
    visualize_metrics.add_subparser(subparsers)

    args = parser.parse_args()
    args.func(args)
