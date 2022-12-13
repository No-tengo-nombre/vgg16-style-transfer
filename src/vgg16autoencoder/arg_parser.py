import argparse

from vgg16autoencoder import BASE_DIR
from vgg16autoencoder.main import train_main, eval_main


desc_str = """VGG16 encoder-decoder training and evaluation code."""

PARSER = argparse.ArgumentParser(prog="vgg16autoencoder", description=desc_str)
PARSER.set_defaults(main_func=lambda *_: PARSER.print_help())
PARSER.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="run in debug mode",
)
PARSER.add_argument(
    "--log",
    action="store",
    default=f"{BASE_DIR}/logs",
    help="generate log files for the current run in the given directory",
)
PARSER.add_argument(
    "-q",
    "--quiet",
    action="store_true",
    help="disable logging",
)
PARSER.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="run in verbose mode",
)

subparser = PARSER.add_subparsers(dest="subparser_name")


### Training parser ###
train_parser = subparser.add_parser("train", help="train the network", aliases=("t"))
train_parser.set_defaults(main_func=train_main)
train_parser.add_argument(
    "-r",
    "--reduced",
    action="store",
    type=int,
    help="run the training with a reduced dataset",
)
train_parser.add_argument(
    "-b",
    "--batch-size",
    action="store",
    nargs=2,
    default=(1, 1),
    help="set the training and validation batch size",
)
train_parser.add_argument(
    "-e",
    "--epochs",
    action="store",
    default=1,
    help="set the number of epochs",
)
train_parser.add_argument(
    "-s",
    "--show-curves",
    action="store_true",
    help="show the training curves",
)
train_parser.add_argument(
    "-S",
    "--save-weights",
    action="store",
    default="",
    help="save the model's weights",
)
train_parser.add_argument(
    "-w",
    "--from-weights",
    action="store",
    nargs="?",
    default=False,
    help="load the model from weights",
)
train_parser.add_argument(
    "-t",
    "--train-split",
    action="store",
    type=float,
    default=0.7,
    help="set the training split",
)
train_parser.add_argument(
    "-N",
    "--never-save",
    action="store_true",
    help="never save the model",
)
train_parser.add_argument(
    "-D",
    "--depth",
    action="store",
    type=int,
    default=5,
    help="set the depth of the encoder and decoder",
)


### Evaluation parser ###
eval_parser = subparser.add_parser(
    "eval", help="evaluate the encoder and decoder", aliases=("e")
)
eval_parser.set_defaults(main_func=eval_main)
