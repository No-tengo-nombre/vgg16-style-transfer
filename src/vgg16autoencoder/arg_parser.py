import argparse

from vgg16autoencoder.main import train_main, eval_main


desc_str = """VGG16 encoder-decoder training and evaluation code."""

PARSER = argparse.ArgumentParser(description=desc_str)
PARSER.set_defaults(main_func=lambda *_: PARSER.print_help())
PARSER.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="Run in debug mode.",
)
PARSER.add_argument(
    "--log",
    action="store",
    help="Generate log files for the current run in the given directory.",
)
PARSER.add_argument(
    "-q",
    "--quiet",
    action="store_true",
    help="Disable logging.",
)
PARSER.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Run in verbose mode.",
)

subparser = PARSER.add_subparsers(dest="subparser_name")


### Training parser ###
train_parser = subparser.add_parser("train", help="Train the network.", aliases=("t"))
train_parser.set_defaults(main_func=train_main)
train_parser.add_argument(
    "-r",
    "--reduced",
    action="store",
    type=int,
    help="Run the training with a reduced dataset.",
)
train_parser.add_argument(
    "-b",
    "--batch-size",
    action="store",
    nargs=2,
    default=(1, 1),
    help="Set the training and validation batch size.",
)
train_parser.add_argument(
    "-e",
    "--epochs",
    action="store",
    default=1,
    help="Set the number of epochs.",
)
train_parser.add_argument(
    "-s",
    "--show-curves",
    action="store_true",
    help="Show the training curves.",
)
train_parser.add_argument(
    "-S",
    "--save-weights",
    action="store",
    default="",
    help="Save the model's weights.",
)
train_parser.add_argument(
    "-w",
    "--from-weights",
    action="store",
    nargs="?",
    default=False,
    help="Load the model from weights.",
)
train_parser.add_argument(
    "-t",
    "--train-split",
    action="store",
    type=float,
    default=0.7,
    help="Set the training split.",
)
train_parser.add_argument(
    "-N",
    "--never-save",
    action="store_true",
    help="Never save the model.",
)
train_parser.add_argument(
    "-D",
    "--depth",
    action="store",
    type=int,
    default=5,
    help="Set the depth of the encoder and decoder.",
)


### Evaluation parser ###
eval_parser = subparser.add_parser("eval", help="Evaluate the encoder and decoder.", aliases=("e"))
eval_parser.set_defaults(main_func=eval_main)
