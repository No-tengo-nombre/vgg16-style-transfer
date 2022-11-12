import argparse


desc_str = """VGG16 training and evaluation code."""

PARSER = argparse.ArgumentParser(description=desc_str)
PARSER.add_argument(
    "-T",
    "--train",
    action="store_true",
    help="Train the model.",
)
PARSER.add_argument(
    "-E",
    "--evaluate",
    action="store",
    help="Evaluate the model.",
)
PARSER.add_argument(
    "-d",
    "--debug",
    action="store",
    default=False,
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
PARSER.add_argument(
    "-b",
    "--batch-size",
    action="store",
    nargs=2,
    default=(1, 1),
    help="Set the training and validation batch size.",
)
PARSER.add_argument(
    "-e",
    "--epochs",
    action="store",
    default=1,
    help="Set the number of epochs.",
)
PARSER.add_argument(
    "-s",
    "--show-curves",
    action="store_true",
    help="Show the training curves.",
)
PARSER.add_argument(
    "-S",
    "--save-weights",
    action="store",
    default="",
    help="Save the model's weights.",
)
PARSER.add_argument(
    "-w",
    "--from-weights",
    action="store",
    nargs="?",
    default=False,
    help="Load the model from weights.",
)
PARSER.add_argument(
    "-t",
    "--train-split",
    action="store",
    type=float,
    default=0.7,
    help="Set the training split.",
)
PARSER.add_argument(
    "-N",
    "--never-save",
    action="store_true",
    help="Never save the model.",
)
PARSER.add_argument(
    "-D",
    "--depth",
    action="store",
    type=int,
    default=5,
    help="Set the depth of the encoder and decoder.",
)
