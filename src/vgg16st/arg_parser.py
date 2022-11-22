import argparse

from vgg16st.main import st_main
from vgg16common import DEFAULT_ST_SQUARE_SIZE


desc_str = """Perform a style transfer using the VGG16 network."""

PARSER = argparse.ArgumentParser(prog="vgg16st", description=desc_str)
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


### Style transfer parser ###
st_epilog_str = """\
The way the program interprets the depths is the following:
 - If a list of numbers is given, they are interpreted as the
   the depths to use. This also applies to the case that a single
   number is given.
 - If a single number is given and is preppended with "-" or "+",
   then it has special behaviour. If it is preppended with "-",
   then all depths up to and including that one are used. If it
   is preppended with "+", then all depths from and including
   that one are used.

   Examples
    - `vgg16st st ... -D -3` means use depths 1, 2 and 3.
    - `vgg16st st ... -D +3` means use depths 3, 4 and 5.
"""
st_parser = subparser.add_parser("style_transfer", help="perform a style transfer", aliases=("st",))
st_parser.set_defaults(main_func=st_main)
st_parser.add_argument(
    "content",
    action="store",
    type=str,
    help="content image to transfer the style to",
)
st_parser.add_argument(
    "style",
    action="store",
    type=str,
    help="style image to use",
)
st_parser.add_argument(
    "-D",
    "--depth",
    action="store",
    nargs="+",
    type=str,
    default=["5-"],
    help="layers to use for the transfer",
)
st_parser.add_argument(
    "-p",
    "--plot",
    action="store_true",
    help="show the images",
)
st_parser.add_argument(
    "-S",
    "--save",
    action="store",
    type=str,
    default="",
    help="save the images to the target directory",
)
st_parser.add_argument(
    "-a",
    "--alpha",
    action="store",
    type=float,
    default=1.0,
    help="determine the alpha for blending",
)
st_parser.add_argument(
    "--gpu",
    action="store_true",
    help="use the gpu for the style transfer",
)
st_parser.add_argument(
    "--square-size",
    action="store",
    type=int,
    default=DEFAULT_ST_SQUARE_SIZE,
    help="square size to use for the content and style image",
)
st_parser.add_argument(
    "--method",
    action="store",
    type=str,
    default="paper",
    help="method to use for the WCT",
)
