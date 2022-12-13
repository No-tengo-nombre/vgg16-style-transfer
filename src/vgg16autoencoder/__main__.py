from vgg16common.logger import setup_logger
from vgg16autoencoder.arg_parser import PARSER


args = PARSER.parse_args()
setup_logger(args.quiet, args.debug, args.verbose, args.log)
args.main_func(args)
