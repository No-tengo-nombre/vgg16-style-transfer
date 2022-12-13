from vgg16st.arg_parser import PARSER
from vgg16common.logger import setup_logger


args = PARSER.parse_args()
setup_logger(args.quiet, args.debug, args.verbose, args.log)
args.main_func(args)
