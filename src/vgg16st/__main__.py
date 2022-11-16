from vgg16autoencoder.logger import PARSER, setup_logger


args = PARSER.parse_args()
setup_logger(args.quiet, args.debug, args.verbose, args.log)
args.main_func(args)