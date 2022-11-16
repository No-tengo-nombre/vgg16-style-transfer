import re


DEPTH_PATTERN = re.compile(r"([\+\-])(\d*)")


def st_main(args):
    pass
    # depths = args.depth
    # match depths:
    #     case [x]:
    #         match DEPTH_PATTERN.search(x).groups():
    #             case ("+", d):
    #                 final_depths = 
