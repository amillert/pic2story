import argparse

argument_parser = argparse.ArgumentParser(
    prog="pic2story",
    description="Argument parser of the Pic2Story",
    epilog="Generate a story for your kids while enjoying a warm cup of coffee ☕",
    allow_abbrev=True
)
argument_parser.version = "0.1"
argument_parser.add_argument(
    "-p",
    "--paths_img",
    action="store",
    type=str,
    help="Provide images paths to generate story",
    required=True,
    nargs="+"
)
argument_parser.add_argument(
    "--train",
    action="store",
    help="If provided, the training pipeline process will be evaluated"
)
argument_parser.add_argument(
    "-l",
    "--labels",
    action="store",
    type=str,
    help="Provide file's path with labels names",
    required=True
)
argument_parser.add_argument(
    "-n",
    "--net_config",
    action="store",
    type=str,
    help="Provide file's path with object detection's network config",
    required=True
)
argument_parser.add_argument(
    "-t",
    "--transfer_weights",
    action="store",
    type=str,
    help="Provide file's path with networks pretrained weights (transfer learning)",
    required=True
)
argument_parser.add_argument(
    "-c",
    "--confidence",
    action="store",
    type=float,
    help="Provide confidence to filter out irrelevant objects' predictions from the YOLO object detection",
    required=True
)
argument_parser.add_argument(
    "-b",
    "--paths_books",
    action="store",
    type=str,
    help="Provide children books paths to build corpus",
    required=True,
    nargs="+"
)
argument_parser.add_argument(
    "--ngrams",
    action="store",
    type=int,
    help="Provide the amount of context words (per side -> e.g. if you want to consider 4 neighbour-words, use - 2)"
         " around the target word to create ngrams",
    required=True
)

args = argument_parser.parse_args()
