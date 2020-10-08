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
    action="store_true",
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
argument_parser.add_argument(
    "--load_data",
    action="store_true",
    help="If provided, corpus module must have been already run and its results saved"
)
argument_parser.add_argument(
    "--logging",
    action="store_true",
    help="If provided, logging mode will be turned on; adds some more insights to the learning process, "
         "but may result in slower time of learning due to irrelevant side effects"
)
argument_parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    help="Provide the single mini-batch size"
    required=True
)
argument_parser.add_argument(
    "--epochs",
    action="store",
    type=int,
    help="Provide the amount of training iterations",
    required=True
)
argument_parser.add_argument(
    "--eta",
    action="store",
    type=float,
    help="Provide the learning rate for the optimizer to adjust weights after learning",
    required=True
)
argument_parser.add_argument(
    "-g",
    "--gradient_normalization",
    action="store",
    type=int,
    help="Provide the gradient normalization factor",
    required=True
)

args = argument_parser.parse_args()
