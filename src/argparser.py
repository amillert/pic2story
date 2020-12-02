"""
Module with the whole argparser
"""
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
    help="Provide images paths to generate story "
         "(preferebly folder in which text files can be found recursively)",
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
    help="Provide confidence to filter out irrelevant "
         "objects' predictions from the YOLO object detection",
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
    "--window",
    action="store",
    type=int,
    help="Provide the size of window to generate subsentences",
    required=True
)
argument_parser.add_argument(
    "--read_corpus",
    action="store",
    type=str,
    help="If provided, unpickles the corpus object from the file; "
         "if doesn't work, standard procedure will kick in"
)
argument_parser.add_argument(
    "--save_corpus",
    action="store",
    type=str,
    help="If provided, pickles the corpus object to the file"
)
argument_parser.add_argument(
    "--load_data",
    action="store_true",
    help="If provided, corpus module must have been already run and its results saved"
)
argument_parser.add_argument(
    "--logging",
    action="store_true",
    help="If provided, logging mode will be turned on; adds some more insights "
         "to the learning process, but may result in slower time of learning "
         "due to irrelevant side effects"
)
argument_parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    help="Provide the single mini-batch size",
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
argument_parser.add_argument(
    "--save_weights",
    action="store_true",
    help="If provided, weights will be saved for future access"
)
argument_parser.add_argument(
    "--load_pretrained",
    action="store",
    type=str,
    help="If provided, embedding layer in the model will be loaded with the pretrained weights"
)
argument_parser.add_argument(
    "--hidden",
    action="store",
    type=int,
    help="hidden size",
    required=True
)
argument_parser.add_argument(
    "--layers",
    action="store",
    type=int,
    help="layers",
    required=True
)
argument_parser.add_argument(
    "--drop_prob",
    action="store",
    type=float,
    help="drop probs",
    required=True
)
argument_parser.add_argument(
    "--books",
    action="store",
    type=int,
    help="Amount of books to read; more books more RAM used "
         "(32GB not sufficient for whole dataset)",
    required=False
)
argument_parser.add_argument(
    "--synonyms",
    action="store",
    type=str,
    help="Provide synonyms to the labels of the provided pictures for an evalutaion",
    required=False,
    nargs="+"
)

args = argument_parser.parse_args()
