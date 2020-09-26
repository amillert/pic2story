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
    "--path",
    action="store",
    type=str,
    help="Provide images paths to generate story",
    required=True,
    nargs="+"
)

args = argument_parser.parse_args()
