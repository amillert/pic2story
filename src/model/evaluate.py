"""
Module responsible for calculating measures based on intersection with synonyms
"""
import re


def score(text, synonyms):
    """
    Function for computing (precision / recall / f-1)-ish scores

    :param text: str
    :param synonyms: list[str]
    :return: None
    """
    s_t = set([x.lower() for x in re.split(r"\W", text) if x])
    s_s = set([x.lower() for x in synonyms])

    precision = len(s_t & s_s) / len(s_t | s_s)
    recall = len(s_t & s_s) / len(s_t)
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f"precision  -> {precision}")
    print(f"recall     -> {recall}")
    print(f"f score    -> {f1}")
