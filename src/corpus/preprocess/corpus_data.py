"""
Module providing the dataclass CorpusData
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class CorpusData:
    """
    CorpusData dataclass providing main corpus parameters
    """
    windowed: List[List[int]]
    vocabulary: List[str]
    word2idx: Dict[(str, int)]
    idx2word: Dict[(int, str)]
