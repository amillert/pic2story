from dataclasses import dataclass
from typing import List, Dict


@dataclass
class CorpusData:
    windowed: List[List[int]]
    vocabulary: List[str]
    word2idx: Dict[(str, int)]
    idx2word: Dict[(int, str)]
