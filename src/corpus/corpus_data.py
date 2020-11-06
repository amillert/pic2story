from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple


@dataclass
class CorpusData:
    # ngrams: List[List[Iterable[Tuple[str, List[str]]]]]
    windowed: List[List[int]]
    vocabulary: List[str]
    word2idx: Dict[(str, int)]
    idx2word: Dict[(int, str)]
