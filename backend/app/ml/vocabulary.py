import string
from typing import List, Dict

# Connectionist Temporal Classification (CTC) Character Vocabulary
# Index 0 is strictly reserved as the CTC Blank token
CHARS = ["<blank>", " ", "'"] + list(string.ascii_lowercase) + ["<unk>"]
CHAR_TO_ID: Dict[str, int] = {char: idx for idx, char in enumerate(CHARS)}
ID_TO_CHAR: Dict[int, str] = {idx: char for idx, char in enumerate(CHARS)}

VOCAB_SIZE = len(CHARS)
BLANK_ID = 0
SPACE_ID = 1

def text_to_ids(text: str) -> List[int]:
    """Converts a string of English text to token IDs."""
    text = text.lower()
    ids = []
    for c in text:
        if c in CHAR_TO_ID:
            ids.append(CHAR_TO_ID[c])
        elif c == " ":
            ids.append(SPACE_ID)
        else:
            ids.append(CHAR_TO_ID["<unk>"])
    return ids

def ids_to_text(ids: List[int]) -> str:
    """Decodes token IDs into a string."""
    tokens = []
    for i in ids:
        if i in ID_TO_CHAR and i not in (BLANK_ID, CHAR_TO_ID["<unk>"]):
            tokens.append(ID_TO_CHAR[i])
    return "".join(tokens)