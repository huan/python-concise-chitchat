'''chit chat'''
from .chit_chat import ChitChat
from .data_loader import DataLoader
from .vocabulary import Vocabulary

from .config import (
    BATCH_SIZE,
    MAX_LEN,
    END_TOKEN,
    LEARNING_RATE,
    PAD_TOKEN,
)

name = 'chit-chat'

__all__ = [
    'BATCH_SIZE',
    'MAX_LEN',
    'END_OTKEN',
    'LEARNING_RATE',
    'PAD_TOKEN',

    'ChitChat',
    'DataLoader',
    'Vocabulary',
]
