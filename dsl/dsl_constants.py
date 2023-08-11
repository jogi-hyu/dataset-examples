from enum import Enum

class Type(Enum):
    NUMPY   = 1001
    PIL     = 1002

class Set(Enum):
    TRAIN   = 2001
    VAL     = 2002
    TEST    = 2003