#!/usr/bin/env python3

import re
import sys
from typing import Tuple

import pytest                   # type: ignore


def version() -> Tuple[int, int, int]:
    try:
        ver = re.findall('^\d+\.\d+\.\d+', sys.version)[0]
        senior, minor, patch = re.findall('\d+', ver)
        return (int(senior), int(minor), int(patch))
    except:
        return (0, 0, 0)


# major, minor, patch = version()

def version_test():
    assert sys.version_info >= (3, 6), 'Python 3.6 or above is required.'
