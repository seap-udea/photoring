from photoring.version import *
import os

#Root directory
try:
    FILE=__file__
    ROOTDIR=os.path.abspath(os.path.dirname(FILE))
except:
    FILE=""
    ROOTDIR=os.path.abspath('')

DATA_DIR = os.path.join(ROOTDIR, 'data')

from photoring.geotrans import *
from photoring.montecarlo import *
from photoring.plot import *

version = '0.2.0'
print(f"Photoring version {version}")
