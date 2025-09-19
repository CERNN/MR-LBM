from dataSave import *
from fileTreat import *
import math
import sys
path = "./../bin/" + sys.argv[1] + "/"

# Get the macroscopics in the folder
macrSteps = getMacrSteps(path)
info = getSimInfo(path)

# for all steps saved
for step in macrSteps:
    macr = getMacrsFromStep(step,path)
    # Save macroscopics to VTK format
    print("Processing step", step)
    saveVTK3D(macr, path, info['ID'] + "macr" + str(step).zfill(6), points=True)