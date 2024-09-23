
from termcolor import colored

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colorize(statement, color='blue', bold=True, underline=False):
    statement = colored(statement, color)

    if bold:
        statement = bcolors.BOLD + statement
    if underline:
        statement = bcolors.UNDERLINE + statement



    return statement + bcolors.ENDC


def random_colors(N, bright=True):
    #https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
    import colorsys, random
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

