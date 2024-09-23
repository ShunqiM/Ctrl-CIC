import argparse
import json
from utils.utils_color import colorize

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to config file.")
    parser.add_argument("-W","--wandb", action="store_true", help="Whether to use wandb.")
    parser.add_argument("-D","--debug", action="store_true", help="Whether to use debug mode.")
    args, remaining_args = parser.parse_known_args()

    if args.debug:
        assert not args.wandb, 'debug mode and wandb are mutually exclusive'
        print(colorize("DEBUG_QRUN is set to True"))

    return args, remaining_args


def parse_undefined_args(args):
    params = []
    for i in range(0, len(args)):
        arg = args[i].strip("--")
        assert '=' in arg, 'All undefined args must be in the form --arg=value'
        arg = arg.split('=')
        value = arg[1]
        arg = arg[0]

        # Make them json parseable
        json_value = value.replace("True", "true").replace("False", "false").replace("None", "null")

        try:
            tmp_value = json.loads(json_value)
            # In case there were any string values of True, False or None use the original value
            # rather than the json_value
            if isinstance(tmp_value, str):
                value = json.loads(value)
            else:
                value = tmp_value
        except json.JSONDecodeError:
            pass  # It can't be parsed as some other type so it must be a string
        params.append((arg, value))

    return params
