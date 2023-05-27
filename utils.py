import os
import os.path as op
import logging
import sys
import yaml
import argparse
import json


def init_logging():
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    logger_fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(process)d:%(filename)s:%(lineno)s %(funcName)10s(): %(message)s"
    )
    ch.setFormatter(logger_fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []
    root.addHandler(ch)


def load_from_yaml_str(s):
    return yaml.load(s, Loader=yaml.UnsafeLoader)


def parse_inference_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--param", help="parameter string, yaml format", type=str)
    args = parser.parse_args()
    kwargs = load_from_yaml_str(args.param)
    return kwargs


def ensure_directory(path):
    if path == "" or path == ".":
        return
    if path != None and len(path) > 0:
        assert not op.isfile(path), "{} is a file".format(path)
        if not os.path.exists(path) and not op.islink(path):
            try:
                os.makedirs(path)
            except:
                if os.path.isdir(path):
                    # another process has done makedir
                    pass
                else:
                    raise


def json_dump(obj):
    # order the keys so that each operation is deterministic though it might be
    # slower
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def write_to_file(contxt, file_name, append=False):
    p = os.path.dirname(file_name)
    ensure_directory(p)
    if type(contxt) is str:
        contxt = contxt.encode()
    flag = "wb"
    if append:
        flag = "ab"
    with open(file_name, flag) as fp:
        fp.write(contxt)
