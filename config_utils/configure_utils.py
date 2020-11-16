import os, json
import pickle
from collections import namedtuple

supporting_type = ('str', 'int', 'bool', 'float', 'none')

def convert_param(config_list):
    assert isinstance(config_list, list), 'invalid type = {:}'.format(config_list)
    ctype, value = config_list[0], config_list[1]
    assert ctype in supporting_type, 'Ctype={:}, support={:}'.format(ctype, supporting_type)
    is_list = isinstance(value, list)
    if not is_list:
        value = [value]
    outs = []
    for x in value:
        if ctype == 'str':
            x = str(x)
        elif ctype == 'int':
            x = int(x)
        elif ctype == 'bool':
            x = bool(int(x))
        elif ctype == 'float':
            x = float(x)
        elif ctype == 'none':
            x = None
        else:
            raise ValueError('Does not know this type={:}'.format(ctype))
        outs.append(x)
    if not is_list:
        outs = outs[0]
    return outs

        

def load_config(path, extra):
    path = str(path)
    assert os.path.exists(path), 'invalid path = {:}'.format(path)
    with open(path, 'r') as f:
        data = json.load(f)
    content = {k:convert_param(v) for k,v in data.items()}
    assert extra is None or isinstance(extra, dict), 'invalid extra type = {:}'.format(type(extra))
    if isinstance(extra, dict):
        content = {**content, **extra}
    Arguments = namedtuple('Configure', ' '.join(content.keys()))
    content = Arguments(**content)
    return content
