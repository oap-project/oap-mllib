import os
import sys


def get_oap_path():
    current_module = sys.modules[__name__]
    current_path = os.path.dirname(current_module.__file__)
    return current_path

def get_jars_path():
    current_path = get_oap_path()
    jar_path = current_path + '/jars'
    if os.path.exists(jar_path):
        return jar_path
    else:
        print("error: can not find jars path")
        return None
