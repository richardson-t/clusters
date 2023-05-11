from __future__ import unicode_literals, division, print_function

import random
import string
import os
import sys


def combinations(list1, list2):
    "Compute combinations of two lists of strings"
    result = []
    for item1 in list1:
        result += [item1 + item2 for item2 in list2]
    return result


def random_id(length=8):
    chars = string.ascii_letters + string.digits
    s = ""
    for i in range(length):
        s += random.choice(chars)
    return s


def create_dir(dir_name):
    delete_dir(dir_name)
    os.system("mkdir %s " % dir_name)


def delete_dir(dir_name):
    if os.path.exists(dir_name):
        reply = raw_input("Delete directory %s? [y/[n]] " % dir_name)
        if reply == 'y':
            os.system("rm -r %s" % dir_name)
        else:
            print("Aborting...")
            sys.exit()
