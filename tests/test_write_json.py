import os
import platform


def is_windows():
    return platform.system() == 'Windows'


def is_64bit_ubuntu():
    uname = str(platform.uname())
    return ('Ubuntu' in uname) & ('Linux' in uname)


def test_wjson():
    json_str = '{"username":"headwater","key":"96e459c7c7353d1233d0d702292f5b0d"}'
    if is_64bit_ubuntu():
        os.mkdir("/home/runner/.kaggle")
        with open('/home/runner/.kaggle/kaggle.json', 'w+') as fp:
            fp.write(json_str)
    elif is_windows():
        os.mkdir("/home/runner/.kaggle")
        with open('C:\\Users\\runneradmin\\.kaggle\\kaggle.json', 'w+') as fp:
            fp.write(json_str)
