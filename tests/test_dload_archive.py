import platform
import kaggle
import json


def is_windows():
    return platform.system() == 'Windows'


def is_64bit_ubuntu():
    uname = str(platform.uname())
    return ('Ubuntu' in uname) & ('Linux' in uname)


def test_download_archive():
    json_str = '{"username":"headwater","key":"96e459c7c7353d1233d0d702292f5b0d"}'
    api = kaggle.KaggleApi(json_str)
    api.authenticate()
    if is_64bit_ubuntu():
        with open('/home/runner/.kaggle/kaggle.json', 'w') as fp:
            fp.write(json_str)
        api.datasets_download(owner_slug='headwater', dataset_slug='Camels')
    elif is_windows():
        with open('C:\\Users\\runneradmin\\.kaggle\\kaggle.json', 'w') as fp:
            fp.write(json_str)
        api.datasets_download(owner_slug='headwater', dataset_slug='Camels')
