import platform
import kaggle


def is_windows():
    return platform.system() == 'Windows'


def is_64bit_ubuntu():
    uname = str(platform.uname())
    return ('Ubuntu' in uname) & ('Linux' in uname)


def test_download_archive():
    api = kaggle.KaggleApi('{"username":"headwater","key":"96e459c7c7353d1233d0d702292f5b0d"}')
    api.authenticate()
    if is_64bit_ubuntu():
        api.datasets_download(owner_slug='headwater', dataset_slug='Camels')
    elif is_windows():
        api.datasets_download(owner_slug='headwater', dataset_slug='Camels')
