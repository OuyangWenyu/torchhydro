import io
import os.path
import platform
import requests as req
import zipfile as zif


def is_windows():
    return platform.system() == 'Windows'


def is_64bit_ubuntu():
    uname = str(platform.uname())
    return ('Ubuntu' in uname) & ('Linux' in uname)


def test_download_archive():
    url = 'https://storage.googleapis.com/kaggle-data-sets/3595884/6256802/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog' \
          '-Credential=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com/20230806/auto/storage/goog4_request&X-Goog-Date' \
          '=20230806T094350Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature' \
          '=3210ad95ea59c4d9eba14020b626cf830443b314fefe695e11d298367d97b6b9e1722cece73541d9112bbdd31d2b4118f74cb16e44e60c2ebc86e2225225d1653d20ee724c2bde17a3630498b52437504cba44ca1aff8a016de8a169a4888115c32a85f511a1ed2785cd6ee619b86d1cff0f9710c0ee7e9e754d1f71d603a3e13e863c2b0328405d229fb11d47c31a1490d5a962ac5b5bbf553eca6c86ea7c8a859ebf54d102497a866f995b826848c9b7c782421c2e63d590878d4bb3fdd6ed64472ba794920c0389c4896aab701b07344fc1dd6c8540a7ee4829cd67937dc3e70500ab9b016fb47df5deb2fe2ec4f6a714b13085c9beb9e678a4248e5e8a95'
    if is_64bit_ubuntu():
        res = req.request(method='GET', url=url)
        zip_file = zif.ZipFile(io.BytesIO(res.content), 'r')
        zif.ZipFile.extractall(zip_file, path=os.path.abspath('/home/runner/.hydrodataset/cache/camels/camels_us/'))
    elif is_windows():
        res = req.request(method='GET', url=url)
        zip_file = zif.ZipFile(io.BytesIO(res.content), 'r')
        zif.ZipFile.extractall(zip_file, path='datas')


