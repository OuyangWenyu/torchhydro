import kaggle


def test_download_archive():
    json_str = '{"username":"headwater","key":"96e459c7c7353d1233d0d702292f5b0d"}'
    api = kaggle.KaggleApi(json_str)
    api.authenticate()
    api.datasets_download(owner_slug='headwater', dataset_slug='Camels')

