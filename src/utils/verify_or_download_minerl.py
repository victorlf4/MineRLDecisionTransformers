import minerl
import minerl.data
import os
def verify_or_download_dataset(environment=None,directory=None):
    data_missing=True
    try:
        data = minerl.data.make(environment, data_dir=directory)
        assert len(data._get_all_valid_recordings(data.data_dir)) > 0
        data_missing=False
    except FileNotFoundError:
        print("The data directory does not exist in your submission, are you running this script from"
              " the root of the repository? data_dir={}".format(directory))
    except RuntimeError:
        print("The data contained in your data directory is out of date! data_dir={}".format(data_dir))
    except AssertionError:
        print(f"No {environment} data found.")
    
    if data_missing:
        print("Attempting to download the dataset...")
        minerl.data.download(directory=directory, environment=environment)
        data = minerl.data.make(environment, data_dir=directory)
    return data