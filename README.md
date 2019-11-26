# practical-machine-learning

## Forky Spoony and Knifey Image Classification

### Downloading the Dataset

Make sure you've got the `kaggle` CLI tool installed using the API token associated to your account.
```zsh
cd 
kaggle datasets download kilianovski/forky-spoony-and-knifey
unzip forky-spoony-and-knifey.zip -d dataset
rm -r dataset/forky-dataset
 ```

 ### Restoring Environment Considerations

 If `h5py` mentions a version mismatch, you can set `HDF5_DISABLE_VERSION_CHECK` environment variable to `1` and ignore it.

 To check for available CPUs/GPUs, use the following statements
 ```python
 from keras import backend as K
 K.tensorflow_backend._get_available_gpus()
 ```
 ```python
 from tensorflow.python.client import device_lib
 device_lib.list_local_devices()
 ```