import numpy as np
from DataConverter import getAllData
import os
from sklearn.cross_validation import train_test_split
import h5py
from scipy import signal
import librosa

def to_categorical(y, num_classes):
    """Converts a class vector (integers) to binary class matrix.
    
    Args:
      y: class vector to be converted into a matrix
        (integers from 0 to num_classes).
      num_classes: total number of classes.
    
    Returns:
      A binary matrix representation of the input. The classes axis
      is placed last.
    """
    y = np.array(y, dtype='int')
    y = y.ravel()
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = y.shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

### change workspace to your own if needed 
# workspace = os.path.join('C:','Users','iwonka','Desktop', 'DNN_emg')
workspace = os.getcwd()
tmp_path = os.path.join(workspace, 'tmp')
if not os.path.isdir(tmp_path):
    os.makedirs(tmp_path)

hdf5 = os.path.join(tmp_path,'dataset.hdf5')

fs = 1000
nfft = 512
win_len = 64

gest_all = getAllData('data') 
y=[]

gest_time = [np.delete(g, [0,2,4], axis=2) for g in gest_all]
for idx, g in enumerate(gest_time):
    y.append(np.ones(g.shape[0])*idx)

gest_flat = np.vstack(gest_time)
y_flat = np.hstack(y)

# extract spectrograms 
gest_spec = [[np.log(np.abs(librosa.core.stft(x, nfft, win_len))) for x in x_all.T] for x_all in gest_flat]           
y_cat = [to_categorical(y,len(gest_all)) for y in y_flat]

X_train, X_test, y_train, y_test = train_test_split(np.array(gest_spec), np.squeeze(np.array(y_cat)), test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

hdf5_file = h5py.File(hdf5, mode="w")
hdf5_file.create_dataset('X_train',data=X_train)
hdf5_file.create_dataset('X_val',data=X_val)
hdf5_file.create_dataset('X_test',data=X_test)
hdf5_file.create_dataset('y_test',data=y_test)
hdf5_file.create_dataset('y_val',data=y_val)
hdf5_file.create_dataset('y_train',data=y_train)
hdf5_file.close()

