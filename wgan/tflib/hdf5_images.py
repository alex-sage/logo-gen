import numpy as np
import scipy.misc
import time
import h5py

def make_generator(hdf5_file, n_images, batch_size, res, label_name=None):
    epoch_count = [1]
    def get_epoch():
        # print('new epoch!')
        images = np.zeros((batch_size, 3, res, res), dtype='int32')
        labels = np.zeros(batch_size, dtype='int32')
        indices = range(n_images)
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(indices)
        epoch_count[0] += 1
        for n, i in enumerate(indices):
            # assuming (N)CHW format
            images[n % batch_size] = hdf5_file['data'][i]
            if label_name is not None:
                labels[n % batch_size] = hdf5_file[label_name][i]
            if n > 0 and n % batch_size == 0:
                yield (images, labels)
    return get_epoch


def load(batch_size, data_file='/home/sagea/scratch/data/icons/icon_data.hdf5', resolution=32, label_name=None):
    hdf5_file = h5py.File(data_file, 'r')
    n_images = len(hdf5_file['data'])
    if label_name is not None:
        n_labels = len(hdf5_file[label_name])
        print('Labels: %i' % n_labels)
        n_images = min(n_images, n_labels)
    print('Images: %i' % n_images)
    return make_generator(hdf5_file, n_images, batch_size, res=resolution, label_name=label_name)


def load_new(cfg):
    label_name = cfg.LABELS if cfg.LABELS != 'None' else None
    return load(cfg.BATCH_SIZE, cfg.DATA, cfg.OUTPUT_RES, label_name=label_name)


if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()