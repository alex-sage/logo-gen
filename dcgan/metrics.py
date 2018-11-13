import numpy as np
from scipy.spatial.distance import cdist
from scipy.misc import imresize


def nearest_icons(icons, dataset, scale=None, grayscale=False, get_dist=False):
    """ Find nearest neighbour in the set for a given icon
        Args:
            icons: array of icons to which the nearest neighbour is to be found (N_i,y_res,x_res,chan)
            dataset: array of original icons which is searched (N_s,y_res,x_res,chan)
            scale: integer resolution to which icons are rescaled before comparing (optional)
            grayscale: boolean if true, channel values are summed before comparing
        Returns index of nearest neighbour in set """
    if len(icons.shape) == 3:
        icons = icons.reshape((1,) + icons.shape)

    if scale is not None:
        icons = np.array([imresize(icon, (scale, scale)) for icon in icons])
        dataset = np.array([imresize(icon, (scale, scale)) for icon in dataset])

    if grayscale and (icons.shape[3] == 3):
        # average over all colour channels and flatten
        icons = [[sum(pixel) / 3 for pixel in icon]
                 for icon in icons.reshape((icons.shape[0], icons.shape[1] * icons.shape[2], 3))]
        dataset = [[sum(pixel) / 3 for pixel in icon]
                 for icon in dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2], 3))]
    else:
        icons = icons.reshape((icons.shape[0], icons.shape[1] * icons.shape[2] * icons.shape[3]))
        dataset = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2] * dataset.shape[3]))
    # array of shape (n_icons, n_data) containing the distances
    dist = cdist(icons, dataset, metric='euclidean')
    # get the index of the corresponding data_icon with lowest distance for each icon
    idxs = [np.argmin(d) for d in dist]
    # flag get_dist determines if distances are returned or not
    if not get_dist:
        return idxs
    else:
        # range(len(dist)) selects each row individually, and for each row the index in idxs
        return idxs, dist[range(len(dist)), idxs]

def icon_dist(icons1, icons2):
    icons1 = icons1.reshape((icons1.shape[0], icons1.shape[1] * icons1.shape[2] * icons1.shape[3]))
    icons2 = icons2.reshape((icons2.shape[0], icons2.shape[1] * icons2.shape[2] * icons2.shape[3]))
    dist = cdist(icons1, icons2, metric='euclidean')
    return dist
