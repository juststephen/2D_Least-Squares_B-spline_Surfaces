from glob import glob
import mat73
import numpy as np
import scipy.io as sio

# Files
files = glob('D:/MBESData/*ShipNameU5.mat')

# Selected area by Dr.
if False:
    files = [
        'D:/MBESData\\0001_20171031_200133_ShipNameU5.mat',
        'D:/MBESData\\0007_20171031_225633_ShipNameU5.mat',
        'D:/MBESData\\0031_20171031_012959_ShipNameU5.mat',
        'D:/MBESData\\0035_20171103_190504_ShipNameU5.mat',
        'D:/MBESData\\0037_20171031_043532_ShipNameU5.mat',
        'D:/MBESData\\0038_20171031_050532_ShipNameU5.mat',
        'D:/MBESData\\0038_20171103_203944_ShipNameU5.mat',
        'D:/MBESData\\0039_20171103_211008_ShipNameU5.mat'
    ]
# 1 x 1 [km]
if True:
    files = [
        'D:/MBESData\\0001_20171031_200133_ShipNameU5.mat',
        'D:/MBESData\\0004_20171102_211547_ShipNameU5.mat',
        'D:/MBESData\\0031_20171031_012959_ShipNameU5.mat',
        'D:/MBESData\\0035_20171103_190504_ShipNameU5.mat',
        'D:/MBESData\\0037_20171031_043532_ShipNameU5.mat',
        'D:/MBESData\\0038_20171031_050532_ShipNameU5.mat',
        'D:/MBESData\\0038_20171103_203944_ShipNameU5.mat',
        'D:/MBESData\\0039_20171103_211008_ShipNameU5.mat',
        'D:/MBESData\\0042_20171103_223551_ShipNameU5.mat'
    ]
# 100 x 100 [m]
if False:
    files = [
        'D:/MBESData\\0001_20171031_200133_ShipNameU5.mat',
        'D:/MBESData\\0035_20171103_190504_ShipNameU5.mat'
    ]

sizes = range(100, 1000+100, 100)
for i in sizes:
    # Boundaries
    # Easting
    xmin = 524000 
    xmax = xmin + i
    # Northing
    ymin = 5826000
    ymax = ymin + i

    # Empty np.ndarray for x, y and z
    x = np.array([])
    y = np.array([])
    z = np.array([])

    # List for storing files that contain data
    used_files = []

    # Iterate over files and get coordinates within the boundaries
    for filename in files:
        # Loading MatLab file
        try:
            data = sio.loadmat(filename)
        except NotImplementedError:
            # Mat v7.3 file format
            data = mat73.loadmat(filename, verbose=False)
            data = data['data']

        easting = data['easting']
        northing = data['northing']
        depth = data['depth']

        # Get indices of selected coordinates within the boundaries
        indices = np.nonzero(
            ( easting >= xmin ) &
            ( easting <= xmax ) &
            ( northing >= ymin ) &
            ( northing <= ymax )
        )[0]

        # Add found coordinates to the x, y and z arrays
        if indices.size:
            # FiLe contains points -> store filename
            used_files.append(filename)

            # Concat to coordinates
            x = np.concatenate((x, easting[indices]))
            y = np.concatenate((y, northing[indices]))
            z = np.concatenate((z, depth[indices]))

    # Saving to file
    np.save(f'selected_area_{i}m.npy', (x, y, z))

    # Can be used later for faster operation
    print(used_files)
