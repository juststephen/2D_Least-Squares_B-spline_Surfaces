import numpy as np

# Loading from file
x, y, z = np.load('selected_area_500m.npy')

sizes = range(100, 500, 50)
for i in sizes:
    # Boundaries
    # Easting
    xmin = 524000 
    xmax = xmin + i
    # Northing
    ymin = 5826000
    ymax = ymin + i

    # Get indices of selected coordinates within the boundaries
    indices = np.nonzero(
        ( x >= xmin ) &
        ( x <= xmax ) &
        ( y >= ymin ) &
        ( y <= ymax )
    )[0]

    # Add found coordinates to the x, y and z arrays
    if indices.size:
        # Saving to file
        np.save(f'selected_area_{i}m.npy', (x[indices], y[indices], z[indices]))
