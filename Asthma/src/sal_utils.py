import os
import pickle
import numpy as np
import pyvista as pv
import proplot as pplt

def plot_sal(PLOT_DIR, file):
    
    with open(file, 'rb') as handle:
        x = pickle.load(handle)
        s = pickle.load(handle)

    c = np.copy(x)
    c[c == 1] = 0

    w = np.copy(c)
    w[s >= 0.05] = 1.0

    data = pv.wrap(c * 256)
    opacity = [0.0, 0.20, 0.95, 0.9, 0.3]
    plotter = pv.Plotter()
    plotter.add_volume(data, cmap='bone', show_scalar_bar=False, opacity=opacity)
    plotter.camera_position = 'yz'
    plotter.camera.azimuth = -70
    plotter.camera.elevation = 30
    plotter.set_background([1.0, 1.0, 1.0])
    plotter.show()

    a1 = plotter.image[150:700, 260:750, :]


    data = pv.wrap(w * 256)
    opacity = [0.0, 0.25, 0.95, 0.9, 0.3]
    plotter = pv.Plotter()
    plotter.set_background([1.0, 1.0, 1.0])
    plotter.add_volume(data, cmap='coolwarm', show_scalar_bar=False, opacity=opacity)
    plotter.camera_position = 'yz'
    plotter.camera.azimuth = -70
    plotter.camera.elevation = 30
    plotter.show()

    a2 = plotter.image[150:700, 260:750, :]

    data = pv.wrap(w * 256)
    opacity = [0.0, 0.25, 0.95, 0.9, 0.3]
    plotter = pv.Plotter()
    plotter.set_background([1.0, 1.0, 1.0])
    plotter.add_volume(data, cmap='coolwarm', show_scalar_bar=False, opacity=opacity)
    plotter.camera_position = 'yz'
    plotter.camera.azimuth = -90
    plotter.camera.elevation = 90
    plotter.show()

    a3 = plotter.image[150:700, 260:750, :]

    data = pv.wrap(w * 256)
    opacity = [0.0, 0.25, 0.95, 0.9, 0.3]
    plotter = pv.Plotter()
    plotter.set_background([1.0, 1.0, 1.0])
    plotter.add_volume(data, cmap='coolwarm', show_scalar_bar=False, opacity=opacity)
    plotter.camera_position = 'yz'
    plotter.camera.azimuth = 180
    plotter.camera.elevation = 0
    plotter.show()

    a4 = plotter.image[150:700, 260:750, :]

    data = pv.wrap(w[::-1, :, :] * 256)
    opacity = [0.0, 0.25, 0.95, 0.9, 0.3]
    plotter = pv.Plotter()
    plotter.set_background([1.0, 1.0, 1.0])
    plotter.add_volume(data, cmap='coolwarm', show_scalar_bar=False, opacity=opacity)
    plotter.camera_position = 'yz'
    plotter.camera.azimuth = -180
    plotter.camera.elevation = 0
    plotter.show()

    a5 = plotter.image[150:700, 260:750, :]

    data = pv.wrap(w * 256)
    opacity = [0.0, 0.25, 0.95, 0.9, 0.3]
    plotter = pv.Plotter()
    plotter.set_background([1.0, 1.0, 1.0])
    plotter.add_volume(data, cmap='coolwarm', show_scalar_bar=False, opacity=opacity)
    plotter.camera_position = 'yz'
    plotter.camera.azimuth = -90
    plotter.camera.elevation = 0
    plotter.show()

    a6 = plotter.image[150:700, 260:750, :]

    fig, ax = pplt.subplots(nrows=2, ncols=3)

    a = [a1, a2, a3, a4, a5, a6]

    for i in range(6):
        a_temp = np.copy(a[i])
        a_temp[a_temp == 0] = 1
        ax[i].imshow(a[i])

    ax.axis('off')
    ax.format(abc=True)

    name = file.split('/')[-1].split('.pickle')[0]
    
    if not os.path.exists(os.path.join(PLOT_DIR, 'sal_fig/')):
        os.makedirs(os.path.join(PLOT_DIR, 'sal_fig/'))
        
    fig.savefig(os.path.join(PLOT_DIR, f'sal_fig/{name}.png'), dpi=300, transparent=True)
    return a