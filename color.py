# register_colormap.py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_yuanshen_cmap():
    # Define the colors
    colors = [
        (81 / 255, 132 / 255, 178 / 255),  # Color 1
        (170 / 255, 212 / 255, 248 / 255), # Color 2
        (242 / 255, 245 / 255, 250 / 255), # Color 3
        (244 / 255, 167 / 255, 181 / 255), # Color 4
        (204 / 255, 82 / 255, 118 / 255)   # Color 5
    ]

    # Create the custom colormap
    cmap_name = 'yuanshen'
    yuanshen_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    # Register the colormap using the new method
    plt.colormaps.register(name=cmap_name, cmap=yuanshen_cmap)

# Call this function to ensure the colormap is registered
create_yuanshen_cmap()