import numpy as np
import matplotlib.pyplot as plt

# Alap paraméterek
image_size = (800, 800)    # csak a bal mellkas lefedéséhez 
dot_radius = 3             # pontok sugara
dot_density = 0.15         # mennyi részt fedjen le a pontok most 15% v

# Egy fehér kép generálása 
speckle_pattern = np.ones(image_size, dtype=np.uint8) * 255  # White background

# A sűrűségere építve kiszámitjuk mennyi pont kell a fehér háttérre 
num_dots = int(dot_density * image_size[0] * image_size[1] / (np.pi * dot_radius ** 2))

# Random helyen legyenek a pontok 
np.random.seed(42)  
x_positions = np.random.randint(0, image_size[1], size=num_dots)
y_positions = np.random.randint(0, image_size[0], size=num_dots)

# Rajzoljuk rá a háttérre 
for x, y in zip(x_positions, y_positions):
    rr, cc = np.ogrid[:image_size[0], :image_size[1]]
    mask = (rr - y) ** 2 + (cc - x) ** 2 <= dot_radius ** 2
    speckle_pattern[mask] = 0  # ráhelyezzük a fekete pontokat a fehér háttérre 

# ábrázolás 
plt.imshow(speckle_pattern, cmap='gray')
plt.axis('off')
plt.savefig("dic_speckle_pattern.png", bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()