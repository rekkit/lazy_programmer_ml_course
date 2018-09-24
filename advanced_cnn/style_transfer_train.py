# general imports
import copy
import matplotlib.pyplot as plt
from advanced_cnn.style_transfer_utilities import ImageHelper, ContentGenerator, StyleGenerator

# instantiate an image helper
img_helper = ImageHelper(img_path="./small_files/The-Grand-Belgrade-Fortress-and-Park-Kalemegdan.jpg")
img_helper.permute_channels()
img_helper.norm_img(kind="rgb")

# instantiate a generator
generator = StyleGenerator(
    target_img=img_helper.img_transformed
)

generator.fit(n_steps=10)

# get the generated image
ii = copy.deepcopy(generator.w)
ii = ii.reshape(*generator.input_img_shape)
ii = img_helper.permute_channels(ii, kind="BGR2RGB")
ii += img_helper.channel_means
ii -= ii.min()
ii /= ii.max()
plt.imshow(ii)
