# general imports
import copy
import matplotlib.pyplot as plt
from advanced_cnn.style_transfer_utilities import ImageHelper, ContentGenerator, StyleGenerator, StyleTransferrer

# get the style and content images
content_image = ImageHelper(img_path="./small_files/The-Grand-Belgrade-Fortress-and-Park-Kalemegdan.jpg")
content_image.permute_channels()
content_image.norm_img(kind="rgb")

style_image = ImageHelper(img_path="./small_files/Van_Gogh_-_Starry_Night.jpg", target_shape=content_image.img.shape)
style_image.permute_channels()
style_image.norm_img(kind="rgb")

# instantiate a generator
generator = StyleTransferrer(
    style_img=style_image.img_transformed,
    content_img=content_image.img_transformed,
    n_conv_layers=9,
    style_weight=0.8
)

generator.fit(n_steps=30)

# get the generated image
ii = copy.deepcopy(generator.w)
ii = ii.reshape(*generator.content_shape)
ii = content_image.permute_channels(ii, kind="BGR2RGB")
ii += content_image.channel_means
ii -= ii.min()
ii /= ii.max()
plt.imshow(ii)
