## Results

Starting from ../voxceleb/v2 recipe.


VoxCeleb is downsampled and subjected to GSM FR codec in the data preparation stage. This means
that in later stages it can be processed in the same way as the CTS data.
On the other hand, it could be argued that, in reality, augmentations are applied before any codec
and the data processing should follow this. Of course this is not possible for the CTS data.

TODO:
 * Change "downsample_audio.sh" and "apply_gsm.sh" to something more generic, e.g., take whole
   sox string and/or to be python based.
 * Add a default for the "--remove_prefix_wav" option that automatically detects the longest common
   suffix of all files and use it as "remove_prefix_wav" to remove.


