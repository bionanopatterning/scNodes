from util import *

# # # Annotation to coordinates:
# paths = glob.glob("Z:/mgflast/230812_Esther/ali/bin8_SIRT/*_Ribosome.mrc")
#
# n = 0
# for path in paths:
#     n += get_maxima_3d_watershed(path, threshold=150, min_weight=50000, min_spacing=20, save_txt=True)
#     print(f"Tally: {n}")
#
# print(f"\nFound {n} particles in total, in {len(paths)} volumes.")


# Extracting boxes:

paths = glob.glob("Z:/mgflast/230812_Esther/ali/bin8_SIRT/boxes/*.mrc")

imgs = list()
for p in paths:
    v = p
    c = p.replace("8_SIRT.mrc", "8_SIRT_Ribosome_coords.txt")
    print(c)
    if os.path.exists(c):
        imgs += extract_particles(v, c, 32, unbin=1, two_dimensional=False, normalize=True)

path = "Z:/mgflast/230812_Esther/ali/bin8_SIRT/boxes/"
for i in range(len(imgs)):
    with mrcfile.new(path + f"box_{i}.mrc", overwrite=True) as mrc:
        mrc.set_data(imgs[i])

# # Moving some files on Eta
#src = "Z:\\mgflast\\230808_Fig5"
#
# import os
# import shutil
#
# for i in range(11, 50):
#     #os.rename(src+f"\\tomo_{i}\\tomo_{i}.tlt", src+f"\\tomo_{i}\\tomo_{i}.rawtlt")
#     f = glob.glob(src+f"\\tomo_{i}\\tomo_{i}_rec.mrc")
#     for _f in f:
#         if not os.path.is_file(f"W:/mgflast/14. scSegmentation/Fig5/EMPIAR-10349/tomo_{i}_rec.mrc"):
#             shutil.copy(_f, f"W:/mgflast/14. scSegmentation/Fig5/EMPIAR-10349/tomo_{i}_rec.mrc")
#     # tgt = src+"\\"+name
#     # f = glob.glob(src+"\\"+name+".*")
#     # for _f in f:
#     #     basename = os.path.basename(_f)
#     #     os.rename(_f, src+"\\"+name+"\\"+basename)
