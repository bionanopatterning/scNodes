from util import *

# # Annotation to coordinates:
# paths = glob.glob("U:/mgflast/14. scSegmentation/IgG3NHS_process_all/bin8_SIRT_ali_IgG3NHS-*_C1_complex.mrc")
#
# n = 0
# for path in paths:
# 	n += get_maxima_3d(path, threshold=50, min_weight=7000, min_spacing=10, save_txt=True)
# 	print(f"Tally: {n}")
#
# print(f"\nFound {n} particles in total, in {len(paths)} volumes.")



#
paths = glob.glob("U:/mgflast/14. scSegmentation/IgG3NHS_process_all/bin8_SIRT_ali_IgG3NHS-*_rec.mrc")
imgs = list()
for p in paths:
    v = p
    c = p.replace(".mrc", "_C1_complex_coords.txt")
    if os.path.exists(c):
        imgs += extract_particles(v, c, 64)

path = "U:/mgflast/14. scSegmentation/IgG3NHS_process_all/boxes/"
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
