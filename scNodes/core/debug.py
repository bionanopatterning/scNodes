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


