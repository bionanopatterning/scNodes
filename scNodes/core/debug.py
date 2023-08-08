from util import *

# Annotation to coordinates:
paths = glob.glob("U:/mgflast/14. scSegmentation/Fig4/bin8_SIRT_ali_IgG3_049_corrected_rec_Antibody_platforms.mrc")
EXTRACT_COORDINATES = False
EXTRACT_BOXES = True
if EXTRACT_COORDINATES:
    n = 0
    i = 0
    I = len(paths)
    for path in paths:
        i += 1
        n += get_maxima_3d(path, threshold=40, min_volume=1000, min_spacing=20, save_txt=True)
        print(f"Tally: {n}  (tomo {i}/{I})")

    print(f"\nFound {n} particles in total, in {len(paths)} volumes.")


if EXTRACT_BOXES:
    paths = glob.glob("U:/mgflast/14. scSegmentation/Fig4/bin8_SIRT_ali_IgG3_049_corrected_rec.mrc")
    imgs = list()
    for p in paths:
        v = p
        c = p.replace(".mrc", "_Antibody_platforms_coords.txt")
        if os.path.exists(c):
            imgs += extract_particles(v, c, 64, two_dimensional=True)

    path = "U:/mgflast/14. scSegmentation/Fig4/boxes_abs/"
    for i in range(len(imgs)):
        print(f"Saving box {i+1}/{len(imgs)}")
        with mrcfile.new(path + f"box_{i}.mrc", overwrite=True) as mrc:
            mrc.set_data(imgs[i])


