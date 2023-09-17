from util import *

# Annotation to coordinates:
paths = glob.glob("Z:/mgflast/230905_Relion4_try2/tomograms/*Ribo.mrc")
EXTRACT_COORDINATES = True
EXTRACT_BOXES = False
if EXTRACT_COORDINATES:
    n = 0
    i = 0
    I = len(paths)
    for path in paths:
        i += 1
        n += get_maxima_3d_watershed(path, threshold=200, min_weight=2000, min_spacing=10, save_txt=True)
        print(f"Tally: {n}  (tomo {i}/{I})")

    print(f"\nFound {n} particles in total, in {len(paths)} volumes.")


if EXTRACT_BOXES:
    paths = glob.glob("Z:/mgflast/230812_Esther/ali/bin8_SIRT/*SIRT.mrc")
    imgs = list()
    for p in paths:
        v = p
        c = p.replace("8_SIRT.mrc", "8_SIRT_Ribosome_coords.txt")
        print(c)
        if os.path.exists(c):
            imgs += extract_particles(v, c, 128, normalize=False, two_dimensional=True)

    path = "Z:/mgflast/230812_Esther/ali/bin8_SIRT/boxes/"
    for i in range(len(imgs)):
        print(f"Saving box {i+1}/{len(imgs)}")
        with mrcfile.new(path + f"box_{i}.mrc", overwrite=True) as mrc:
            norm_img = imgs[i] - np.mean(imgs[i])
            norm_img /= np.std(imgs[i])
            mrc.set_data(imgs[i])


