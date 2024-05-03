from util import *

# Annotation to coordinates:
feature_paths = glob.glob("U:/mgflast/14. scSegmentation/Virus_pores/*_Antibody platform.mrc")
tomo_paths = glob.glob("U:/mgflast/14. scSegmentation/Virus_pores/pores/*_rotx.mrc")
coord_paths = glob.glob("U:/mgflast/14. scSegmentation/Virus_pores/pores/*coords.txt")
box_folder = "U:/mgflast/14. scSegmentation/Virus_pores/pores/boxes/"


EXTRACT_COORDINATES = False
EXTRACT_BOXES = True
if EXTRACT_COORDINATES:
    n = 0
    i = 0
    I = len(feature_paths)
    for path in feature_paths:
        i += 1
        n += get_maxima_3d_watershed(path, threshold=50, min_size=400, min_spacing=10.0, save_txt=True)
        print(f"Tally: {n}  (tomo {i}/{I})")

    print(f"\nFound {n} particles in total, in {len(feature_paths)} volumes.")

#bin8_SIRT_ali_IgG3_041_corrected_rec_Antibody platform_coords.txt

if EXTRACT_BOXES:
    imgs = list()
    for p in tomo_paths:
        key = os.path.basename(p).split('_')[0]
        c = None
        for f in coord_paths:
            if key in f:
                c = f
        if c is not None and os.path.exists(c):
            print(p, c)
            imgs += extract_particles(p, c, 64, normalize=True, two_dimensional=False)

    for i in range(len(imgs)):
        print(f"Saving box {i+1}/{len(imgs)}")
        with mrcfile.new(box_folder + f"box_{i}.mrc", overwrite=True) as mrc:
            norm_img = imgs[i] - np.mean(imgs[i])
            norm_img /= np.std(imgs[i])
            mrc.set_data(imgs[i])
