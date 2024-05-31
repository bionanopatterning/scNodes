import colorcet as cc
import numpy as np

n_cpus = -1

ne_window_title = "scNodes editor"
iv_window_title = "scNodes image viewer"

luts = dict()
luts["Gray"] = cc.linear_grey_0_100_c0
luts["Cerulean"] = np.zeros((256, 3))
luts["Amber"] = np.zeros((256, 3))
luts["Red"] = cc.linear_ternary_red_0_50_c52
luts["Green"] = cc.linear_kgy_5_95_c69
luts["Green"][0] = [0.0, 0.0, 0.0]
luts["Blue"] = cc.linear_kbc_5_95_c73
luts["Blue"][0] = [0.0, 0.0, 0.0]
luts["Fire"] = cc.linear_kryw_0_100_c71
luts["Ice"] = cc.linear_blue_95_50_c20
luts["Parula"] = cc.linear_bgyw_20_98_c66
#luts["Forest"] = cc.linear_gow_65_90_c35
luts["Heatmap"] = cc.rainbow_bgyr_10_90_c83
luts["Neon"] = cc.linear_worb_100_25_c53
#luts["Isoluminant hue"] = cc.isoluminant_cgo_80_c38
luts["Over/under A"] = cc.diverging_linear_bjr_30_55_c53
luts["Over/under B"] = cc.diverging_gwr_55_95_c38

cerulean = np.array([63.0 / 255, 222.0 / 255, 255.0 / 255])
amber = np.array([255.0 / 255, 185.0 / 255, 22.0 / 255])
for i in range(256):
    luts["Amber"][i, :] = (i / 255) * amber
    luts["Cerulean"][i, :] = (i / 255) * cerulean
luts["Glasbey"] = cc.glasbey_bw_minc_20_minl_30
lut_names = list(luts.keys())

def_img_size = (2048, 2048)

autocontrast_saturation = 0.03  # Autocontrast will set contrast lims such that this % of pixels is over/under saturated.
autocontrast_subsample = 2  # Autocontrast works on sub-sampled images to avoid costly computations. When this value is e.g. 2, every 2nd pixel in X/Y is used.
joblib_mmmode = 'c'