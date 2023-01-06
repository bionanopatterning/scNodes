from dataset import *
import colorcet as cc

n_cpus = -1



ne_window_width = 1100
ne_window_height = 700
ne_window_title = "srNodes editor"

iv_window_width = 600
iv_window_height = 700
iv_window_title = "srNodes image viewer"

luts = dict()
luts["Gray"] = cc.linear_grey_0_100_c0
luts["Red"] = cc.linear_ternary_red_0_50_c52
luts["Green"] = cc.linear_kgy_5_95_c69
luts["Blue"] = cc.linear_kbc_5_95_c73
luts["Fire"] = cc.linear_kryw_0_100_c71
luts["Ice"] = cc.linear_blue_95_50_c20
luts["Parula"] = cc.linear_bgyw_20_98_c66
#luts["Forest"] = cc.linear_gow_65_90_c35
luts["Heatmap"] = cc.rainbow_bgyr_10_90_c83
luts["Neon"] = cc.linear_worb_100_25_c53
#luts["Isoluminant hue"] = cc.isoluminant_cgo_80_c38
luts["Over/under A"] = cc.diverging_linear_bjr_30_55_c53
luts["Over/under B"] = cc.diverging_gwr_55_95_c38
#luts["Glasbey"] = cc.glasbey_bw_minc_20_minl_30
lut_names = list(luts.keys())

def_img_size = (2048, 2048)

autocontrast_saturation = 0.3  # Autocontrast will set contrast lims such that this % of pixels is over/under saturated.
autocontrast_subsample = 2  # Autocontrast works on sub-samples images to avoid costly computations. When this value is e.g. 2, every 2nd pixel in X/Y is used.

joblib_mmmode = 'c'