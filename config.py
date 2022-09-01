from dataset import *
import colorcet as cc

current_dataset = Dataset("W:/mgflast/9. Devitrification paper/Figure_4_data/s1t4_centered_stack.tif")

# TODO: only do imgui process inputs when window focused.
# TODO: limit load dataset and reconstruction renderer nodes to 1 instance

## Node editor GUI config

node_editor = None
ne_window_width = 1100
ne_window_height = 700
ne_window_title = "srNodes editor"

## Image Viewer GUI config
image_viewer = None
iv_window_width = 600
iv_window_height = 700
iv_window_title = "srNodes image viewer"
iv_camera_pan_speed = 1.0
iv_camera_zoom_step = 0.1
iv_camera_zoom_max = 40
iv_hist_bins = 50
iv_autocontrast_subsample = 5
iv_autocontrast_saturate = 0.3  # percentage of over/under saturated pixels after performing autocontrast
iv_info_bar_height = 100
iv_frame_select_button_width = 20
iv_frame_select_button_height = 19
iv_frame_select_button_spacing = 4
iv_cm_size = [200, 100] # context menu
iv_contrast_window_size = [200, 240]
iv_clr_frame_background = (0.24, 0.24, 0.24)
iv_clr_main = (0.3 / 1.3, 0.3 / 1.3, 0.3 / 1.3)
iv_clr_theme_active = (0.59, 1.0, 0.86)
iv_clr_theme = (0.729, 1.0, 0.95)
iv_clr_main_bright = (0.3 / 1, 0.3 / 1, 0.3 / 1)
iv_clr_main_dark = (0.3 / 1.5, 0.3 / 1.5, 0.3 / 1.5)
iv_clr_window_background = (0.14, 0.14, 0.14, 0.8)
iv_clear_clr = (0.94, 0.94, 0.94, 1.0)

luts = dict()
luts["Gray"] = cc.linear_grey_0_100_c0
luts["Red"] = cc.linear_ternary_red_0_50_c52
luts["Green"] = cc.linear_kgy_5_95_c69
luts["Blue"] = cc.linear_kbc_5_95_c73
luts["Fire"] = cc.linear_kryw_0_100_c71
luts["Parula"] = cc.linear_bgyw_20_98_c66
luts["Heatmap"] = cc.rainbow_bgyr_10_90_c83
luts["Neon"] = cc.linear_worb_100_25_c53
lut_names = list(luts.keys())



## Overall config
def_img_size = (2048, 2048)
