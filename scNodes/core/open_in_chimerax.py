from chimerax.core.commands import run

paths = ['U:\\mgflast\\14. scSegmentation\\IgG3_reanalyze\\bin8_SIRT_ali_IgG3_040_corrected_rec_Antibody platform.mrc', 'U:\\mgflast\\14. scSegmentation\\IgG3_reanalyze\\bin8_SIRT_ali_IgG3_040_corrected_rec_Carbon.mrc', 'U:\\mgflast\\14. scSegmentation\\IgG3_reanalyze\\bin8_SIRT_ali_IgG3_040_corrected_rec_Membrane.mrc']
level = [63, 82, 95]
colour = [(1.0, 0.40784314274787903, 0.0), (1.0, 0.9529411792755127, 0.0), (0.25882354378700256, 0.8392156958580017, 0.6431372761726379)]
dust = [1.0, 1.0, 1.0]
bgclr = (9.999999974752427e-07, 9.99999883788405e-07, 9.999899930335232e-07)

for i in range(len(paths)):
    run(session, f'open "{paths[i]}"')
    run(session, f'volume #{i+1} level {level[i]}')
    run(session, f'color #{i+1} rgb({colour[i][0]},{colour[i][1]},{colour[i][2]})')
    run(session, f'surface dust #{i+1} size {dust[i]} metric volume')

run(session, f'set bgColor rgb({bgclr[0]},{bgclr[1]},{bgclr[2]})')
run(session, f'graphics silhouettes true')
run(session, f'lighting soft')
run(session, f'lighting shadows false')