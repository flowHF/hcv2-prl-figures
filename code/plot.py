import ROOT
import array
import os
import numpy as np
import subprocess
from ROOT import TFile, TLegend
import sys
sys.path.append('./')
from plot_untils import GetLegend, GetCanvas4sub, SetGlobalStyle, SetObjectStyle, GetInvMassHistAndFit, GetV2HistAndFit
from plot_untils import get_edges_from_hist, preprocess_ncq, read_txt, DrawLineAt0, rebin_safely, preprocess, preprocess_data, read_hists, get_band, get_latex, merge_asymmetric_errors, fill_graph, model_chi2, graph_to_hist_with_errors, get_interp_hist
from plot_untils import scale_x_errors, compute_ratio_graph, pdf2eps_imagemagick
SetGlobalStyle(padleftmargin=0.15, padrightmargin=0.08, padbottommargin=0.16, padtopmargin=0.075,
               titleoffset=1.3, palette=ROOT.kRainBow, titlesize=0.06, labelsize=0.055, maxdigits=4)


def compare_allD(color_lc=ROOT.kRed+2, color_d0=ROOT.kBlue+2, no_J_Psi=False, no_Tamu=False):
    '''Compare Lc v2 with D0, Ds, Dplus, J/psi and pi'''
    SetGlobalStyle(padleftmargin=0.12, padrightmargin=0.03, padbottommargin=0.12, padtopmargin=0.03,
               titleoffset=1.1, palette=ROOT.kRainBow, titlesize=0.05, labelsize=0.035, maxdigits=4)
    data_path = '../input-data'
    colors_allD = {'ds':ROOT.kSpring+4, 'dplus':ROOT.kCyan+2, 'jpsi':ROOT.kViolet+1}
    markerstyles_allD = {'ds':ROOT.kFullCross, 'dplus':ROOT.kFullDiamond, 'jpsi':ROOT.kFullCrossX}
    Ds_path = f'{data_path}/lc-d0-data/v2_prompt_wsyst_Ds_3050.root'
    Dplus_path = f'{data_path}/lc-d0-data/v2_prompt_wsyst_Dplus_3050.root'
    J_psi_path = f'{data_path}/lc-d0-data/HEPData-ins1798507-v1-dataFig2C.csv'

    ds_hists = read_hists(Ds_path, markerstyle=markerstyles_allD["ds"], markersize=3, colors=colors_allD["ds"], gname=['gvn_prompt_stat', 'tot_syst'])
    Dplus_hists = read_hists(Dplus_path, markerstyle=markerstyles_allD["dplus"], markersize=3, colors=colors_allD["dplus"], gname=['gvn_prompt_stat', 'tot_syst'])
    scale_x_errors(Dplus_hists[1], target_graph=Dplus_hists[0], scale_factor=0.6)
    scale_x_errors(ds_hists[1], target_graph=ds_hists[0], scale_factor=0.6)
    df_J_psi = preprocess(J_psi_path, sep=",", header=12, do_ncq=True)
    stat_J_psi = fill_graph(df_J_psi, columns=['PT [GEV/C]', 'V2', 'stat +'])
    syst_J_psi = fill_graph(df_J_psi, columns=['PT [GEV/C]', 'V2', 'syst +'])
    J_psi_hists = [stat_J_psi, syst_J_psi]
    SetObjectStyle(stat_J_psi, markerstyle=markerstyles_allD["jpsi"], color=colors_allD["jpsi"], linewidth=2, markersize=3)
    SetObjectStyle(syst_J_psi, color=colors_allD["jpsi"], linewidth=2, fillstyle=0)
    syst_J_psi.SetFillStyle(0)

    data_pi_30_40 = f'{data_path}/light-flavor-data/HEPData-ins2093750-v1-csv/Table20.csv'
    data_pi_40_50 = f'{data_path}/light-flavor-data/HEPData-ins2093750-v1-csv/Table29.csv'
    df_pi_30_50 = preprocess_data([data_pi_30_40, data_pi_40_50])
    data_pi_30_40_sp = f'{data_path}/light-flavor-data/HEPData-ins1672822-v1-csv-SP/Table5.csv'
    data_pi_40_50_sp = f'{data_path}/light-flavor-data/HEPData-ins1672822-v1-csv-SP/Table6.csv'
    df_pi_30_50_sp = preprocess_data([data_pi_30_40_sp, data_pi_40_50_sp], header=12)
    graph_pi_30_50 = fill_graph(df_pi_30_50)
    graph_pi_30_50_sp = fill_graph(df_pi_30_50_sp)
    df_pi_30_40_sp  = preprocess_data([data_pi_30_40_sp], header=12, get_source_data=True)
    graph_pi_30_40_stat = fill_graph(df_pi_30_40_sp, columns=["PT [GEV]", "CH_PIONS_v2_3040", "stat +"])
    graph_pi_30_40_syst = fill_graph(df_pi_30_40_sp, columns=["PT [GEV]", "CH_PIONS_v2_3040", "sys +"])
    target_bins=list(df_pi_30_40_sp['PT [GEV] LOW'])
    target_bins.append(df_pi_30_40_sp['PT [GEV] HIGH'].values[-1])
    # print('target_bins', target_bins)
    scale_x_errors(graph_pi_30_40_syst, scale_factor=0.6, target_bins=target_bins)
    SetObjectStyle(graph_pi_30_50_sp, color=ROOT.kBlack, markerstyle=ROOT.kFullDoubleDiamond, linewidth=2, markersize=3)
    SetObjectStyle(graph_pi_30_40_stat, color=ROOT.kBlack, markerstyle=ROOT.kFullDoubleDiamond, linewidth=2, markersize=3)
    SetObjectStyle(graph_pi_30_40_syst, color=ROOT.kBlack, linewidth=2, fillstyle=0)

    models_path = '../input-models/arxivv1905.09216-tamu'
    tamu_lc_high=f'{models_path}/lc-up.dat'
    tamu_lc_low=f'{models_path}/lc-low.dat'
    tamu_d0_high=f'{models_path}/d0-up.dat'
    tamu_d0_low=f'{models_path}/d0-low.dat'
    tamu_Ds_path = f'{models_path}/PromptDs_TAMU_v2_5TeV_3050.txt'
    tamu_J_psi_path = f'{models_path}/tamu_jpsi_3050.txt'

    tamu_lc_high_x, tamu_lc_high_y = preprocess(tamu_lc_high, sep=",")
    tamu_lc_low_x, tamu_lc_low_y = preprocess(tamu_lc_low, sep=",")
    tamu_d0_high_x, tamu_d0_high_y = preprocess(tamu_d0_high, sep=",")
    tamu_d0_low_x, tamu_d0_low_y = preprocess(tamu_d0_low, sep=",")
    tamu_J_psi_x, tamu_J_psi_y = preprocess(tamu_J_psi_path, sep=" ", header=0)
    tamu_ds_x, tamu_ds_low_y, tamu_ds_high_y = preprocess(tamu_Ds_path, catania=True, sep=" ", header=0)

    polyline_lc_tamu = get_band(tamu_lc_low_x, tamu_lc_high_x, tamu_lc_low_y, tamu_lc_high_y, color_lc)
    polyline_d0_tamu = get_band(tamu_d0_low_x, tamu_d0_high_x, tamu_d0_low_y, tamu_d0_high_y, color_d0)
    polyline_ds_tamu = get_band(tamu_ds_x, tamu_ds_x, tamu_ds_low_y, tamu_ds_high_y, colors_allD["ds"])
    graph_tamu_J_psi = ROOT.TGraph(len(tamu_J_psi_x), array.array('d', tamu_J_psi_x), array.array('d', tamu_J_psi_y))
    polyline_lc_tamu.SetFillColorAlpha(colors_lc["tamu"], 0.5)
    polyline_d0_tamu.SetFillColorAlpha(colors_d0["tamu"], 0.4)
    polyline_ds_tamu.SetFillColorAlpha(colors_allD["ds"], 0.5)
    SetObjectStyle(graph_tamu_J_psi, color=colors_allD["jpsi"], linewidth=2)

    canvas = ROOT.TCanvas("LcVsAllD_wTamu", "LcVsAllD_wTamu", 1200, 1000)
    canvas.SetLogx()
    frame = canvas.DrawFrame(0.499, -0.01, 27.7, 0.4,
                            ';#it{p}_{T} (GeV/#it{c});#it{v}_{2}')
    frame.GetYaxis().SetDecimals()
    frame.GetXaxis().SetMoreLogLabels()
        
    if not no_Tamu:
        polyline_lc_tamu.Draw('F')
        polyline_d0_tamu.Draw('F')
        polyline_ds_tamu.Draw('F')
        if not no_J_Psi:
            graph_tamu_J_psi.Draw('same l')    
    lc_hists[0].Draw('same epz')
    lc_hists[1].Draw('same e2z')
    if not no_J_Psi:
        J_psi_hists[0].Draw('same epz')
        J_psi_hists[1].Draw('same e2z')
    ds_hists[0].Draw('same epz')
    ds_hists[1].Draw('same e2z')
    Dplus_hists[0].Draw('same epz')
    Dplus_hists[1].Draw('same e2z')
    d0_hists[0].Draw('same epz')
    d0_hists[1].Draw('same e2z')
    # graph_pi_30_50_sp.Draw('same epz')
    graph_pi_30_40_stat.Draw('same epz')
    graph_pi_30_40_syst.Draw('same e2z')

    lat_large, lat_mid, latLabel = get_latex()
    latLabel.SetTextSize(0.03)
    legend_TextSize = latLabel.GetTextSize()
    position_x = 0.15
    position_y = 0.62
    position_xtop = 0.5
    position_ytop = 0.89

    legend = TLegend(position_x, position_y, position_xtop, position_ytop-0.06)
    legend.SetTextFont(42)
    legend.SetTextSize(legend_TextSize)
    legend.SetNColumns(2)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
        
    legend.AddEntry(lc_hists[0],  "Prompt #Lambda^{+}_{c}  ", "p")
    legend.AddEntry(Dplus_hists[0], "Prompt D^{+}           ", "p")
    legend.AddEntry(d0_hists[0],   "Prompt D^{0}           ", "p")
    legend.AddEntry(ds_hists[0],   "Prompt D^{+}_{s}       ", "p")
    legend.AddEntry("", "", "") 
    legend.AddEntry("", "", "") 
    
    # legend.AddEntry(graph_pi_30_50_sp, "#pi^{#pm}  JHEP 09 (2018) 006", "p")
    legend.AddEntry(graph_pi_30_40_stat, "#pi^{#pm}  JHEP 09 (2018) 006", "p")
    # legend.AddEntry(graph_pi_30_50, "#pi #pm  JHEP 05 (2023) 243", "p")
    legend.Draw("same")
    legend_r = TLegend(position_x+0.415, position_y+0.01, position_xtop+0.4, position_ytop-0.06)
    legend_r.SetTextFont(42)
    legend_r.SetTextSize(legend_TextSize)
    legend_r.SetBorderSize(0)
    legend_r.SetFillStyle(0)
    if not no_J_Psi:
            legend_r.AddEntry(J_psi_hists[0], "inclusive J/#psi JHEP 10 (2020) 141", "p")
            legend_r.AddEntry("", "", "")
            legend_r.AddEntry("", "", "") 
            legend_r.AddEntry("", "", "") 
            legend_r.SetNColumns(2)
    if not no_Tamu:
        legend_r.AddEntry(polyline_lc_tamu, "  #Lambda^{+}_{c}      ", "f")
        legend_r.AddEntry(polyline_d0_tamu, "  D^{0}        ", "f")
        legend_r.AddEntry(polyline_ds_tamu, "  D^{+}_{s}        ", "f")
        if not no_J_Psi:
            legend_r.AddEntry(graph_tamu_J_psi, "  J/#psi        ", "l")
    else:
        legend_r.AddEntry("", "", "")
        legend_r.AddEntry("", "", "")
        legend_r.AddEntry("", "", "")
        legend_r.AddEntry("", "", "")
    legend_r.Draw("same")
    lat_large.DrawLatex(position_x, position_ytop, 'ALICE  Pb#font[122]{-}Pb 30#font[122]{-}50% ')
    latLabel.DrawLatex(position_x, position_ytop-0.05, "#it{v}_{2} {SP, |#Delta#it{#eta}| > 1.3},  #sqrt{#it{s}_{NN}} = 5.36 TeV")  # , 2.5 < y < 4
    if not no_J_Psi:
        latLabel.DrawLatex(position_x+0.415, position_ytop-0.05, 
                        "#it{v}_{2} {SP, |#Delta#it{#eta}| > 1.1},  #sqrt{#it{s}_{NN}} = 5.02 TeV")  # , |y| < 0.8"
    latLabel.DrawLatex(position_x, position_ytop-0.2, "#it{v}_{2} {2, |#Delta#it{#eta}| > 2},  #sqrt{#it{s}_{NN}} = 5.02 TeV, 30#font[122]{-}40%")
    # latLabel.DrawLatex(position_x+0.02, 0.66, "#it{v}_{2} {2, |#Delta#it{#eta}| > 0.8},  #sqrt{#it{s}_{NN}} = 5.02 TeV")
    if not no_Tamu:
        latLabel.DrawLatex(position_x+0.415, position_ytop-0.145 if not no_J_Psi else position_ytop-0.05, "TAMU,  #sqrt{#it{s}_{NN}} = 5.02 TeV")

    # lat_mid.DrawLatex(position_x, 0.81, 'Pb#font[122]{-}Pb, 30#font[122]{-}50% ')
    canvas.Update()
    canvas.Draw()
    out_name = f'{outDir}/LcVsAllD_wTamu.pdf'
    if no_Tamu:
        out_name = f'{outDir}/LcVsAllD_woTamu.pdf'
    if no_J_Psi:
        out_name = out_name.replace('.pdf', '_woJpsi.pdf')
    canvas.SaveAs(out_name)
    if debug:
        from IPython.display import display, Image
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png") as tmpfile:
            canvas.SaveAs(tmpfile.name)
            display(Image(filename=tmpfile.name))   


def compare_with_model(color_lc=ROOT.kRed+2, color_d0=ROOT.kBlue+2, do_ncq=False, split=False, logx=False):
    colors_lc=colors_d0
    '''Compare Lc and D0 v2 with transport model calculations'''
    d0_hists[0].SetMarkerSize(1)
    lc_hists[0].SetMarkerSize(1)
    data_path = '../input-models'
    tamu_lc_high=f'{data_path}/arxivv1905.09216-tamu/lc-up.dat'
    tamu_lc_low=f'{data_path}/arxivv1905.09216-tamu/lc-low.dat'
    tamu_d0_high=f'{data_path}/arxivv1905.09216-tamu/d0-up.dat'
    tamu_d0_low=f'{data_path}/arxivv1905.09216-tamu/d0-low.dat'
    lc_catania = f'{data_path}/Fwd_ Predictions for LambdaC elliptic flow/v2_Lc_502_3050_Catania_band.dat'
    d0_catania = f'{data_path}/Fwd_ Predictions for LambdaC elliptic flow/v2_D0_502_3050_Catania_band.dat'
    # lc_Langevin_fnwsnloaver = f'{data_path}/Langevin-results-to pxy25.2.13/Lambda_c-v2-pbpb5.02-30-50-data/Lcv2fnwsnloaver30-50.dat'
    # d0_Langevin = f'{data_path}/FinalMarcosAndFiles/InputFile/theory-driven/CCNU_1/D0-v2-pbpb5020-30-50.dat'
    # lc_Langevin_fnwsnloaver = f'{data_path}/langevin-d4-results-to-pxy_25.3.21/Lcv2fnwsnlo30-50.dat'
    # d0_Langevin = f'{data_path}/langevin-d4-results-to-pxy_25.3.21/Dv2fnwsnlo30-50.dat'
    lc_Langevin_fnwsnloaver = f'{data_path}/langevin-d4-results-to-pxy25.3.22/Lcv2fnwsnlo30-50.dat'
    d0_Langevin = f'{data_path}/langevin-d4-results-to-pxy25.3.22/D0v2fnwsnlo30-50.dat'
    powlang = f'{data_path}/POWLANG-v2-PbPb3050.txt'
    lbt_d0 = f'{data_path}/v2_PbPb5360-LBT-PNP/v2_D_30-50.dat'
    lbt_lc = f'{data_path}/v2_PbPb5360-LBT-PNP/v2_lambdac_60-80.dat'
    epos4hq_d0 = f'{data_path}/epos4hq/v2pt_D0_PbPb5.02TeV_30-50.dat'
    epos4hq_lc = f'{data_path}/epos4hq/v2pt_Lambdac_PbPb5.02TeV_30-50.dat'
    # all = pd.read_csv(powlang, sep=" ", header=None, nrows=None)
    # global HTL_D0, HTL_L_c, latQCD_D0, latQCD_L_c
    HTL_D0 = read_txt(powlang, header=1, sep=' ', nrows=18)
    HTL_L_c = read_txt(powlang, header=61, sep=' ', nrows=18)
    latQCD_D0 = read_txt(powlang, header=81, sep=' ', nrows=18)
    latQCD_L_c = read_txt(powlang, header=141, sep=' ', nrows=18)
    d0_lbt = read_txt(lbt_d0, sep='\t')
    lc_lbt = read_txt(lbt_lc, sep='\t')
    d0_epos4hq = read_txt(epos4hq_d0, sep=' ', header=0)
    lc_epos4hq = read_txt(epos4hq_lc, sep=' ', header=0)

    if do_ncq:
        lc_df_tamu_low = preprocess(tamu_lc_low, sep=",", do_ncq=True)
        lc_df_tamu_high = preprocess(tamu_lc_high, sep=",", do_ncq=True)
        d0_df_tamu_low = preprocess(tamu_d0_low, sep=",", do_ncq=True)
        d0_df_tamu_high = preprocess(tamu_d0_high, sep=",", do_ncq=True)

        lc_df_catania = preprocess(lc_catania, sep=" ", do_ncq=True)
        d0_df_catania = preprocess(d0_catania, sep=" ", do_ncq=True)

        lc_df_Langevin_fnwsnloaver = preprocess(lc_Langevin_fnwsnloaver, sep="        ", do_ncq=True)
        d0_df_Langevin = preprocess(d0_Langevin, sep="        ", do_ncq=True)
        all_data = {"tamu": {"lc":[lc_df_tamu_low, lc_df_tamu_high],
                            "d0":[d0_df_tamu_low, d0_df_tamu_high]},
                    "catania": {"lc":[lc_df_catania],
                            "d0":[d0_df_catania]},
                    "langevin": {"lc":[lc_df_Langevin_fnwsnloaver],
                                 "d0":[d0_df_Langevin],},
                    "htl": {"lc":[HTL_L_c],
                         "d0":[HTL_D0]},
                    "latQCD": {"lc":[latQCD_L_c], 
                         "d0":[latQCD_D0]},
                    "lbt": {"lc":[lc_lbt], 
                         "d0":[d0_lbt]},
                    "epos4hq": {"lc":[lc_epos4hq],
                         "d0":[d0_epos4hq]}
                    }
        return all_data
    do_interp = True
    if do_interp:
        pt_bins = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 16, 24]
        d0_hists_origin = graph_to_hist_with_errors(d0_hists[0], 'd0', pt_bins, title="")
        d0_hists_target = d0_hists_origin.Clone("d0_hists_target")
        d0_hists_target.Reset()
        pt_bins = [2, 3, 4, 5, 6, 8, 12, 24]
        lc_hists_origin = graph_to_hist_with_errors(lc_hists[0], 'lc', pt_bins, title="")
        lc_hists_target = lc_hists_origin.Clone("lc_hists_target")
        lc_hists_target.Reset()
        
        x_tamu_lc, lc_df_tamu_low_interp = preprocess(tamu_lc_low, sep=",", do_interp=True)
        x_tamu_lc, lc_df_tamu_high_interp = preprocess(tamu_lc_high, sep=",", do_interp=True)
        x_tamu_d0, d0_df_tamu_low_interp = preprocess(tamu_d0_low, sep=",", do_interp=True)
        x_tamu_d0, d0_df_tamu_high_interp = preprocess(tamu_d0_high, sep=",", do_interp=True)
        x_catania_lc, lc_df_catania_interp, lc_df_catania_interp_high = preprocess(lc_catania, sep=" ", do_interp=True, catania=True,)
        x_catania_d0, d0_df_catania_interp, d0_df_catania_interp_high = preprocess(d0_catania, sep=" ", do_interp=True, catania=True,)
        x_Langevin_lc, lc_df_Langevin_fnwsnloaver_interp = preprocess(lc_Langevin_fnwsnloaver, sep="        ", do_interp=True)
        x_Langevin_d0, d0_df_Langevin_interp = preprocess(d0_Langevin, sep="        ", do_interp=True)
        x_htl_d0, HTL_D0_interp = preprocess(powlang, header=1, sep=' ', nrows=18, do_interp=True)
        x_htl_lc, HTL_L_c_interp = preprocess(powlang, header=61, sep=' ', nrows=18, do_interp=True)
        x_lat_d0, latQCD_D0_interp = preprocess(powlang, header=81, sep=' ', nrows=18, do_interp=True)
        x_lat_lc, latQCD_L_c_interp = preprocess(powlang, header=141, sep=' ', nrows=18, do_interp=True)
        x_lbt_d0, d0_lbt_interp = preprocess(lbt_d0, sep='\t', do_interp=True)
        x_lbt_lc, lc_lbt_interp = preprocess(lbt_lc, sep='\t', do_interp=True)
        x_epos4hq_d0, d0_epos4hq_interp = preprocess(epos4hq_d0, sep=' ', header=0, do_interp=True)
        x_epos4hq_lc, lc_epos4hq_interp = preprocess(epos4hq_lc, sep=' ', header=0, do_interp=True)

        hist_tamu_lc = get_interp_hist(lc_hists_target, x_tamu_lc, [lc_df_tamu_low_interp, lc_df_tamu_high_interp], name='tamu_lc')
        hist_tamu_d0 = get_interp_hist(d0_hists_target, x_tamu_d0, [d0_df_tamu_low_interp, d0_df_tamu_high_interp], name='tamu_d0')
        hist_catania_d0 = get_interp_hist(d0_hists_target, x_catania_d0, [d0_df_catania_interp, d0_df_catania_interp_high], name='catania_d0')
        hist_catania_lc = get_interp_hist(lc_hists_target, x_catania_lc, [lc_df_catania_interp, lc_df_catania_interp_high], name='catania_lc')
        hist_langevin_lc = get_interp_hist(lc_hists_target, x_Langevin_lc, [lc_df_Langevin_fnwsnloaver_interp], name='langevin_lc')
        hist_langevin_d0 = get_interp_hist(d0_hists_target, x_Langevin_d0, [d0_df_Langevin_interp], name='langevin_d0')
        hist_htl_lc = get_interp_hist(lc_hists_target, x_htl_lc, [HTL_L_c_interp], name='htl_lc')
        hist_htl_d0 = get_interp_hist(d0_hists_target, x_htl_d0, [HTL_D0_interp], name='htl_d0')
        hist_lat_lc = get_interp_hist(lc_hists_target, x_lat_lc, [latQCD_L_c_interp], name='latQCD_lc')
        hist_lat_d0 = get_interp_hist(d0_hists_target, x_lat_d0, [latQCD_D0_interp], name='latQCD_d0')
        hist_lbt_lc = get_interp_hist(lc_hists_target, x_lbt_lc, [lc_lbt_interp], name='lbt_lc')
        hist_lbt_d0 = get_interp_hist(d0_hists_target, x_lbt_d0, [d0_lbt_interp], name='lbt_d0')
        hist_epos4hq_lc = get_interp_hist(lc_hists_target, x_epos4hq_lc, [lc_epos4hq_interp], name='epos4hq_lc')
        hist_epos4hq_d0 = get_interp_hist(d0_hists_target, x_epos4hq_d0, [d0_epos4hq_interp], name='epos4hq_d0')
        SetObjectStyle(hist_tamu_lc, color=colors_lc["tamu"])
        SetObjectStyle(hist_tamu_d0, color=colors_d0["tamu"])
        SetObjectStyle(hist_catania_lc, color=colors_lc["catania"])
        SetObjectStyle(hist_catania_d0, color=colors_d0["catania"])
        SetObjectStyle(hist_langevin_lc, color=colors_lc["langevin"])
        SetObjectStyle(hist_langevin_d0, color=colors_d0["langevin"])
        SetObjectStyle(hist_htl_lc, color=colors_lc["htl"])
        SetObjectStyle(hist_htl_d0, color=colors_d0["htl"])
        SetObjectStyle(hist_lat_lc, color=colors_lc["latQCD"])
        SetObjectStyle(hist_lat_d0, color=colors_d0["latQCD"])
        SetObjectStyle(hist_lbt_lc, color=colors_lc["lbt"])
        SetObjectStyle(hist_lbt_d0, color=colors_d0["lbt"])
        SetObjectStyle(hist_epos4hq_lc, color=colors_lc["epos4hq"])
        SetObjectStyle(hist_epos4hq_d0, color=colors_d0["epos4hq"])
        d0_tot = merge_asymmetric_errors(d0_hists[0], d0_hists[1])
        lc_tot = merge_asymmetric_errors(lc_hists[0], lc_hists[1])
        lc_hists_interp = [hist_tamu_lc, hist_catania_lc, hist_langevin_lc, hist_htl_lc, hist_lat_lc, hist_lbt_lc, hist_epos4hq_lc]
        d0_hists_interp = [hist_tamu_d0, hist_catania_d0, hist_langevin_d0, hist_htl_d0, hist_lat_d0, hist_lbt_d0, hist_epos4hq_d0]
        d0_hists_interp_rebin = [rebin_safely(hist, '', new_bin_edges=get_edges_from_hist(d0_hists_target), fixed_rebin=1)
                                 for hist in d0_hists_interp]
        lc_hists_interp_rebin = [rebin_safely(hist, '', new_bin_edges=get_edges_from_hist(lc_hists_target), fixed_rebin=1)
                                 for hist in lc_hists_interp]
        d0_hists_interp = [hist_tamu_d0, hist_catania_d0, hist_langevin_d0, hist_htl_d0, hist_lat_d0, hist_lbt_d0, hist_epos4hq_d0]
        for hist in d0_hists_interp_rebin:
            model_chi2(d0_tot, hist, ndf=12)
        for hist in lc_hists_interp_rebin:
            model_chi2(lc_tot, hist, ndf=5)
    tamu_lc_high_x, tamu_lc_high_y = preprocess(tamu_lc_high, sep=",")
    tamu_lc_low_x, tamu_lc_low_y = preprocess(tamu_lc_low, sep=",")
    tamu_d0_high_x, tamu_d0_high_y = preprocess(tamu_d0_high, sep=",")
    tamu_d0_low_x, tamu_d0_low_y = preprocess(tamu_d0_low, sep=",")
    d0_catania_x, d0_catania_low_y, d0_catania_high_y = preprocess(d0_catania, catania=True, sep=" ")
    lc_catania_x, lc_catania_low_y, lc_catania_high_y = preprocess(lc_catania, catania=True, sep=" ")
    lc_Langevin_fnwsnloaver_x, lc_Langevin_fnwsnloaver_y = preprocess(lc_Langevin_fnwsnloaver, sep="        ")
    # d0_Langevin_x, d0_Langevin_y = preprocess(d0_Langevin, sep="  ")
    d0_Langevin_x, d0_Langevin_y = preprocess(d0_Langevin, sep='        ')
    polyline_lc_tamu = get_band(tamu_lc_low_x, tamu_lc_high_x, tamu_lc_low_y, tamu_lc_high_y, color_lc)
    polyline_d0_tamu = get_band(tamu_d0_low_x, tamu_d0_high_x, tamu_d0_low_y, tamu_d0_high_y, color_d0)
    polyline_lc_tamu.SetFillColorAlpha(colors_lc["tamu"], 0.5)
    polyline_d0_tamu.SetFillColorAlpha(colors_d0["tamu"], 0.4)
    # polyline_d0_tamu.SetFillStyle(3002) 
    # polyline_lc_tamu.SetFillStyle(3002) 
    polyline_d0_catania = get_band(d0_catania_x, d0_catania_x, d0_catania_low_y, d0_catania_high_y, color_d0)
    polyline_lc_catania = get_band(lc_catania_x, lc_catania_x, lc_catania_low_y, lc_catania_high_y, color_lc)
    polyline_lc_catania.SetFillColorAlpha(colors_lc["catania"], 0.3)
    polyline_d0_catania.SetFillColorAlpha(colors_d0["catania"], 0.8)
    
    graph_lc_Langevin_fnwsnloaver = ROOT.TGraph(len(lc_Langevin_fnwsnloaver_x), array.array('d', lc_Langevin_fnwsnloaver_x), array.array('d', lc_Langevin_fnwsnloaver_y))
    graph_lc_Langevin_fnwsnloaver.SetLineColor(colors_lc["langevin"])
    graph_lc_Langevin_fnwsnloaver.SetLineWidth(2)
    graph_lc_Langevin_fnwsnloaver.SetLineStyle(2)
    graph_d0_Langevin_fnwsnloaver = ROOT.TGraph(len(d0_Langevin_x), array.array('d', d0_Langevin_x), array.array('d', d0_Langevin_y))
    graph_d0_Langevin_fnwsnloaver.SetLineColor(colors_d0["langevin"])
    graph_d0_Langevin_fnwsnloaver.SetLineWidth(2)
    graph_d0_Langevin_fnwsnloaver.SetLineStyle(2)

    graph_lc_htl = ROOT.TGraph(len(HTL_L_c.iloc[:, 0]), array.array('d', HTL_L_c.iloc[:, 0]), array.array('d', HTL_L_c.iloc[:, 1]))
    graph_lc_htl.SetLineColor(colors_lc["htl"])
    graph_lc_htl.SetLineWidth(2)
    graph_lc_htl.SetLineStyle(6)
    graph_d0_htl = ROOT.TGraph(len(HTL_D0.iloc[:, 0]), array.array('d', HTL_D0.iloc[:, 0]), array.array('d', HTL_D0.iloc[:, 1]))
    graph_d0_htl.SetLineColor(colors_d0["htl"])
    graph_d0_htl.SetLineWidth(2)
    graph_d0_htl.SetLineStyle(6)

    graph_lc_lat = ROOT.TGraph(len(latQCD_L_c.iloc[:, 0]), array.array('d', latQCD_L_c.iloc[:, 0]), array.array('d', latQCD_L_c.iloc[:, 1]))
    graph_lc_lat.SetLineColor(colors_lc["latQCD"])
    graph_lc_lat.SetLineWidth(2)
    graph_lc_lat.SetLineStyle(9)
    graph_d0_lat = ROOT.TGraph(len(latQCD_D0.iloc[:, 0]), array.array('d', latQCD_D0.iloc[:, 0]), array.array('d', latQCD_D0.iloc[:, 1]))
    graph_d0_lat.SetLineColor(colors_d0["latQCD"])
    graph_d0_lat.SetLineWidth(2)
    graph_d0_lat.SetLineStyle(9)

    graph_lc_lbt = ROOT.TGraph(len(lc_lbt.iloc[:, 0]), array.array('d', lc_lbt.iloc[:, 0]), array.array('d', lc_lbt.iloc[:, 1]))
    graph_lc_lbt.SetLineColor(colors_lc["lbt"])
    graph_lc_lbt.SetLineWidth(2)
    graph_lc_lbt.SetLineStyle(7)
    graph_d0_lbt = ROOT.TGraph(len(d0_lbt.iloc[:, 0]), array.array('d', d0_lbt.iloc[:, 0]), array.array('d', d0_lbt.iloc[:, 1]))
    graph_d0_lbt.SetLineColor(colors_d0["lbt"])
    graph_d0_lbt.SetLineWidth(2)
    graph_d0_lbt.SetLineStyle(7)

    graph_lc_epos4hq = ROOT.TGraph(len(lc_epos4hq.iloc[:, 0]), array.array('d', lc_epos4hq.iloc[:, 0]), array.array('d', lc_epos4hq.iloc[:, 1]))
    graph_lc_epos4hq.SetLineColor(colors_lc["epos4hq"])
    graph_lc_epos4hq.SetLineWidth(3)
    graph_lc_epos4hq.SetLineStyle(3)
    graph_d0_epos4hq = ROOT.TGraph(len(d0_epos4hq.iloc[:, 0]), array.array('d', d0_epos4hq.iloc[:, 0]), array.array('d', d0_epos4hq.iloc[:, 1]))
    graph_d0_epos4hq.SetLineColor(colors_d0["epos4hq"])
    graph_d0_epos4hq.SetLineWidth(3)
    graph_d0_epos4hq.SetLineStyle(3)
    # print(lc_Langevin_fnwsnloaver_x)
    # graph_lc_Langevin_naver = ROOT.TGraph(len(lc_Langevin_naver_x), array.array('d', lc_Langevin_naver_x), array.array('d', lc_Langevin_naver_y))
    # graph_lc_Langevin_naver.SetLineColor(ROOT.kBlack)
    # graph_lc_Langevin_naver.SetLineWidth(2)
    
    lat_large, lat_mid, latLabel = get_latex()
    latLabel.SetTextSize(0.035)
    legend_TextSize = latLabel.GetTextSize()

    canvas = ROOT.TCanvas("c", "Four-partition canvas", 1200, 800)
    # Define height ratios: 70% upper, 30% lower
    upper_height = 0.7
    lower_height = 0.3
    gap = 0.01
    # Create upper container (70% height)
    upper = ROOT.TPad("uppe", "Upper part", 0, lower_height - gap, 1, 1)
    upper.SetMargin(0.16, 0.08, 0.005, 0.075)  # Adjust padding
    upper.Divide(2, 1, 0, 0)  # Divide upper part into left and right
    upper.Draw()
    # Create lower container (30% height)
    lower = ROOT.TPad("lowe", "Lower part", 0, 0, 1, lower_height + gap)
    lower.SetMargin(0.16, 0.08, 0.25, 0.005)  # Adjust padding
    lower.Divide(2, 1, 0, 0)  # Divide lower part into left and right
    lower.Draw()
    left_pad = upper.cd(1)
    frame = left_pad.DrawFrame(0., -0.0, 24.9, 0.31,
                            ';#it{p}_{T} (GeV/#it{c});#it{v}_{2} {SP, |#Delta#it{#eta}| > 1.3}')
    frame.GetXaxis().SetLabelSize(0)  # Hide x-axis labels
    frame.GetXaxis().SetTitleSize(0)
    frame.GetYaxis().SetDecimals()
    polyline_d0_tamu.Draw("F z")
    polyline_d0_catania.Draw("F z")
    graph_d0_Langevin_fnwsnloaver.Draw('same')
    graph_d0_htl.Draw('same')
    graph_d0_lat.Draw('same')
    graph_d0_epos4hq.Draw('same')
    graph_d0_lbt.Draw('same')
    d0_hists[0].Draw('same epz')
    d0_hists[1].Draw('same e2z')
    position_x = 0.54
    y_alice = 0.921
    position_y_top = y_alice
    legend = TLegend(position_x-0.01, position_y_top-0.56, position_x+0.35, position_y_top+0.04)
    legend.Draw("same z")
    legend.SetTextSize(legend_TextSize)
    legend.SetTextFont(42)
    legend.AddEntry(lc_hists[0], "Prompt #Lambda_{c}^{+}", "p")
    legend.AddEntry(d0_hists[0], "Prompt D^{0}", "p")
    legend.AddEntry("", "", "")
    legend.SetNColumns(1)
    legend.AddEntry(graph_d0_lat, "POWLANG lQCD", "l")
    legend.AddEntry(graph_d0_htl, "POWLANG HTL", "l")
    legend.AddEntry(graph_d0_epos4hq, "EPOS4HQ", "l")
    legend.AddEntry(graph_d0_lbt, "LBT-PNP", "l")
    legend.AddEntry(graph_d0_Langevin_fnwsnloaver, "Langevin", "l")
    legend.AddEntry(polyline_d0_catania, "Catania", "F")
    legend.AddEntry(polyline_d0_tamu, "TAMU", "F")
    legend.SetBorderSize(0)
    position_x_l = 0.22
    lat_large.DrawLatex(position_x_l, y_alice, 'ALICE')
    # latLabel.DrawLatex(0.18, 0.74, 'Preliminary')
    lat_mid.DrawLatex(position_x_l, y_alice-0.075, 'Pb#font[122]{-}Pb, 30#font[122]{-}50%')  #  centrality
    lat_mid.DrawLatex(position_x_l, y_alice-0.15, '#sqrt{#it{s}_{NN}} = 5.36 TeV')
    latLabel.DrawLatex(position_x, position_y_top-0.12, "Transport models,  #sqrt{#it{s}_{NN}} = 5.02 TeV")
    # latLabel.DrawLatex(0.18, position_y_top-0.2, "#sqrt{#it{s}_{NN}} = 5.02 TeV")
    # for hist in d0_hists_interp:
    #     hist.Draw('same p')
    ratio_left_pad = lower.cd(1)
    frame = ratio_left_pad.DrawFrame(0., 0.1, 24.9, 15,
                            ';#it{p}_{T} (GeV/#it{c}) ;#frac{data}{model}')
    ratio_left_pad.SetLogy()
    ratio_left_pad.SetGrid(1, 1)
    # frame.GetYaxis().SetDecimals()
    frame.GetYaxis().SetTitleSize(0.14)
    frame.GetXaxis().SetTitleSize(0.14)  
    frame.GetYaxis().SetLabelSize(0.12) 
    frame.GetXaxis().SetLabelSize(0.12) 
    frame.GetYaxis().SetTitleOffset(0.55)
    frame.GetXaxis().SetTitleOffset(1)
    global_lines = []
    line = DrawLineAt0(0.5, 24.9)
    global_lines.append(line)  # Critical: retain reference
    line.Draw("same")
    ratios = []
    for hist in d0_hists_interp_rebin:
        ratio_stat = compute_ratio_graph(d0_hists[0], hist)
        ratio_syst = compute_ratio_graph(d0_hists[1], hist)
        ratio = merge_asymmetric_errors(ratio_stat, ratio_syst)
        # ratio = d0_hists_origin.Clone('ratio_' + hist.GetName())
        # ratio.Divide(hist)
        SetObjectStyle(ratio, color=colors_d0[hist.GetName()[:-9]])
        ratio.Draw('same pez')
        ratios.append(ratio)
    right_pad = upper.cd(2)
    frame = right_pad.DrawFrame(0.5, -0.0, 24.9, 0.31,
                            ';#it{p}_{T} (GeV/#it{c});#it{v}_{2} {SP, |#Delta#it{#eta}| > 1.3}')
    frame.GetXaxis().SetLabelSize(0)  # Hide x-axis labels
    frame.GetXaxis().SetTitleSize(0)
    polyline_lc_tamu.Draw("F z")
    polyline_lc_catania.Draw("F z")
    graph_lc_Langevin_fnwsnloaver.Draw('same')
    graph_lc_htl.Draw('same')
    graph_lc_lat.Draw('same')
    graph_lc_lbt.Draw('same')
    graph_lc_epos4hq.Draw('same')
    lc_hists[0].Draw('same epz')
    lc_hists[1].Draw('same e2z')
    position_x = 0.28
    position_y_top = y_alice
    # legend_r = TLegend(position_x-0.015, position_y_top-0.28, position_x+0.7, position_y_top+0.04)
    # legend_r.Draw("same z")
    # legend_r.SetTextSize(legend_TextSize)
    # legend_r.SetTextFont(42)
    # legend_r.SetFillColorAlpha(ROOT.kWhite, 0)  # 0 means fully transparent
    # legend_r.AddEntry(lc_hists[0], "Prompt #Lambda_{c}^{+}", "p")
    # legend_r.AddEntry("", "", "")
    # legend_r.AddEntry("", "", "")
    # legend_r.AddEntry("", "", "")
    # legend_r.SetNColumns(2)
    # legend_r.AddEntry(graph_lc_htl, "POWLANG HTL #Lambda_{c}^{+}", "l")
    # legend_r.AddEntry(graph_lc_lbt, "LBT-PNP #Lambda_{c}^{+}", "l")  # , #sqrt{#it{s}_{NN}} = 5.36 TeV
    # legend_r.AddEntry(graph_lc_lat, "POWLANG lQCD #Lambda_{c}^{+}", "l")
    # legend_r.AddEntry(graph_lc_Langevin_fnwsnloaver, "Langevin #Lambda_{c}^{+}", "l")
    # legend_r.AddEntry(graph_lc_epos4hq, "EPOS4HQ #Lambda_{c}^{+}", "l")
    # legend_r.AddEntry(polyline_lc_catania, "Catania #Lambda_{c}^{+}", "F")
    # legend_r.AddEntry("", "", "")
    # legend_r.AddEntry(polyline_lc_tamu, "TAMU #Lambda_{c}^{+}", "F")
    # legend_r.SetBorderSize(0)
    # latLabel.DrawLatex(position_x, position_y_top-0.045, "Transport models,  #sqrt{#it{s}_{NN}} = 5.02 TeV")
    # for hist in lc_hists_interp:
    #     hist.Draw('same ep')
    ratio_right_pad = lower.cd(2)
    frame = ratio_right_pad.DrawFrame(0.5, 0.1, 24.9, 15,
                            ';#it{p}_{T} (GeV/#it{c}) ;#frac{data}{model}')
    ratio_right_pad.SetLogy()
    ratio_right_pad.SetGrid(1, 1)
    frame.GetYaxis().SetTitleSize(0.14)  # Set X-axis title font size
    frame.GetXaxis().SetTitleSize(0.14)  # Set X-axis title font size
    frame.GetYaxis().SetLabelSize(0.12)  # Set Y-axis title font size
    frame.GetXaxis().SetLabelSize(0.12)  # Set Y-axis title font size
    frame.GetYaxis().SetTitleOffset(0.55)
    frame.GetXaxis().SetTitleOffset(1)
    line = DrawLineAt0(0.5, 24.9)
    line.Draw('same')
    for hist in lc_hists_interp_rebin:
        ratio_stat = compute_ratio_graph(lc_hists[0], hist)
        ratio_syst = compute_ratio_graph(lc_hists[1], hist)
        ratio = merge_asymmetric_errors(ratio_stat, ratio_syst)
        # ratio = lc_hists_origin.Clone('ratio_' + hist.GetName())
        SetObjectStyle(ratio, color=colors_lc[hist.GetName()[:-9]])
        # ratio.Divide(hist)
        ratio.Draw('same ep')
        ratios.append(ratio)
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f'{outDir}/compare-model.pdf')
    if debug:
        outFileName = f'{outDir}/compare-model.root'
        outFile = TFile(outFileName, "recreate")
        outFile.cd()
        canvas.cd()
        d0_hists[0].Write()
        d0_hists[1].Write()
        lc_hists[0].Write()
        lc_hists[1].Write()
        d0_tot.Write()
        lc_tot.Write()
        d0_hists_origin.Write()
        lc_hists_origin.Write()
        canvas_test = ROOT.TCanvas("test", "test", 1200, 800)
        # for hist in ratios:
        #     hist.Draw('same')
        #     hist.Write()
        d0_graphs = [polyline_d0_tamu, polyline_d0_catania, graph_d0_Langevin_fnwsnloaver, graph_d0_htl, graph_d0_lat, graph_d0_lbt, graph_d0_epos4hq]
        lc_graphs = [polyline_lc_tamu, polyline_lc_catania, graph_lc_Langevin_fnwsnloaver, graph_lc_htl, graph_lc_lat, graph_lc_lbt, graph_lc_epos4hq]
        for i in range(len(d0_graphs)):
            name = d0_hists_interp[i].GetName()[:-3]
            canvas_test = ROOT.TCanvas(name, name, 1200, 800)
            option = ''
            if 'tamu' in name or 'catania' in name:
                option = 'F'
            canvas_test.Divide(2, 1, 0, 0)
            canvas_test.cd(1).DrawFrame(0., -0.0, 24.9, 0.31,
                                    ';#it{p}_{T} (GeV/#it{c});#it{v}_{2} {SP, |#Delta#it{#eta}| > 1.3}')
            d0_graphs[i].Draw(option)
            d0_hists_interp[i].Draw('same ep')
            d0_hists[0].Draw('same epz')
            d0_hists[1].Draw('same e2z')
            canvas_test.cd(2).DrawFrame(0., -0.0, 24.9, 0.31,
                                    ';#it{p}_{T} (GeV/#it{c});#it{v}_{2} {SP, |#Delta#it{#eta}| > 1.3}')
            lc_graphs[i].Draw(option)
            lc_hists_interp[i].Draw('same ep')
            lc_hists[0].Draw('same epz')
            lc_hists[1].Draw('same e2z')
            lat_large.DrawLatex(position_x+0.4, position_y_top-0.04, name)
            canvas_test.Update()
            canvas_test.Write()
            d0_hists_interp[i].Write()
            lc_hists_interp[i].Write()
            # canvas_test.SaveAs(f'{outDir}/{name}.pdf')
            
        canvas.Write()
        canvas_test.Write()
        outFile.Close()


def compare_with_data(color_lc=ROOT.kRed+2, color_d0=ROOT.kBlue+2, do_ncq=False):
    '''compare v2 of D0 and Lambda_c with light-flavor hadrons (K0s and Lambda)'''
    d0_hists[0].SetMarkerSize(1)
    lc_hists[0].SetMarkerSize(1)
    data_path = '../input-data/light-flavor-data/HEPData-ins2093750-v1-csv'
    data_path_sp = '../input-data/light-flavor-data/HEPData-ins1672822-v1-csv-SP'

    data_ks_30_40 = f'{data_path}/Table23.csv'
    data_ks_40_50 = f'{data_path}/Table32.csv'
    df_ks_30_50 = preprocess_data([data_ks_30_40, data_ks_40_50])
    data_ks_30_40_sp = f'{data_path_sp}/Table73.csv'
    data_ks_40_50_sp = f'{data_path_sp}/Table74.csv'
    df_ks_30_50_sp = preprocess_data([data_ks_30_40_sp, data_ks_40_50_sp], header=12)
    
    h_ks_30_50 = fill_graph(df_ks_30_50)
    SetObjectStyle(h_ks_30_50, color=ROOT.kAzure+1, markerstyle=ROOT.kFullSquare, linewidth=1)
    h_ks_30_50_sp = fill_graph(df_ks_30_50_sp)
    SetObjectStyle(h_ks_30_50_sp, color=ROOT.kAzure+1, markerstyle=ROOT.kOpenSquare, linewidth=1)

    data_lambda_30_40 = f'{data_path}/Table24.csv'
    data_lambda_40_50 = f'{data_path}/Table33.csv'
    df_lambda_30_50 = preprocess_data([data_lambda_30_40, data_lambda_40_50])
    data_lambda_30_40_sp = f'{data_path_sp}/Table87.csv'
    data_lambda_40_50_sp = f'{data_path_sp}/Table88.csv'
    df_lambda_30_50_sp = preprocess_data([data_lambda_30_40_sp, data_lambda_40_50_sp], header=12)
    
    h_lambda_30_50 = fill_graph(df_lambda_30_50)
    SetObjectStyle(h_lambda_30_50, color=ROOT.kMagenta, markerstyle=ROOT.kFullSquare, linewidth=1)
    h_lambda_30_50_sp = fill_graph(df_lambda_30_50_sp)
    SetObjectStyle(h_lambda_30_50_sp, color=ROOT.kMagenta, markerstyle=ROOT.kOpenSquare, linewidth=1)
    all_data = {"ks":df_ks_30_50_sp,
                "lambda":df_lambda_30_50_sp
                }
    if do_ncq:
        return all_data
    
    lat_large, lat_mid, latLabel = get_latex()
    legend_TextSize = latLabel.GetTextSize()
    canvas = ROOT.TCanvas("canvas", "Canvas", 800, 600)
    frame = canvas.DrawFrame(0.0, -0.0, 24.9, 0.4,
                            ';#it{p}_{T} (GeV/#it{c});#it{v}_{2}')
    xaxis = frame.GetXaxis()
    yaxis = frame.GetYaxis()
    frame.GetYaxis().SetDecimals()
    
    xaxis.SetTitleOffset(1.2)
    yaxis.SetTitleOffset(1.1)
    d0_hists[0].Draw('same epz')
    d0_hists[1].Draw('same e2z')
    lc_hists[0].Draw('same epz')
    lc_hists[1].Draw('same e2z')
    h_lambda_30_50_sp.Draw('same epz')
    h_ks_30_50_sp.Draw('same epz')
    
    position_x = 0.44
    legend = TLegend(position_x, 0.53, 0.63, 0.81)
    legend.SetTextSize(legend_TextSize)
    legend.SetTextFont(42)
    legend.AddEntry(lc_hists[0], "Prompt #Lambda_{c}^{+}", "p")
    legend.AddEntry(d0_hists[0], "Prompt D^{0}", "p")
    legend.AddEntry("", "", "") 
    # legend.AddEntry(h_lambda_30_50, "#Lambda(#bar{#Lambda}) JHEP 05 (2023) 243", "p")
    # legend.AddEntry(h_ks_30_50, "K_{S}^{0}     JHEP 05 (2023) 243", "p")
    legend.AddEntry(h_lambda_30_50_sp, "#Lambda(#bar{#Lambda}) JHEP 09 (2018) 006", "p")
    legend.AddEntry(h_ks_30_50_sp, "K_{S}^{0}     JHEP 09 (2018) 006", "p")
    legend.SetBorderSize(0)
    legend.Draw("same")
    
    lat_large.DrawLatex(0.18, 0.83, 'ALICE')
    lat_mid.DrawLatex(0.18, 0.77, 'Pb#font[122]{-}Pb, 30#font[122]{-}50% ')
    latLabel.DrawLatex(position_x+0.02, 0.83, "#it{v}_{2} {SP, |#Delta#it{#eta}| > 1.3},  #sqrt{#it{s}_{NN}} = 5.36 TeV")
    latLabel.DrawLatex(position_x+0.02, 0.66, "#it{v}_{2} {2, |#Delta#it{#eta}| > 2},  #sqrt{#it{s}_{NN}} = 5.02 TeV")
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f'{outDir}/compare-LF.pdf')
    if debug:
        outFileName = f"{outDir}/compare-LF.root"
        outFile = TFile(outFileName, "recreate")
        outFile.cd()
        d0_hists[0].Write()
        d0_hists[1].Write()
        lc_hists[0].Write()
        lc_hists[1].Write()
        h_ks_30_50.Write()
        h_ks_30_50_sp.Write()
        h_lambda_30_50.Write()
        h_lambda_30_50_sp.Write()
        canvas.Write()
        outFile.Close()


def compare_dataWmodel_ncq(color_lc=ROOT.kRed+2, color_d0=ROOT.kBlue+2, do_ket_nq=False):
    '''compare v2/nq vs kEt/nq(pt/nq) of D0 and Lambda_c with light-flavor hadrons (K0s and Lambda) and models'''
    all_data = compare_with_data(do_ncq=True)
    all_data_scaled = preprocess_ncq(all_data, do_ket_nq=do_ket_nq, ismodel=False)
    columns = ["pt/nq", "v2/nq", "Total Error/nq"]
    if do_ket_nq:
        columns = ["kEt/nq", "v2/nq", "Total Error/nq"]
    h_ks_30_50_scaled = fill_graph(all_data_scaled['ks'], columns)
    h_lambda_30_50_scaled = fill_graph(all_data_scaled['lambda'], columns)
    SetObjectStyle(h_ks_30_50_scaled, color=ROOT.kAzure+1, markerstyle=ROOT.kOpenSquare, linewidth=1)
    SetObjectStyle(h_lambda_30_50_scaled, color=ROOT.kMagenta, markerstyle=ROOT.kOpenSquare, linewidth=1)

    all_model_data = compare_with_model(do_ncq=True)
    all_model_data_scaled = preprocess_ncq(all_model_data, do_ket_nq=do_ket_nq, ismodel=True)
    tamu_band_lc = get_band(all_model_data_scaled["tamu"]["lc"][0].iloc[:,0], all_model_data_scaled["tamu"]["lc"][1].iloc[:,0],
                         all_model_data_scaled["tamu"]["lc"][0].iloc[:,1], all_model_data_scaled["tamu"]["lc"][1].iloc[:,1], color=color_lc)
    tamu_band_d0 = get_band(all_model_data_scaled["tamu"]["d0"][0].iloc[:,0], all_model_data_scaled["tamu"]["d0"][1].iloc[:,0],
                         all_model_data_scaled["tamu"]["d0"][0].iloc[:,1], all_model_data_scaled["tamu"]["d0"][1].iloc[:,1], color=color_d0)
    catania_band_lc = get_band(all_model_data_scaled["catania"]["lc"][0].iloc[:,0], all_model_data_scaled["catania"]["lc"][0].iloc[:,0],
                         all_model_data_scaled["catania"]["lc"][0].iloc[:,1], all_model_data_scaled["catania"]["lc"][0].iloc[:,2], color=color_lc)
    catania_band_d0 = get_band(all_model_data_scaled["catania"]["d0"][0].iloc[:,0], all_model_data_scaled["catania"]["d0"][0].iloc[:,0],
                         all_model_data_scaled["catania"]["d0"][0].iloc[:,1], all_model_data_scaled["catania"]["d0"][0].iloc[:,2], color=color_d0)
    tamu_band_lc.SetFillColorAlpha(colors_lc["tamu"], 0.5)
    tamu_band_d0.SetFillColorAlpha(colors_d0["tamu"], 0.4)
    # tamu_band_lc.SetFillStyle(3002) 
    # tamu_band_d0.SetFillStyle(3002) 
    catania_band_lc.SetFillColorAlpha(colors_lc["catania"], 0.3)
    catania_band_d0.SetFillColorAlpha(colors_d0["catania"], 0.8)

    graph_lc_Langevin_fnwsnloaver = ROOT.TGraph(len(all_model_data_scaled["langevin"]["lc"][0].iloc[:,0]), 
                                                array.array('d', all_model_data_scaled["langevin"]["lc"][0].iloc[:,0]), 
                                                array.array('d', all_model_data_scaled["langevin"]["lc"][0].iloc[:,1]))
    graph_lc_Langevin_fnwsnloaver.SetLineColor(ROOT.kRed+2)
    graph_lc_Langevin_fnwsnloaver.SetLineWidth(2)
    graph_lc_Langevin_fnwsnloaver.SetLineStyle(2)
    graph_d0_Langevin_fnwsnloaver = ROOT.TGraph(len(all_model_data_scaled["langevin"]["d0"][0].iloc[:,0]), 
                                                array.array('d', all_model_data_scaled["langevin"]["d0"][0].iloc[:,0]), 
                                                array.array('d', all_model_data_scaled["langevin"]["d0"][0].iloc[:,1]))
    graph_d0_Langevin_fnwsnloaver.SetLineColor(ROOT.kBlue+2)
    graph_d0_Langevin_fnwsnloaver.SetLineWidth(2)
    graph_d0_Langevin_fnwsnloaver.SetLineStyle(2)

    graph_lc_htl = ROOT.TGraph(len(all_model_data_scaled["htl"]["lc"][0].iloc[:,0]), 
                                                array.array('d', all_model_data_scaled["htl"]["lc"][0].iloc[:,0]), 
                                                array.array('d', all_model_data_scaled["htl"]["lc"][0].iloc[:,1]))
    graph_lc_htl.SetLineColor(ROOT.kRed+1)
    graph_lc_htl.SetLineWidth(2)
    graph_lc_htl.SetLineStyle(6)
    graph_d0_htl = ROOT.TGraph(len(all_model_data_scaled["htl"]["d0"][0].iloc[:,0]), 
                                                array.array('d', all_model_data_scaled["htl"]["d0"][0].iloc[:,0]), 
                                                array.array('d', all_model_data_scaled["htl"]["d0"][0].iloc[:,1]))
    graph_d0_htl.SetLineColor(ROOT.kCyan+2)
    graph_d0_htl.SetLineWidth(2)
    graph_d0_htl.SetLineStyle(6)

    graph_lc_lat = ROOT.TGraph(len(all_model_data_scaled["latQCD"]["lc"][0].iloc[:,0]), 
                                                array.array('d', all_model_data_scaled["latQCD"]["lc"][0].iloc[:,0]), 
                                                array.array('d', all_model_data_scaled["latQCD"]["lc"][0].iloc[:,1]))
    graph_lc_lat.SetLineColor(ROOT.kOrange+1)
    graph_lc_lat.SetLineWidth(2)
    graph_lc_lat.SetLineStyle(9)
    graph_d0_lat = ROOT.TGraph(len(all_model_data_scaled["latQCD"]["d0"][0].iloc[:,0]), 
                                                array.array('d', all_model_data_scaled["latQCD"]["d0"][0].iloc[:,0]), 
                                                array.array('d', all_model_data_scaled["latQCD"]["d0"][0].iloc[:,1]))
    graph_d0_lat.SetLineColor(ROOT.kP10Green)
    graph_d0_lat.SetLineWidth(2)
    graph_d0_lat.SetLineStyle(9)


    graph_lc_lbt = ROOT.TGraph(len(all_model_data_scaled["lbt"]["lc"][0].iloc[:,0]), 
                                                array.array('d', all_model_data_scaled["lbt"]["lc"][0].iloc[:,0]), 
                                                array.array('d', all_model_data_scaled["lbt"]["lc"][0].iloc[:,1]))
    graph_lc_lbt.SetLineColor(ROOT.kP10Brown)
    graph_lc_lbt.SetLineWidth(2)
    graph_lc_lbt.SetLineStyle(7)
    graph_d0_lbt = ROOT.TGraph(len(all_model_data_scaled["lbt"]["d0"][0].iloc[:,0]), 
                                                array.array('d', all_model_data_scaled["lbt"]["d0"][0].iloc[:,0]), 
                                                array.array('d', all_model_data_scaled["lbt"]["d0"][0].iloc[:,1]))
    graph_d0_lbt.SetLineColor(ROOT.kP10Violet)
    graph_d0_lbt.SetLineWidth(2)
    graph_d0_lbt.SetLineStyle(7)

    graph_lc_epos4hq = ROOT.TGraph(len(all_model_data_scaled["epos4hq"]["lc"][0].iloc[:, 0]), array.array('d', all_model_data_scaled["epos4hq"]["lc"][0].iloc[:, 0]), array.array('d', all_model_data_scaled["epos4hq"]["lc"][0].iloc[:, 1]))
    graph_lc_epos4hq.SetLineColor(colors_lc["epos4hq"])
    graph_lc_epos4hq.SetLineWidth(3)
    graph_lc_epos4hq.SetLineStyle(3)
    graph_d0_epos4hq = ROOT.TGraph(len(all_model_data_scaled["epos4hq"]["d0"][0].iloc[:, 0]), array.array('d', all_model_data_scaled["epos4hq"]["d0"][0].iloc[:, 0]), array.array('d', all_model_data_scaled["epos4hq"]["d0"][0].iloc[:, 1]))
    graph_d0_epos4hq.SetLineColor(colors_d0["epos4hq"])
    graph_d0_epos4hq.SetLineWidth(3)
    graph_d0_epos4hq.SetLineStyle(3)
    result_nq = f'../input-data/lc-d0-data/usekEt_{do_ket_nq}-test-syst.root'
    # d0_hists = read_hists(result_nq, markerstyle=ROOT.kFullCircle, colors=[ROOT.kBlue], gname=['d0_hist_prompt', 'gVnPromptSystTot-d0'])
    # lc_hists = read_hists(result_nq, markerstyle=ROOT.kFullCircle, colors=[ROOT.kRed], gname=['hist_prompt_all', 'gVnPromptSystTot-All'])
    file = TFile.Open(result_nq)
    if not file:
        print('error: faild to open file')
        return
    d0_h_prompt_cent = file.Get('d0_hist_prompt')
    SetObjectStyle(d0_h_prompt_cent, color=color_d0, markerstyle=ROOT.kFullCircle, linewidth=1, fillalpha=0.2)
    d0_h_prompt_systtot = file.Get('gVnPromptSystTot-d0')
    SetObjectStyle(d0_h_prompt_systtot, color=color_d0)
    d0_h_prompt_systtot.SetFillStyle(0)
    d0_h_prompt_systtot.SetLineWidth(1)

    lc_h_prompt_cent = file.Get('hist_prompt_all')
    SetObjectStyle(lc_h_prompt_cent, color=color_lc, markerstyle=ROOT.kFullCircle, linewidth=1)  #  linealpha=0.8
    lc_h_prompt_systtot = file.Get('gVnPromptSystTot-All')
    SetObjectStyle(lc_h_prompt_systtot, color=color_lc)
    lc_h_prompt_systtot.SetFillStyle(0)
    lc_h_prompt_systtot.SetLineWidth(1)

    d0_hists = [d0_h_prompt_cent, d0_h_prompt_systtot]
    lc_hists = [lc_h_prompt_cent, lc_h_prompt_systtot]
    canvas = ROOT.TCanvas("canvas", "Canvas", 2400, 800)
    canvas.Divide(3, 1, 0, 0)
    x_label = "#it{p}_{T}/#it{n}_{q}"
    x_min = 0.2
    x_max = 12+0.5
    y_max = 0.15
    if do_ket_nq:
        x_max = 11.9
        x_min = -0.15
        x_label = "kE_{T}/#it{n}_{q}"
    y_label = "#it{v}_{2}/#it{n}_{q}"
    pad_left = canvas.cd(1)
    frame = pad_left.DrawFrame(x_min, -0.0, x_max, y_max,
                            f';{x_label} (GeV/#it{{c}})  ;{y_label}')
    frame.GetYaxis().SetDecimals()
    d0_hists[0].Draw('same epz')
    d0_hists[1].Draw('same e2z')
    lc_hists[0].Draw('same epz')
    lc_hists[1].Draw('same e2z')

    h_ks_30_50_scaled.Draw('same epz')
    h_lambda_30_50_scaled.Draw('same epz')

    lat_large, lat_mid, latLabel = get_latex()
    legend_TextSize = latLabel.GetTextSize()

    position_x = 0.45
    position_y_top = 0.9
    position_y_bottom = position_y_top-0.3
    legend = TLegend(position_x, position_y_bottom-0.05, position_x+0.2, position_y_top-0.02)
    legend.SetTextSize(legend_TextSize)
    legend.SetTextFont(42)
    legend.AddEntry(lc_hists[0], "Prompt #Lambda_{c}^{+}", "p")
    legend.AddEntry(d0_hists[0], "Prompt D^{0}", "p")
    legend.AddEntry("", "", "")
    legend.AddEntry("", "", "")

    legend.AddEntry(h_ks_30_50_scaled, "K_{S}^{0}     JHEP 09 (2018) 006", "p")
    legend.AddEntry(h_lambda_30_50_scaled, "#Lambda(#bar{#Lambda}) JHEP 09 (2018) 006", "p")
    legend.SetBorderSize(0)
    legend.Draw("same")

    lat_large.DrawLatex(0.2, position_y_top, 'ALICE')
    lat_mid.DrawLatex(0.2, position_y_top-0.06, 'Pb#font[122]{-}Pb, 30#font[122]{-}50%')
    latLabel.DrawLatex(position_x+0.03, position_y_top, "#it{v}_{2} {SP, |#Delta#it{#eta}| > 1.3},  #sqrt{#it{s}_{NN}} = 5.36 TeV")
    latLabel.DrawLatex(position_x+0.03, position_y_top-0.21, "#it{v}_{2} {2, |#Delta#it{#eta}| > 2},  #sqrt{#it{s}_{NN}} = 5.02 TeV")
    pad_mid = canvas.cd(2)
    frame = pad_mid.DrawFrame(x_min, -0.0, x_max, y_max,
                            f';{x_label} (GeV/#it{{c}})  ;{y_label}')
    position_x = 0.22
    tamu_band_lc.Draw('F')
    tamu_band_d0.Draw('F')
    graph_lc_htl.Draw('same')
    graph_d0_htl.Draw('same')
    graph_lc_lat.Draw('same')
    graph_d0_lat.Draw('same')
    d0_hists[0].Draw('same epz')
    d0_hists[1].Draw('same e2z')
    lc_hists[0].Draw('same epz')
    lc_hists[1].Draw('same e2z')
    
    legend_m = TLegend(position_x-0.11, position_y_bottom+0.08, position_x+0.75, position_y_top-0.03)
    legend_m.SetFillStyle(0)
    legend_m.SetBorderSize(0)
    legend_m.SetNColumns(2)
    legend_m.Draw("same")
    legend_m.SetTextSize(legend_TextSize)
    legend_m.SetTextFont(42)
    
    legend_m.AddEntry(graph_lc_htl, "POWLANG HTL #Lambda_{c}^{+}", "l")
    legend_m.AddEntry(graph_d0_htl, "POWLANG HTL D^{0}", "l")
    legend_m.AddEntry(graph_lc_lat, "POWLANG lQCD #Lambda_{c}^{+}", "l")
    legend_m.AddEntry(graph_d0_lat, "POWLANG lQCD D^{0}", "l")
    legend_m.AddEntry(tamu_band_lc, "TAMU #Lambda_{c}^{+}", "F")
    legend_m.AddEntry(tamu_band_d0, "TAMU D^{0}", "F")
    
    latLabel.DrawLatex(position_x-0.1, position_y_top, "Transport models,  #sqrt{#it{s}_{NN}} = 5.02 TeV")
    pad_right = canvas.cd(3)
    frame = pad_right.DrawFrame(x_min, -0.0, x_max, y_max,
                            f';{x_label} (GeV/#it{{c}})  ;{y_label}')
    catania_band_lc.Draw('F')
    catania_band_d0.Draw('F')
    graph_lc_lbt.Draw('same')
    graph_d0_lbt.Draw('same')
    graph_lc_Langevin_fnwsnloaver.Draw('same')
    graph_d0_Langevin_fnwsnloaver.Draw('same')
    graph_lc_epos4hq.Draw('same')
    graph_d0_epos4hq.Draw('same')
    d0_hists[0].Draw('same epz')
    d0_hists[1].Draw('same e2z')
    lc_hists[0].Draw('same epz')
    lc_hists[1].Draw('same e2z')
    
    legend_r = TLegend(position_x-0.05, position_y_bottom+0.04, position_x+0.73, position_y_top-0.03)
    legend_r.SetFillStyle(0)
    legend_r.SetBorderSize(0)
    legend_r.SetNColumns(2)
    legend_r.Draw("same")
    legend_r.SetTextSize(legend_TextSize)
    legend_r.SetTextFont(42)
    legend_r.AddEntry(graph_lc_epos4hq, "EPOS4HQ #Lambda_{c}^{+}", "l")
    legend_r.AddEntry(graph_d0_epos4hq, "EPOS4HQ D^{0}", "l")
    legend_r.AddEntry(graph_lc_lbt, "LBT-PNP #Lambda_{c}^{+}", "l")
    legend_r.AddEntry(graph_d0_lbt, "LBT-PNP D^{0}", "l")
    legend_r.AddEntry(graph_lc_Langevin_fnwsnloaver, "Langevin #Lambda_{c}^{+}", "l")
    legend_r.AddEntry(graph_d0_Langevin_fnwsnloaver, "Langevin D^{0}", "l")
    legend_r.AddEntry(catania_band_lc, "Catania #Lambda_{c}^{+}", "F")
    legend_r.AddEntry(catania_band_d0, "Catania D^{0}", "F")
    
    latLabel.DrawLatex(position_x-0.04, position_y_top, "Transport models,  #sqrt{#it{s}_{NN}} = 5.02 TeV")
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f'{outDir}/ncq.pdf')
    if debug:
        outFileName = f"{outDir}/ncq.root"
        outFile = TFile(outFileName, "recreate")
        outFile.cd()
        d0_hists[0].Write()
        d0_hists[1].Write()
        lc_hists[0].Write()
        lc_hists[1].Write()
        canvas.Write()
        outFile.Close()


def plot_performance(plot_invmass_fit = True, plot_v2ffd_fit = True, plot_cutvar_fit = True):
    SetGlobalStyle(
                padleftmargin=0.15, padrightmargin=0.03, padbottommargin=0.12, padtopmargin=0.06,
                opttitle=1, titleoffsety=1.6, labelsize=0.05, titlesize=0.05,
                labeloffset=0.01, titleoffset=1.2, labelfont=42, titlefont=42)

    axisname = ';#it{p}_{T} (GeV/#it{c}); #it{v}_{2}{SP, |#Delta#it{#eta}| > 1.3}'
    # Initialize TLatex objects for text annotation (different sizes for hierarchy)
    latex = ROOT.TLatex()
    latex.SetTextFont(42)
    latex.SetTextSize(0.06)  # Medium text size for main annotations
    latexdetail = ROOT.TLatex()
    latexdetail.SetTextFont(42)
    latexdetail.SetTextSize(0.045)  # Small text size for detailed annotations
    latexdetail2 = ROOT.TLatex()
    latexdetail2.SetTextFont(42)
    latexdetail2.SetTextSize(0.05)  # Medium-small text size for secondary details
    latexlarge = ROOT.TLatex()
    latexlarge.SetTextFont(42)
    latexlarge.SetTextSize(0.07)  # Large text size for prominent labels (e.g., experiment name)
    legsize = 0.045  # Font size for legend text
    number = '00'  # Identifier for input data version/batch
    # Input directory path containing raw data and fit results
    indir = '../input-data/lc-d0-data/lc-performance'
    # Suffix for output files (includes pT range and data version)
    suffix = f'pass4-pt4-5-{number}'

    # Path to ROOT file with uncorrected raw yields
    ry_file = f'{indir}/raw_yields_uncorr_{number}.root'
    # Flag for single pT bin mode (False = use pT range defined by ptlow/pthigh)
    one_p = False

    # Paths to additional input files
    cutvar_file = f'{indir}/CutVarFrac_corr.root'  # File with cut-variation yield data
    linear_fit_file = f'{indir}/V2VsFrac_combined.root'  # File with v2 vs fraction linear fit results
    ptlow = 4  # Lower bound of transverse momentum range (GeV/c)
    pthigh = 5  # Upper bound of transverse momentum range (GeV/c)

    # --------------------------
    # Plot invariant mass fit (if enabled)
    # --------------------------
    if plot_invmass_fit:
        # Get invariant mass histogram and fit functions (total + background)
        # GetInvMassHistAndFit args: (file path, pT lower, pT upper, single-pT-bin flag)
        if one_p:
            hInvMassD0, fMassTotD0, fMassBkgD0, hV2VsMassD0, fV2TotD0, fV2BkgD0 = GetInvMassHistAndFit(ry_file, 3, 4, 1)
        else:
            hInvMassD0, fMassTotD0, fMassBkgD0, hV2VsMassD0, fV2TotD0, fV2BkgD0 = GetInvMassHistAndFit(ry_file, ptlow, pthigh, 0)
        
        # Initialize axis range defaults for invariant mass and v2 plots
        xmins_mass = [2.18]  # Minimum x-axis (invariant mass, GeV/c²)
        xmaxs_mass = [2.36]  # Maximum x-axis (invariant mass, GeV/c²)
        ymins_mass = [-4000]  # Default minimum y-axis (counts)
        ymaxs_mass = [2e4+5555]  # Default maximum y-axis (counts)

        # Override axis ranges based on data version (number) for multi-pT-bin mode
        if not one_p:
            if number == '00':
                ymins_mass = [1.65e4]
                ymaxs_mass = [2.6e4]
                ymins_v2 = [0.17]  # Minimum y-axis (observed v2)
                ymaxs_v2 = [0.222]  # Maximum y-axis (observed v2)
            elif number == '01':
                ymins_mass = [7.5e3]
                ymaxs_mass = [1.5e4]
                ymins_v2 = [0.24]
                ymaxs_v2 = [0.299]
            elif number == '02':
                ymins_mass = [700]
                ymaxs_mass = [0.65e4]
                ymins_v2 = [0.24]
                ymaxs_v2 = [0.35]
        
        # Adjust axis ranges for single-pT-bin mode
        if one_p:
            ymaxs_mass = [3e4]  # Widen y-axis for counts
        if one_p:
            ymins_v2 = [0.1]  # Lower minimum for v2 axis

        # Axis titles for bottom and top subplots
        # Bottom subplot: Invariant mass vs observed v2
        axisnamebottoms = [';#it{M}(pK#pi) (GeV/#it{c}^{2});#it{v}_{2}^{obs.} {SP, |#Delta#it{#eta}| > 1.3}',
                        ]
        # Top subplot: Invariant mass vs counts (bin width dynamically calculated)
        axisnametops = [f';#it{{M}}(pK#pi) (GeV/#it{{c}}^{{2}});Counts per {hInvMassD0.GetBinWidth(1)*1000:.0f} MeV/#it{{c}}^{{2}}',]

        # Create legend for invariant mass fit (positioned at x:0.22-0.5, y:0.18-0.32)
        legD = GetLegend(xmax=0.5, ncolumns=1, ymin=0.18, ymax=0.32, textsize=0.055, xmin=0.22)
        legD.SetTextSize(legsize)
        legD.AddEntry(fV2TotD0, 'Total fit function', 'l')  # Add total fit to legend (line entry)
        legD.AddEntry(fV2BkgD0, 'Combinatorial background', 'l')  # Add background fit to legend (line entry)

        cD0v2run3, frames = GetCanvas4sub('cDv2run3', 
                                        xmins_mass[0],
                                        xmaxs_mass[0],
                                        ymins_mass[0],
                                        ymaxs_mass[0],
                                        ymins_v2[0],
                                        ymaxs_v2[0],
                                        axisnametops[0],
                                        axisnamebottoms[0]
                                        )

        # Configure axis styling for top subplot (invariant mass vs counts)
        frames[0].GetYaxis().SetNoExponent(False)  # Enable scientific notation for y-axis
        frames[0].GetYaxis().SetNdivisions(508)  # Set y-axis divisions (5 main, 8 minor)
        frames[0].GetXaxis().SetNdivisions(504)  # Set x-axis divisions (5 main, 4 minor)
        # Configure axis styling for bottom subplots (v2 related)
        frames[2].GetXaxis().SetNdivisions(504)
        frames[2].GetYaxis().SetDecimals()  # Show decimal places for y-axis labels
        frames[3].GetYaxis().SetDecimals()

        # Draw top subplot (invariant mass histogram + fits)
        pad_top = cD0v2run3.cd(1)  # Activate top subplot pad
        hInvMassD0.Draw('esame')  # Draw histogram with error bars (e), markers (s), axis (a), mean (m), error (e)
        fMassBkgD0.Draw('same')  # Draw background fit on top of histogram
        fMassTotD0.Draw('same')  # Draw total fit on top of background fit

        # Define text position parameters for annotations
        x_ = 0.2  # Horizontal position (normalized to canvas)
        topy = 0.85  # Vertical position for topmost text
        gap = 0.08  # Vertical spacing between text lines

        # Draw experiment and collision system annotations
        latexlarge.DrawLatexNDC(x_, topy, 'ALICE')  # Experiment name (large font)
        # Collision system: Pb-Pb, 30-50% centrality, center-of-mass energy sqrt(s_NN) = 5.36 TeV
        latex.DrawLatexNDC(x_, topy - gap, 'Pb#font[122]{-}Pb, 30#font[122]{-}50%, #sqrt{#it{s}_{NN}} = 5.36 TeV')
        legD.Draw()  # Draw legend
        # Particle decay channel: Lambda_c+ → p K⁻ π⁺ and charge conjugate
        latex.DrawLatexNDC(x_, topy - 2 * gap, '#Lambda_{c}^{+} #rightarrow pK^{#font[122]{-}}#pi^{+} and charge conj.')
        # Draw pT range annotation (conditional on single-pT-bin mode)
        if one_p:
            latex.DrawLatexNDC(x_, topy - 3 * gap, '3 < #it{p}_{T} < 4 GeV/#it{c}')
        else:
            latex.DrawLatexNDC(x_, topy - 3 * gap, '4 < #it{p}_{T} < 5 GeV/#it{c}')

        # Draw bottom subplot (v2 vs invariant mass)
        pad_bottom = cD0v2run3.cd(3)  # Activate bottom subplot pad
        print(latex.GetTextSize())  # Debug: print current text size of latex object

        hV2VsMassD0.Draw('esame')  # Draw v2 histogram with error bars
        fV2BkgD0.Draw('same')  # Draw background fit for v2
        fV2TotD0.Draw('same')  # Draw total fit for v2

        # Draw BDT score range and observed v2 value (conditional on data version)
        if not one_p:
            if number == '00':
                latexdetail2.DrawLatexNDC(x_, topy, '0 < BDT score to be prompt < 0.03')  # BDT score range for prompt particles
                latexdetail2.DrawLatexNDC(x_, topy - gap, '#it{v}_{2}^{obs.} = 0.012 #pm 0.051')  # Observed v2 with uncertainty
            elif number == '01':
                latexdetail2.DrawLatexNDC(x_, topy, '0.03 < BDT score to be prompt < 0.8')
                latexdetail2.DrawLatexNDC(x_, topy - gap, '#it{v}_{2}^{obs.} = 0.146 #pm 0.051')
            elif number == '02':
                latexdetail2.DrawLatexNDC(x_, topy, '0.8 < BDT score to be prompt < 1')
                latexdetail2.DrawLatexNDC(x_, topy - gap, '#it{v}_{2}^{obs.} = 0.186 #pm 0.027')

    #_____________________________________________________________________________________
    # --------------------------
    # Plot v2 vs fraction linear fit (if enabled)
    # --------------------------
    if plot_v2ffd_fit:
        # Get v2 graph, confidence interval histogram, and linear fit function
        # GetV2HistAndFit args: (file path, pT range suffix, pT lower, pT upper, fit type)
        gv2D0, hv2D0, tf1D0 = GetV2HistAndFit(linear_fit_file,
                                    f'', ptlow, pthigh, 0)
                                    # Commented: Alternative pT range suffix (e.g., 'pt_50_60' for 50-60% centrality)
                                    # f'pt_50_60', ptlow, pthigh, 2)

        # Draw in 4th subplot (bottom-right)
        cD0v2run3.cd(4)  # Activate bottom-right subplot pad
        hframe = frames[3]
        hframe.GetYaxis().SetRangeUser(-0.05, 0.28)  # Set y-axis range for v2 fit

        # Draw confidence interval histogram and style it
        hv2D0.Draw('same')
        color = ROOT.kP10Blue  # Placeholder: unused color variable
        hv2D0.SetFillColorAlpha(ROOT.kAzure+4, 0.4)  # Set fill color (light blue) with transparency (40%)
        gv2D0.Draw('pez same')  # Draw v2 data points (p=points, e=error bars, z=connect points)
        tf1D0.Draw('same')  # Draw linear fit line

        # Draw chi-squared per degree of freedom (χ²/ndf) for fit quality
        px = 0.2  # Horizontal position for χ² label
        latexdetail.DrawLatexNDC(px, 0.2, f'#it{{#chi}}^{{2}}/ndf = {tf1D0.GetChisquare()/tf1D0.GetNDF():.2f}')

        # Create legend for v2 linear fit (positioned at x:0.45-0.75, y:0.75-topy+0.05)
        px2 = 0.45
        legDistr = ROOT.TLegend(px2, 0.75, 0.75, topy+0.05)
        legDistr.SetFillStyle(0)  # Transparent legend background
        legDistr.SetBorderSize(0)  # No legend border
        legDistr.SetTextSize(legsize)
        legDistr.AddEntry(tf1D0, 'Linear fit', 'l')  # Add fit line to legend
        legDistr.AddEntry(hv2D0, '68% confidence level', 'f')  # Add confidence interval (fill entry)
        legDistr.Draw()


    # --------------------------
    # Plot cut-variation yield data (if enabled)
    # --------------------------
    if plot_cutvar_fit:
        infile = ROOT.TFile.Open(cutvar_file)
        # Get the subdirectory corresponding to the current transverse momentum (pT) range 
        # (e.g., "pt4.0_5.0" for pT between 4.0 and 5.0 GeV/c)
        cutvar_dir = infile.Get(f'pt{ptlow}.0_{pthigh}.0')
        
        yield_fd = cutvar_dir.Get(f'hRawYieldFDVsCut_pT{ptlow}.0_{pthigh}.0')
        yield_p = cutvar_dir.Get(f'hRawYieldPromptVsCut_pT{ptlow}.0_{pthigh}.0')
        yield_tot = cutvar_dir.Get(f'hRawYieldsVsCutReSum_pT{ptlow}.0_{pthigh}.0')
        yield_data = cutvar_dir.Get(f'hRawYieldsVsCutPt_pT{ptlow}.0_{pthigh}.0')
        
        # Import ROOT style constants: colors (for lines/fills) and marker styles (for data points)
        from ROOT import kBlack, kRed, kAzure, kGreen, kOpenCircle, kFullCircle, kOpenSquare
        SetObjectStyle(yield_data, linecolor=kBlack, markercolor=kBlack, markerstyle=kFullCircle)
        SetObjectStyle(yield_p, color=kRed+1, fillcolor=kRed+1, fillalpha=0.3)
        SetObjectStyle(yield_fd, color=kAzure+4, fillcolor=kAzure+4, fillalpha=0.3)
        SetObjectStyle(yield_tot, linecolor=kGreen+2)

        # Activate the 2nd subplot (top-right) of the 4-subplot canvas for cut-variation plots
        cD0v2run3.cd(2)
        # Get the frame (axis container) of the 2nd subplot to configure axis styles
        cutvar_hframe = frames[1]
        
        cutvar_hframe.GetYaxis().SetNoExponent(False) 
        cutvar_hframe.GetXaxis().SetLabelSize(0)
        cutvar_hframe.GetYaxis().SetRangeUser(0, 16e3)
        y_pos = 0  # Vertical position of the custom x-axis (aligned to the bottom of the subplot)

        # Create a custom x-axis (TGaxis) to replace the default frame axis (for precise range control)
        custom_x_axis = ROOT.TGaxis(
            0.5, y_pos, 20.5, y_pos,    # Axis drawing range: from (x=0.5, y=y_pos) to (x=20.5, y=y_pos) (canvas coordinates)
            -0.0005, 0.0195,            # Data range of the axis (matches the cut parameter range of the histograms)
            505,                        # Tick mark configuration: 5 major ticks, 0 minor ticks, optimization flag 5
            "L"                         # Label position: draw labels on the "Left" (bottom) side of the axis
        )
        custom_x_axis.SetLabelFont(42)
        custom_x_axis.SetLabelSize(0.05)
        custom_x_axis.Draw("same")                    # Draw the custom x-axis on top of the subplot

        # - Font size matches other legends (legsize = 0.045)
        legDistr_cutvar = ROOT.TLegend(px2, 0.5, 0.8, topy+0.05)
        legDistr_cutvar.SetFillStyle(0)
        legDistr_cutvar.SetBorderSize(0)
        legDistr_cutvar.SetTextSize(legsize)
        legDistr_cutvar.AddEntry(yield_p, 'Prompt #Lambda_{c}^{+}', 'f')
        legDistr_cutvar.AddEntry(yield_fd, 'Non-prompt #Lambda_{c}^{+}', 'f')
        legDistr_cutvar.AddEntry(yield_data, 'Data', 'lpe')
        legDistr_cutvar.AddEntry(yield_tot, 'Total', 'l')
        legDistr_cutvar.Draw('same')  # Draw the legend on the subplot

        yield_fd.DrawCopy('histsame')
        yield_p.DrawCopy('histsame')
        yield_tot.Draw('same')
        yield_data.Draw('same p')

    # Save the entire 4-subplot canvas as a pdf file to the output directory
    cD0v2run3.SaveAs(f'{outDir}/performance.pdf')


if __name__ == "__main__":
    global colors_lc, colors_d0, d0_hists, lc_hists, outDir, debug
    color_lc = ROOT.TColor.GetColor("#CD5C5C")
    color_d0 = ROOT.kBlue + 2
    colors_lc = {"langevin":ROOT.kRed+2, "htl":ROOT.kRed+1, "latQCD":ROOT.kOrange+1, "lbt":ROOT.kP10Brown,
                "tamu":ROOT.kP8Pink, "catania":ROOT.kMagenta, "epos4hq":ROOT.TColor.GetColor("#8B0000")}
    colors_d0 = {"langevin":ROOT.kBlue+2, "htl":ROOT.kCyan+2, "latQCD":ROOT.kP10Green, "lbt":ROOT.kP10Violet,
                "tamu":ROOT.kBlue+2, "catania":ROOT.kAzure+1, "epos4hq":ROOT.TColor.GetColor("#4169E1")}
    outDir = '.'
    d0_file_path = '../input-data/lc-d0-data/d0-promptvn_withsysttest.root'
    d0_hists = read_hists(d0_file_path, ROOT.kFullCircle, colors=[color_d0], markersize=2)  # ROOT.kOpenSquare
    lc_file_path = '../input-data/lc-d0-data/lc-prompt-allpt-wTotsyst.root'
    lc_hists = read_hists(lc_file_path, ROOT.kFullSquare, colors=[color_lc], markersize=2)  # ROOT.kOpenSquare
    debug = False
    # debug = True
    plot_performance()
    compare_with_data(color_lc=color_lc, color_d0=color_d0)
    compare_dataWmodel_ncq(color_lc=color_lc, color_d0=color_d0, do_ket_nq=True)  # true:ket/nq;
    compare_dataWmodel_ncq(color_lc=color_lc, color_d0=color_d0, do_ket_nq=False)  # false:pt/nq;
    d0_hists = read_hists(d0_file_path, ROOT.kFullCircle, colors=[color_d0], markersize=2)  # ROOT.kOpenSquare
    lc_hists = read_hists(lc_file_path, ROOT.kFullSquare, colors=[color_lc], markersize=2)  # ROOT.kOpenSquare
    compare_allD(color_lc=color_lc, color_d0=color_d0)
    compare_allD(color_lc=color_lc, color_d0=color_d0, no_J_Psi=True)
    compare_allD(color_lc=color_lc, color_d0=color_d0, no_Tamu=True)
    # black
    color_lc = ROOT.kBlack
    color_d0 = ROOT.kBlack
    d0_hists = read_hists(d0_file_path, ROOT.kFullCircle, colors=[color_d0], markersize=2)  # ROOT.kOpenSquare
    lc_hists = read_hists(lc_file_path, ROOT.kFullSquare, colors=[color_lc], markersize=2)  # ROOT.kOpenSquare
    compare_with_model(color_lc=color_lc, color_d0=color_d0)
    pdf_paths = ['ncq.pdf', 'compare-model.pdf', 'performance.pdf', 'compare-LF.pdf', 'LcVsAllD_wTamu.pdf', 'LcVsAllD_woTamu.pdf', 'LcVsAllD_wTamu_woJpsi.pdf']
    pdf2eps_imagemagick(pdf_paths, target_format='png')
    print("Done!")