import ROOT
import array
import os
import re
import yaml
import numpy as np
import subprocess
from ROOT import TFile, TLegend
import sys

sys.path.append('./')
from plot_untils import (
    GetLegend, SetGlobalStyle, SetObjectStyle, GetInvMassHistAndFit, GetV2HistAndFit,
    preprocess_ncq, read_txt, preprocess,
    preprocess_data, read_hists, get_band, get_latex, fill_graph,
    scale_x_errors, preprocess_graph_ncq,
)

# Load configuration
with open("plot_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Global variables
OUT_DIR = config['general']['out_dir']
DEBUG = config['general']['debug']
COLORS = config['colors']
MODEL_CFG = config['model']
DATA_PATHS = config['data_paths']
PLOT_STYLE = config['plot_style']
PERF_CFG = config['performance']
NCQ_CFG = config['ncq']
MARKERS = config['markers']

# Create output directory
os.makedirs(OUT_DIR, exist_ok=True)

# Helper function to get ROOT color/style constants
def get_root_constant(name):
    """
    Get ROOT constant with support for arithmetic expressions (e.g. "kGray+2")
    
    Args:
        name: Constant name (str) or non-string type
        
    Returns:
        Calculated constant value
    """
    if not isinstance(name, str):
        return name

    # Parse constant name and expression
    match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)([+\-*/%].*)?$', name.strip())
    if not match:
        raise ValueError(f"Invalid format: {name}")

    const_name, expr = match.groups()
    base_val = getattr(ROOT, const_name)
    
    # Return base value or calculate expression
    return base_val if not expr else eval(f"{base_val}{expr}", {}, {})

# Helper function to load histograms
def load_hists(file_path, marker, color, size=MARKERS['size']['default'], gname=None):
    gname = gname or ['gvn_prompt_stat', 'tot_syst']
    hist = read_hists(
        file_path,
        markerstyle=get_root_constant(marker),
        colors=[get_root_constant(color)],
        markersize=size,
        gname=gname
    )
    scale_x_errors(hist[1], target_graph=hist[0], scale_factor=0.6)
    return hist

# Hide log labels (simplified)
def hide_some_log_labels(frame, ticks_to_show=None):
    ticks_to_show = ticks_to_show or config['log_labels']['ticks_to_show']
    xaxis = frame.GetXaxis()
    label_size = xaxis.GetLabelSize()
    label_font = xaxis.GetLabelFont()
    label_color = xaxis.GetLabelColor()
    xmin, xmax = xaxis.GetXmin(), xaxis.GetXmax()
    ymin = frame.GetYaxis().GetXmin()
    ymax = frame.GetYaxis().GetXmax()
    label_offset = xaxis.GetLabelOffset()
    
    xaxis.SetLabelSize(0)
    graphics = []
    
    for tick in ticks_to_show:
        if xmin <= tick <= xmax:
            label = f"{tick:.1f}" if tick != int(tick) else f"{int(tick)}"
            text = ROOT.TLatex()
            text.SetTextAlign(22)
            text.SetTextSize(label_size)
            text.SetTextFont(label_font)
            text.SetTextColor(label_color)
            
            y_pos = ymin - label_offset * (ymax - ymin) * 5
            text.DrawLatex(tick, y_pos, label)
            graphics.append(text)
    
    return graphics

# Compare Lc with all D mesons
def compare_allD(color_lc, color_d0, no_J_Psi=False, no_Tamu=False):
    # Set style
    style_cfg = PLOT_STYLE['compare_allD']
    SetGlobalStyle(
        padleftmargin=style_cfg['padleftmargin'],
        padrightmargin=style_cfg['padrightmargin'],
        padbottommargin=style_cfg['padbottommargin'],
        padtopmargin=style_cfg['padtopmargin'],
        titleoffsety=style_cfg['titleoffsety'],
        titleoffsetx=style_cfg['titleoffsetx'],
        palette=get_root_constant(PLOT_STYLE['global']['palette']),
        titlesize=PLOT_STYLE['global']['titlesize'],
        labelsize=style_cfg['labelsize'],
        maxdigits=PLOT_STYLE['global']['maxdigits']
    )

    # Load data
    d0_hists = load_hists(DATA_PATHS['d0'], MARKERS['d0'], color_d0)
    lc_hists = load_hists(DATA_PATHS['lc'], MARKERS['lc'], color_lc)
    d0_hists[0].SetMarkerSize(MARKERS['size']['default'])
    lc_hists[0].SetMarkerSize(MARKERS['size']['default'])

    # Ds and Dplus
    colors_allD = {
        'ds': get_root_constant(COLORS['ds']),
        'dplus': get_root_constant(COLORS['dplus']),
        'jpsi': get_root_constant(COLORS['jpsi'])
    }
    markers_allD = {
        'ds': get_root_constant(MARKERS['ds']),
        'dplus': get_root_constant(MARKERS['dplus']),
        'jpsi': get_root_constant(MARKERS['jpsi'])
    }

    ds_hists = load_hists(DATA_PATHS['ds'], markers_allD['ds'], colors_allD['ds'], size=3)
    dplus_hists = load_hists(DATA_PATHS['dplus'], markers_allD['dplus'], colors_allD['dplus'], size=3)

    # Pi data
    df_pi = preprocess_data(
        [DATA_PATHS['pi']], 
        header=12, 
        get_source_data=True, 
        columns=["PT [GEV]", "CH_PIONS_v2_3040", "stat +", "sys +"]
    )
    graph_pi_stat = fill_graph(df_pi, columns=["PT [GEV]", "CH_PIONS_v2_3040", "stat +"])
    graph_pi_syst = fill_graph(df_pi, columns=["PT [GEV]", "CH_PIONS_v2_3040", "sys +"])
    
    target_bins = list(df_pi['PT [GEV] LOW'])
    target_bins.append(df_pi['PT [GEV] HIGH'].values[-1])
    scale_x_errors(graph_pi_syst, scale_factor=0.6, target_bins=target_bins)
    
    SetObjectStyle(graph_pi_stat, color=get_root_constant(COLORS['pi']), 
                   markerstyle=get_root_constant(MARKERS['pi']), linewidth=2, markersize=3)
    SetObjectStyle(graph_pi_syst, color=get_root_constant(COLORS['pi']), linewidth=2, fillstyle=0)

    # TAMU models
    models_path = DATA_PATHS['models']
    tamu_files = {
        'lc_high': f'{models_path}/arxivv1905.09216-tamu/lc-up.dat',
        'lc_low': f'{models_path}/arxivv1905.09216-tamu/lc-low.dat',
        'd0_high': f'{models_path}/arxivv1905.09216-tamu/d0-up.dat',
        'd0_low': f'{models_path}/arxivv1905.09216-tamu/d0-low.dat',
        'ds': f'{models_path}/PromptDs_TAMU_v2_5TeV_3050.txt',
        'jpsi': f'{models_path}/tamu_jpsi_3050.txt'
    }

    # Plot canvas
    canvas = ROOT.TCanvas("LcVsAllD_wTamu", "LcVsAllD_wTamu", 
                          *config['general']['canvas_size']['compare_allD'])
    canvas.SetLogx()
    frame = canvas.DrawFrame(0.455, -0.01, 27.7, 0.43,
                            ';#it{p}_{T} (GeV/#it{c});#it{v}_{2}')
    frame.GetYaxis().SetDecimals()
    frame.GetXaxis().SetNoExponent(True)
    frame.GetXaxis().SetMoreLogLabels()
    hide_some_log_labels(frame)

    # Draw elements
    graph_pi_stat.Draw('same epz')
    graph_pi_syst.Draw('same e2z')

    # Draw TAMU bands if enabled
    if not no_Tamu:
        lc_high_x, lc_high_y = preprocess(tamu_files['lc_high'], sep=",")
        lc_low_x, lc_low_y = preprocess(tamu_files['lc_low'], sep=",")
        d0_high_x, d0_high_y = preprocess(tamu_files['d0_high'], sep=",")
        d0_low_x, d0_low_y = preprocess(tamu_files['d0_low'], sep=",")
        ds_x, ds_low_y, ds_high_y = preprocess(tamu_files['ds'], catania=True, sep=" ", header=0)
        
        polyline_lc = get_band(lc_low_x, lc_high_x, lc_low_y, lc_high_y, color_lc)
        polyline_d0 = get_band(d0_low_x, d0_high_x, d0_low_y, d0_high_y, color_d0)
        polyline_ds = get_band(ds_x, ds_x, ds_low_y, ds_high_y, colors_allD["ds"])
        
        polyline_lc.SetFillColorAlpha(get_root_constant(MODEL_CFG['colors']['tamu']['base']), 0.5)
        polyline_d0.SetFillColorAlpha(get_root_constant(MODEL_CFG['colors']['tamu']['base']), 0.4)
        polyline_ds.SetFillColorAlpha(colors_allD["ds"], 0.5)
        
        polyline_lc.Draw('F')
        polyline_d0.Draw('F')
        polyline_ds.Draw('F')

        if not no_J_Psi:
            jpsi_x, jpsi_y = preprocess(tamu_files['jpsi'], sep=" ", header=0)
            graph_jpsi = ROOT.TGraph(len(jpsi_x), array.array('d', jpsi_x), array.array('d', jpsi_y))
            SetObjectStyle(graph_jpsi, color=colors_allD["jpsi"], linewidth=2)
            graph_jpsi.Draw('same l')

    # Draw meson data
    dplus_hists[0].Draw('same epz')
    dplus_hists[1].Draw('same e2z')
    ds_hists[0].Draw('same epz')
    ds_hists[1].Draw('same e2z')
    lc_hists[0].Draw('same epz')
    lc_hists[1].Draw('same e2z')
    d0_hists[0].Draw('same epz')
    d0_hists[1].Draw('same e2z')

    # Add labels and legend
    lat_large, lat_mid, latLabel = get_latex()
    latLabel.SetTextSize(0.045)
    legend_size = latLabel.GetTextSize()

    # Main legend
    legend = TLegend(0.18, 0.70, 0.80, 0.82)
    legend.SetTextFont(42)
    legend.SetTextSize(legend_size)
    legend.SetNColumns(2)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.AddEntry(lc_hists[0],  "Prompt #Lambda^{+}_{c}", "p")
    legend.AddEntry(dplus_hists[0], "Prompt D^{+}", "p")
    legend.AddEntry(d0_hists[0],   "Prompt D^{0}","p")
    legend.AddEntry(ds_hists[0],   "Prompt D^{+}_{s}", "p")
    legend.Draw("same")

    # Pi legend
    legend_r = TLegend(0.18, 0.52, 0.40, 0.7)
    legend_r.AddEntry(graph_pi_stat, "#pi^{#pm}", "p")
    legend_r.SetTextFont(42)
    legend_r.SetTextSize(legend_size)
    legend_r.SetBorderSize(0)
    legend_r.SetFillStyle(0)
    legend_r.Draw("same")

    # Annotations
    lat_large.DrawLatex(0.18, 0.90, 'ALICE, Pb#font[122]{-}Pb, 30#font[122]{-}50% ')
    latLabel.DrawLatex(0.18, 0.84, "#it{v}_{2} {SP, |#Delta#it{#eta}| > 1.3},  #sqrt{#it{s}_{NN}} = 5.36 TeV")
    
    if no_J_Psi and no_Tamu:
        latLabel.DrawLatex(0.18, 0.65, "#it{v}_{2} {2, |#Delta#it{#eta}| > 2},  #sqrt{#it{s}_{NN}} = 5.02 TeV, 30#font[122]{-}40%")

    # Save canvas
    out_name = f'{OUT_DIR}/LcVsAllD_wTamu.pdf'
    if no_J_Psi and no_Tamu:
        out_name = f'{OUT_DIR}/LcVsAllD_woJpsi_woTamu.pdf'
    elif no_Tamu:
        out_name = f'{OUT_DIR}/LcVsAllD_woTamu.pdf'
    elif no_J_Psi:
        out_name = out_name.replace('.pdf', '_woJpsi.pdf')
    
    canvas.Update()
    canvas.SaveAs(out_name)

# Compare with transport models
def compare_with_model(color_lc, color_d0):
    # Set global style
    style_cfg = PLOT_STYLE['global']
    SetGlobalStyle(
        padleftmargin=style_cfg['padleftmargin'],
        padrightmargin=style_cfg['padrightmargin'],
        padbottommargin=style_cfg['padbottommargin'],
        padtopmargin=style_cfg['padtopmargin'],
        titleoffset=style_cfg['titleoffset'],
        titleoffsety=style_cfg['titleoffsety'],
        palette=get_root_constant(style_cfg['palette']),
        titlesize=style_cfg['titlesize'],
        labelsize=style_cfg['labelsize'],
        maxdigits=style_cfg['maxdigits'],
        labelfont=style_cfg['labelfont'],
        titlefont=style_cfg['titlefont']
    )

    # Load data
    d0_hists = load_hists(DATA_PATHS['d0'], MARKERS['d0'], color_d0-9, size=MARKERS['size']['small'])
    lc_hists = load_hists(DATA_PATHS['lc'], MARKERS['lc'], color_lc, size=MARKERS['size']['small'])

    # Model paths
    model_files = {
        'tamu_lc_high': f'{DATA_PATHS["models"]}/arxivv1905.09216-tamu/lc-up.dat',
        'tamu_lc_low': f'{DATA_PATHS["models"]}/arxivv1905.09216-tamu/lc-low.dat',
        'tamu_d0_high': f'{DATA_PATHS["models"]}/arxivv1905.09216-tamu/d0-up.dat',
        'tamu_d0_low': f'{DATA_PATHS["models"]}/arxivv1905.09216-tamu/d0-low.dat',
        'catania_lc': f'{DATA_PATHS["models"]}/Fwd_ Predictions for LambdaC elliptic flow/v2_Lc_502_3050_Catania_band.dat',
        'catania_d0': f'{DATA_PATHS["models"]}/Fwd_ Predictions for LambdaC elliptic flow/v2_D0_502_3050_Catania_band.dat',
        'langevin_lc': f'{DATA_PATHS["models"]}/langevin-d4-results-to-pxy25.3.22/Lcv2fnwsnlo30-50.dat',
        'langevin_d0': f'{DATA_PATHS["models"]}/langevin-d4-results-to-pxy25.3.22/D0v2fnwsnlo30-50.dat',
        'powlang': f'{DATA_PATHS["models"]}/POWLANG-v2-PbPb3050.txt',
        'lbt_d0': f'{DATA_PATHS["models"]}/v2_PbPb5360-LBT-PNP/v2_D_30-50.dat',
        'lbt_lc': f'{DATA_PATHS["models"]}/v2_PbPb5360-LBT-PNP/v2_lambdac_60-80.dat',
        'epos4hq_d0': f'{DATA_PATHS["models"]}/epos4hq/v2pt_D0_PbPb5.02TeV_30-50.dat',
        'epos4hq_lc': f'{DATA_PATHS["models"]}/epos4hq/v2pt_Lambdac_PbPb5.02TeV_30-50.dat'
    }

    # Load model data
    tamu_lc_high_x, tamu_lc_high_y = preprocess(model_files['tamu_lc_high'], sep=",")
    tamu_lc_low_x, tamu_lc_low_y = preprocess(model_files['tamu_lc_low'], sep=",")
    tamu_d0_high_x, tamu_d0_high_y = preprocess(model_files['tamu_d0_high'], sep=",")
    tamu_d0_low_x, tamu_d0_low_y = preprocess(model_files['tamu_d0_low'], sep=",")
    
    d0_catania_x, d0_catania_low_y, d0_catania_high_y = preprocess(model_files['catania_d0'], catania=True, sep=" ")
    lc_catania_x, lc_catania_low_y, lc_catania_high_y = preprocess(model_files['catania_lc'], catania=True, sep=" ")
    
    lc_langevin_x, lc_langevin_y = preprocess(model_files['langevin_lc'], sep="        ")
    d0_langevin_x, d0_langevin_y = preprocess(model_files['langevin_d0'], sep='        ')

    # POWLANG data
    HTL_D0 = read_txt(model_files['powlang'], header=1, sep=' ', nrows=18)
    HTL_L_c = read_txt(model_files['powlang'], header=61, sep=' ', nrows=18)
    latQCD_D0 = read_txt(model_files['powlang'], header=81, sep=' ', nrows=18)
    latQCD_L_c = read_txt(model_files['powlang'], header=141, sep=' ', nrows=18)
    
    # LBT and EPOS4HQ
    d0_lbt = read_txt(model_files['lbt_d0'], sep='\t')
    lc_lbt = read_txt(model_files['lbt_lc'], sep='\t')
    d0_epos4hq = read_txt(model_files['epos4hq_d0'], sep=' ', header=0)
    lc_epos4hq = read_txt(model_files['epos4hq_lc'], sep=' ', header=0)

    # Create bands
    polyline_lc_tamu = get_band(tamu_lc_low_x, tamu_lc_high_x, tamu_lc_low_y, tamu_lc_high_y, color_lc)
    polyline_d0_tamu = get_band(tamu_d0_low_x, tamu_d0_high_x, tamu_d0_low_y, tamu_d0_high_y, color_d0)
    polyline_lc_catania = get_band(lc_catania_x, lc_catania_x, lc_catania_low_y, lc_catania_high_y, color_lc)
    polyline_d0_catania = get_band(d0_catania_x, d0_catania_x, d0_catania_low_y, d0_catania_high_y, color_d0)

    # Style bands
    polyline_lc_tamu.SetFillColorAlpha(get_root_constant(MODEL_CFG['colors']['tamu']['base']), 0.5)
    polyline_d0_tamu.SetFillColorAlpha(get_root_constant(MODEL_CFG['colors']['tamu']['base']), 0.5)
    polyline_lc_catania.SetFillColorAlpha(get_root_constant(MODEL_CFG['colors']['catania']['base']), 0.6)
    polyline_d0_catania.SetFillColorAlpha(get_root_constant(MODEL_CFG['colors']['catania']['base']), 0.6)

    # Create model graphs
    def create_model_graph(x, y, model_name):
        graph = ROOT.TGraph(len(x), array.array('d', x), array.array('d', y))
        
        graph.SetLineColor(ROOT.TColor.GetColorTransparent(get_root_constant(MODEL_CFG['colors'][model_name]['base'])
                                        , MODEL_CFG['colors'][model_name]['alpha']))
        graph.SetLineWidth(3)
        graph.SetLineStyle(MODEL_CFG['styles'][model_name])
        return graph

    graph_lc_langevin = create_model_graph(lc_langevin_x, lc_langevin_y, 'langevin')
    graph_d0_langevin = create_model_graph(d0_langevin_x, d0_langevin_y, 'langevin')
    graph_lc_htl = create_model_graph(HTL_L_c.iloc[:,0], HTL_L_c.iloc[:,1], 'htl')
    graph_d0_htl = create_model_graph(HTL_D0.iloc[:,0], HTL_D0.iloc[:,1], 'htl')
    graph_lc_lat = create_model_graph(latQCD_L_c.iloc[:,0], latQCD_L_c.iloc[:,1], 'latQCD')
    graph_d0_lat = create_model_graph(latQCD_D0.iloc[:,0], latQCD_D0.iloc[:,1], 'latQCD')
    graph_lc_lbt = create_model_graph(lc_lbt.iloc[:,0], lc_lbt.iloc[:,1], 'lbt')
    graph_d0_lbt = create_model_graph(d0_lbt.iloc[:,0], d0_lbt.iloc[:,1], 'lbt')
    graph_lc_epos4hq = create_model_graph(lc_epos4hq.iloc[:,0], lc_epos4hq.iloc[:,1], 'epos4hq')
    graph_d0_epos4hq = create_model_graph(d0_epos4hq.iloc[:,0], d0_epos4hq.iloc[:,1], 'epos4hq')

    # Create canvas
    canvas = ROOT.TCanvas("c1", "Uniform Style Canvas", *config['general']['canvas_size']['compare_model'])
    upper = canvas.cd(1)
    upper.Divide(2, 1, 0, 0)

    # Left pad (D0)
    left_pad = upper.cd(1)
    frame = left_pad.DrawFrame(0., -0.009, 24.9, 0.31,
                            ';#it{p}_{T} (GeV/#it{c}) ;#it{v}_{2}')
    frame.GetYaxis().SetDecimals()
    frame.GetXaxis().SetTickLength(0.036)

    # Draw D0 models and data
    polyline_d0_tamu.Draw("F z")
    polyline_d0_catania.Draw("F z")
    graph_d0_langevin.Draw('same')
    graph_d0_htl.Draw('same')
    graph_d0_lat.Draw('same')
    graph_d0_epos4hq.Draw('same')
    graph_d0_lbt.Draw('same')
    d0_hists[0].Draw('same epz')
    d0_hists[1].Draw('same e2z')

    # D0 legend
    legend = TLegend(0.40, 0.511, 0.90, 0.881)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.SetNColumns(2)
    legend.AddEntry(polyline_d0_tamu, "TAMU", "F")
    legend.AddEntry(polyline_d0_catania, "Catania", "F")
    legend.AddEntry(graph_d0_langevin, "Langevin", "l")
    legend.AddEntry(graph_d0_lbt, "LBT-PNP", "l")
    legend.AddEntry(graph_d0_lat, "POWLANG lQCD", "l")
    legend.AddEntry("", "", "")
    legend.AddEntry(graph_d0_htl, "POWLANG HTL", "l")
    legend.AddEntry("", "", "")
    legend.AddEntry(graph_d0_epos4hq, "EPOS4HQ", "l")
    legend.SetBorderSize(0)
    legend.Draw("same z")

    # D0 annotations
    lat_large, lat_mid, latLabel = get_latex()
    lat_large.SetTextSize(0.06)
    lat_mid.SetTextSize(0.055)
    latLabel.SetTextSize(0.05)
    
    lat_large.DrawLatex(0.20, 0.911, 'ALICE')
    lat_mid.DrawLatex(0.20, 0.836, 'Pb#font[122]{-}Pb')
    lat_mid.DrawLatex(0.20, 0.761, '30#font[122]{-}50%')
    latLabel.DrawLatex(0.35, 0.911, "Transport models,  #sqrt{#it{s}_{NN}} = 5.02 TeV")

    # Right pad (Lc)
    right_pad = upper.cd(2)
    frame = right_pad.DrawFrame(0.5, -0.009, 24.9, 0.31,
                            ';#it{p}_{T} (GeV/#it{c});#it{v}_{2} {SP, |#Delta#it{#eta}| > 1.3}')
    frame.GetYaxis().SetTickLength(0.035)

    # Draw Lc models and data
    polyline_lc_tamu.Draw("F z")
    polyline_lc_catania.Draw("F z")
    graph_lc_langevin.Draw('same')
    graph_lc_htl.Draw('same')
    graph_lc_lat.Draw('same')
    graph_lc_lbt.Draw('same')
    graph_lc_epos4hq.Draw('same')
    lc_hists[0].Draw('same epz')
    lc_hists[1].Draw('same e2z')

    # Lc annotations
    latLabel.DrawLatex(0.22, 0.911, "#it{v}_{2} {SP, |#Delta#it{#eta}| > 1.3},  #sqrt{#it{s}_{NN}} = 5.36 TeV")

    # Lc legend
    legend_r = TLegend(0.205, 0.711, 0.90, 0.951)
    legend_r.SetTextSize(0.05)
    legend_r.SetTextFont(42)
    legend_r.SetFillColorAlpha(ROOT.kWhite, 0)
    legend_r.AddEntry(lc_hists[0], "Prompt #Lambda_{c}^{+}", "p")
    legend_r.AddEntry(d0_hists[0], "Prompt D^{0}", "p")
    legend_r.SetNColumns(2)
    legend_r.SetBorderSize(0)
    legend_r.Draw("same z")

    # Save canvas
    canvas.Draw()
    canvas.SaveAs(f'{OUT_DIR}/compare-model.pdf')

    # Debug output
    if DEBUG:
        outFile = TFile(f'{OUT_DIR}/compare-model.root', "recreate")
        outFile.cd()
        d0_hists[0].Write()
        d0_hists[1].Write()
        lc_hists[0].Write()
        lc_hists[1].Write()
        canvas.Write()
        outFile.Close()

# NCQ comparison (v2/nq vs pT/nq or kET/nq)
def compare_dataWmodel_ncq(color_lc, color_d0):
    # Set style
    style_cfg = PLOT_STYLE['global']
    SetGlobalStyle(
        padleftmargin=style_cfg['padleftmargin'],
        padrightmargin=style_cfg['padrightmargin'],
        padbottommargin=style_cfg['padbottommargin'],
        padtopmargin=style_cfg['padtopmargin'],
        titleoffset=style_cfg['titleoffset'],
        titleoffsety=style_cfg['titleoffsety'],
        palette=get_root_constant(style_cfg['palette']),
        titlesize=style_cfg['titlesize'],
        labelsize=style_cfg['labelsize'],
        maxdigits=style_cfg['maxdigits'],
        labelfont=style_cfg['labelfont'],
        titlefont=style_cfg['titlefont']
    )

    # Load light flavor data
    data_path_sp = '../input-data/light-flavor-data/HEPData-ins1672822-v1-csv-SP'
    df_ks = preprocess_data(
        [f'{data_path_sp}/Table73.csv'],
        header=12,
        get_source_data=True,
        compine_syst_stat=True,
        columns=["PT [GEV]", "K0S_v2_3040", "stat +", "sys +"]
    )
    df_lambda = preprocess_data(
        [f'{data_path_sp}/Table87.csv'],
        header=12,
        get_source_data=True,
        compine_syst_stat=True,
        columns=["PT [GEV]", "LAMBDA_v2_3040", "stat +", "sys +"]
    )

    # Preprocess NCQ data
    all_data = {"ks": df_ks, "lambda": df_lambda}
    all_data_scaled = preprocess_ncq(all_data, do_ket_nq=NCQ_CFG['do_ket_nq'], ismodel=False)
    
    # Columns for graph creation
    columns = ["pt/nq", "v2/nq", "Total Error/nq"] if not NCQ_CFG['do_ket_nq'] else ["kEt/nq", "v2/nq", "Total Error/nq"]
    
    # Create graphs
    h_ks = fill_graph(all_data_scaled['ks'], columns)
    h_lambda = fill_graph(all_data_scaled['lambda'], columns)
    
    SetObjectStyle(h_ks, color=get_root_constant(COLORS['ks']), 
                   markerstyle=get_root_constant(MARKERS['ks']), linewidth=1, markersize=1.5)
    SetObjectStyle(h_lambda, color=get_root_constant(COLORS['lambda']), 
                   markerstyle=get_root_constant(MARKERS['lambda']), linewidth=1, markersize=1.5)

    # Load model data
    model_files = {
        'tamu_lc_high': f'{DATA_PATHS["models"]}/arxivv1905.09216-tamu/lc-up.dat',
        'tamu_lc_low': f'{DATA_PATHS["models"]}/arxivv1905.09216-tamu/lc-low.dat',
        'tamu_d0_high': f'{DATA_PATHS["models"]}/arxivv1905.09216-tamu/d0-up.dat',
        'tamu_d0_low': f'{DATA_PATHS["models"]}/arxivv1905.09216-tamu/d0-low.dat',
        'catania_lc': f'{DATA_PATHS["models"]}/Fwd_ Predictions for LambdaC elliptic flow/v2_Lc_502_3050_Catania_band.dat',
        'catania_d0': f'{DATA_PATHS["models"]}/Fwd_ Predictions for LambdaC elliptic flow/v2_D0_502_3050_Catania_band.dat',
        'langevin_lc': f'{DATA_PATHS["models"]}/langevin-d4-results-to-pxy25.3.22/Lcv2fnwsnlo30-50.dat',
        'langevin_d0': f'{DATA_PATHS["models"]}/langevin-d4-results-to-pxy25.3.22/D0v2fnwsnlo30-50.dat',
        'powlang': f'{DATA_PATHS["models"]}/POWLANG-v2-PbPb3050.txt',
        'lbt_d0': f'{DATA_PATHS["models"]}/v2_PbPb5360-LBT-PNP/v2_D_30-50.dat',
        'lbt_lc': f'{DATA_PATHS["models"]}/v2_PbPb5360-LBT-PNP/v2_lambdac_60-80.dat',
        'epos4hq_d0': f'{DATA_PATHS["models"]}/epos4hq/v2pt_D0_PbPb5.02TeV_30-50.dat',
        'epos4hq_lc': f'{DATA_PATHS["models"]}/epos4hq/v2pt_Lambdac_PbPb5.02TeV_30-50.dat'
    }

    # Preprocess model data for NCQ
    tamu_lc_low = preprocess(model_files['tamu_lc_low'], sep=",", do_ncq=True)
    tamu_lc_high = preprocess(model_files['tamu_lc_high'], sep=",", do_ncq=True)
    tamu_d0_low = preprocess(model_files['tamu_d0_low'], sep=",", do_ncq=True)
    tamu_d0_high = preprocess(model_files['tamu_d0_high'], sep=",", do_ncq=True)
    catania_lc = preprocess(model_files['catania_lc'], sep=" ", do_ncq=True)
    catania_d0 = preprocess(model_files['catania_d0'], sep=" ", do_ncq=True)
    langevin_lc = preprocess(model_files['langevin_lc'], sep="        ", do_ncq=True)
    langevin_d0 = preprocess(model_files['langevin_d0'], sep="        ", do_ncq=True)

    # POWLANG/LBT/EPOS4HQ NCQ data
    HTL_D0 = read_txt(model_files['powlang'], header=1, sep=' ', nrows=18)
    HTL_L_c = read_txt(model_files['powlang'], header=61, sep=' ', nrows=18)
    latQCD_D0 = read_txt(model_files['powlang'], header=81, sep=' ', nrows=18)
    latQCD_L_c = read_txt(model_files['powlang'], header=141, sep=' ', nrows=18)
    d0_lbt = read_txt(model_files['lbt_d0'], sep='\t')
    lc_lbt = read_txt(model_files['lbt_lc'], sep='\t')
    d0_epos4hq = read_txt(model_files['epos4hq_d0'], sep=' ', header=0)
    lc_epos4hq = read_txt(model_files['epos4hq_lc'], sep=' ', header=0)

    # Create model data dict
    all_model_data = {
        "tamu": {"lc": [tamu_lc_low, tamu_lc_high], "d0": [tamu_d0_low, tamu_d0_high]},
        "catania": {"lc": [catania_lc], "d0": [catania_d0]},
        "langevin": {"lc": [langevin_lc], "d0": [langevin_d0]},
        "htl": {"lc": [HTL_L_c], "d0": [HTL_D0]},
        "latQCD": {"lc": [latQCD_L_c], "d0": [latQCD_D0]},
        "lbt": {"lc": [lc_lbt], "d0": [d0_lbt]},
        "epos4hq": {"lc": [lc_epos4hq], "d0": [d0_epos4hq]}
    }

    # Scale model data for NCQ
    all_model_data_scaled = preprocess_ncq(all_model_data, do_ket_nq=NCQ_CFG['do_ket_nq'], ismodel=True)

    # Create model bands
    tamu_band_lc = get_band(
        all_model_data_scaled["tamu"]["lc"][0].iloc[:,0],
        all_model_data_scaled["tamu"]["lc"][1].iloc[:,0],
        all_model_data_scaled["tamu"]["lc"][0].iloc[:,1],
        all_model_data_scaled["tamu"]["lc"][1].iloc[:,1],
        color=color_lc,
        xmin_lim=0.01
    )
    tamu_band_d0 = get_band(
        all_model_data_scaled["tamu"]["d0"][0].iloc[:,0],
        all_model_data_scaled["tamu"]["d0"][1].iloc[:,0],
        all_model_data_scaled["tamu"]["d0"][0].iloc[:,1],
        all_model_data_scaled["tamu"]["d0"][1].iloc[:,1],
        color=color_d0,
        xmin_lim=0.01
    )

    # Style bands
    # tamu_band_lc.SetFillColorAlpha(get_root_constant(MODEL_CFG['colors']['tamu']['base']), 0.4)
    # tamu_band_d0.SetFillColorAlpha(get_root_constant(MODEL_CFG['colors']['tamu']['base']), 0.4)
    tamu_band_lc.SetFillColorAlpha(get_root_constant(color_lc), 0.4)
    tamu_band_d0.SetFillColorAlpha(get_root_constant(color_d0), 0.4)

    # Load Lc/D0 data
    d0_hists = load_hists(DATA_PATHS['d0'], MARKERS['d0'], color_d0)
    lc_hists = load_hists(DATA_PATHS['lc'], MARKERS['lc'], color_lc)
    
    # Preprocess NCQ for data
    d0_hists = preprocess_graph_ncq('d0', d0_hists, do_ket_nq=NCQ_CFG['do_ket_nq'], is_model=False)
    lc_hists = preprocess_graph_ncq('lc', lc_hists, do_ket_nq=NCQ_CFG['do_ket_nq'], is_model=False)
    scale_x_errors(d0_hists[1], target_graph=d0_hists[0], scale_factor=0.6)
    scale_x_errors(lc_hists[1], target_graph=lc_hists[0], scale_factor=0.6)
    
    SetObjectStyle(d0_hists[0], color=color_d0, markerstyle=get_root_constant(MARKERS['d0']), 
                   linewidth=2, markersize=1.5)
    SetObjectStyle(d0_hists[1], color=color_d0, linewidth=2, fillstyle=0)
    SetObjectStyle(lc_hists[0], color=color_lc, markerstyle=get_root_constant(MARKERS['lc']), 
                   linewidth=2, markersize=1.5)
    SetObjectStyle(lc_hists[1], color=color_lc, linewidth=2, fillstyle=0)

    # Create canvas
    canvas = ROOT.TCanvas("canvas", "Canvas", *config['general']['canvas_size']['ncq'])
    canvas.Divide(2, 1, 0, 0)

    # X axis config
    x_label = NCQ_CFG['x_label_pt'] if not NCQ_CFG['do_ket_nq'] else NCQ_CFG['x_label_ket']
    x_min = NCQ_CFG['x_min']
    x_max = NCQ_CFG['x_max_pt'] if not NCQ_CFG['do_ket_nq'] else NCQ_CFG['x_max_ket']
    y_max = NCQ_CFG['y_max']
    y_label = NCQ_CFG['y_label']

    # Left pad (light flavor + Lc/D0)
    pad_left = canvas.cd(1)
    frame = pad_left.DrawFrame(x_min, 0.0, x_max, y_max, f';{x_label} (GeV/#it{{c}})  ;{y_label}')
    frame.GetYaxis().SetDecimals()
    frame.GetXaxis().SetTickLength(0.035)
    frame.GetYaxis().SetTickLength(0.026)

    # Draw data
    d0_hists[0].Draw('same epz')
    d0_hists[1].Draw('same e2z')
    lc_hists[0].Draw('same epz')
    lc_hists[1].Draw('same e2z')
    h_ks.Draw('same epz')
    h_lambda.Draw('same epz')

    # Left pad legend
    lat_large, lat_mid, latLabel = get_latex()
    lat_large.SetTextSize(0.055)
    latLabel.SetTextSize(0.045)
    
    legend = TLegend(0.33, 0.62, 0.76, 0.94)
    legend.SetTextSize(0.045)
    legend.SetTextFont(42)
    legend.SetNColumns(2)
    legend.AddEntry(h_lambda, "#Lambda(#bar{#Lambda})", "p")
    legend.AddEntry(h_ks, "K_{S}^{0}", "p")
    legend.SetBorderSize(0)
    legend.Draw("same")

    # Left pad annotations
    lat_large.DrawLatex(0.20, 0.92, 'ALICE, Pb#font[122]{-}Pb')
    latLabel.DrawLatex(0.26, 0.86, "#it{v}_{2} {2, |#Delta#it{#eta}| > 2},  #sqrt{#it{s}_{NN}} = 5.02 TeV, 30#font[122]{-}40%")

    # Right pad (models + Lc/D0)
    pad_mid = canvas.cd(2)
    frame = pad_mid.DrawFrame(x_min, 0.0, x_max, y_max, f';{x_label} (GeV/#it{{c}})  ;{y_label}')

    # Draw models and data
    tamu_band_lc.Draw('F')
    tamu_band_d0.Draw('F')
    d0_hists[0].Draw('same epz')
    d0_hists[1].Draw('same e2z')
    lc_hists[0].Draw('same epz')
    lc_hists[1].Draw('same e2z')

    # Right pad legend
    legend_m = TLegend(0.20, 0.62, 0.75, 0.91)
    legend_m.SetFillStyle(0)
    legend_m.SetBorderSize(0)
    legend_m.SetNColumns(2)
    legend_m.SetTextSize(0.045)
    legend_m.SetTextFont(42)
    legend_m.AddEntry(lc_hists[0], "Prompt #Lambda_{c}^{+}", "p")
    legend_m.AddEntry(d0_hists[0], "Prompt D^{0}", "p")
    legend_m.AddEntry("", "", "")
    legend_m.AddEntry("", "", "")
    legend_m.AddEntry(tamu_band_lc, "TAMU Prompt #Lambda_{c}^{+}", "F")
    legend_m.AddEntry("", "", "")
    legend_m.AddEntry(tamu_band_d0, "TAMU Prompt D^{0}", "F")
    legend_m.Draw("same")

    # Right pad annotations
    latLabel.DrawLatex(0.12, 0.92, "#it{v}_{2} {SP, |#Delta#it{#eta}| > 1.3},  #sqrt{#it{s}_{NN}} = 5.36 TeV, 30#font[122]{-}50%")
    latLabel.DrawLatex(0.12, 0.79, "Transport models,  #sqrt{#it{s}_{NN}} = 5.02 TeV, 30#font[122]{-}50%")

    # Save canvas
    canvas.Update()
    canvas.SaveAs(f'{OUT_DIR}/ncq.pdf')

    # Debug output
    if DEBUG:
        outFile = TFile(f"{OUT_DIR}/ncq.root", "recreate")
        outFile.cd()
        d0_hists[0].Write()
        d0_hists[1].Write()
        lc_hists[0].Write()
        lc_hists[1].Write()
        canvas.Write()
        outFile.Close()

# Performance plots (invariant mass fit, v2 vs fraction, cut variation)
def plot_performance():
    # Set style
    SetGlobalStyle(
        padleftmargin=0.15,
        padrightmargin=0.03,
        padbottommargin=0.12,
        padtopmargin=0.05,
        opttitle=1,
        titleoffsety=1.6,
        labelsize=0.05,
        titlesize=0.05,
        labeloffset=0.01,
        titleoffset=1.2,
        labelfont=42,
        titlefont=42
    )

    # Config
    pt_low = PERF_CFG['pt_low']
    pt_high = PERF_CFG['pt_high']
    version = PERF_CFG['data_version']
    indir = DATA_PATHS['performance']
    ry_file = f'{indir}/raw_yields_uncorr_{version}.root'
    cutvar_file = f'{indir}/CutVarFrac_corr.root'
    linear_fit_file = f'{indir}/V2VsFrac_combined.root'

    # Axis ranges
    ranges = PERF_CFG['axis_ranges']['versions'][version]
    mass_xmin = PERF_CFG['axis_ranges']['mass_xmin']
    mass_xmax = PERF_CFG['axis_ranges']['mass_xmax']
    v2_xmin = PERF_CFG['axis_ranges']['v2_xmin']
    v2_xmax = PERF_CFG['axis_ranges']['v2_xmax']

    # Get invariant mass fit
    hInvMassD0, fMassTotD0, fMassBkgD0, hV2VsMassD0, fV2TotD0, fV2BkgD0 = GetInvMassHistAndFit(
        ry_file, pt_low, pt_high, 2
    )

    # Create canvas
    canvas, frames = GetCanvas4sub(
        'cDv2run3',
        mass_xmin, mass_xmax,
        ranges['mass_ymin'], ranges['mass_ymax'],
        ranges['v2_ymin'], ranges['v2_ymax'],
        f';#it{{M}}(pK#pi) (GeV/#it{{c}}^{{2}});Counts per {hInvMassD0.GetBinWidth(1)*1000:.0f} MeV/#it{{c}}^{{2}}',
        ';#it{M}(pK#pi) (GeV/#it{c}^{2});#it{v}_{2}^{tot.} {SP, |#Delta#it{#eta}| > 1.3}'
    )

    # Style frames
    frames[0].GetYaxis().SetNoExponent(False)
    frames[0].GetXaxis().SetNdivisions(504)
    frames[2].GetXaxis().SetNdivisions(504)
    frames[2].GetYaxis().SetDecimals()
    frames[3].GetYaxis().SetDecimals()

    # Draw mass plot (top left)
    pad_top = canvas.cd(1)
    hInvMassD0.Draw('esame')
    fMassBkgD0.Draw('same')
    fMassTotD0.Draw('same')

    # Annotations
    latex = ROOT.TLatex()
    latex.SetTextFont(42)
    latex.SetTextSize(0.05)
    latexlarge = ROOT.TLatex()
    latexlarge.SetTextFont(42)
    latexlarge.SetTextSize(0.07)

    latexlarge.DrawLatexNDC(0.2, 0.85, 'ALICE')
    latex.DrawLatexNDC(0.2, 0.77, 'Pb#font[122]{-}Pb, 30#font[122]{-}50%, #sqrt{#it{s}_{NN}} = 5.36 TeV')
    latex.DrawLatexNDC(0.2, 0.69, '#Lambda_{c}^{+} #rightarrow pK^{#font[122]{-}}#pi^{+} and charge conj.')
    latex.DrawLatexNDC(0.2, 0.61, f'{pt_low} < #it{{p}}_{{T}} < {pt_high} GeV/#it{{c}}')

    # Legend
    legD = GetLegend(xmax=0.5, ncolumns=1, ymin=0.18, ymax=0.32, textsize=0.055, xmin=0.22)
    legD.SetTextSize(0.045)
    legD.AddEntry(fV2TotD0, 'Total fit function', 'l')
    legD.AddEntry(fV2BkgD0, 'Combinatorial background', 'l')
    legD.Draw()

    # Draw v2 vs mass (bottom left)
    pad_bottom = canvas.cd(3)
    hV2VsMassD0.Draw('esame')
    fV2BkgD0.Draw('same')
    fV2TotD0.Draw('same')

    # BDT score annotations
    latexdetail2 = ROOT.TLatex()
    latexdetail2.SetTextFont(42)
    latexdetail2.SetTextSize(0.05)
    
    bdt_annotations = {
        "00": ["0 < BDT score to be prompt < 0.03", "#it{v}_{2}^{sig.} = 0.012 #pm 0.051"],
        "01": ["0.03 < BDT score to be prompt < 0.8", "#it{v}_{2}^{sig.} = 0.146 #pm 0.051"],
        "02": ["0.92 < BDT score to be prompt < 1", "#it{v}_{2}^{sig.} = 0.173 #pm 0.026"]
    }
    
    latexdetail2.DrawLatexNDC(0.2, 0.85, bdt_annotations[version][0])
    latexdetail2.DrawLatexNDC(0.2, 0.77, bdt_annotations[version][1])

    # Draw v2 vs fraction fit (bottom right)
    gv2D0, hv2D0, tf1D0 = GetV2HistAndFit(linear_fit_file, f'pt_50_60', pt_low, pt_high, 2)
    canvas.cd(4)
    frames[3].GetYaxis().SetRangeUser(-0.05, 0.28)
    
    hv2D0.SetFillColorAlpha(ROOT.kAzure+4, 0.4)
    hv2D0.Draw('same')
    gv2D0.Draw('pez same')
    tf1D0.Draw('same')

    # Chi2 annotation
    latexdetail = ROOT.TLatex()
    latexdetail.SetTextFont(42)
    latexdetail.SetTextSize(0.045)
    latexdetail.DrawLatexNDC(0.2, 0.2, f'#it{{#chi}}^{{2}}/ndf = {tf1D0.GetChisquare()/tf1D0.GetNDF():.2f}')

    # V2 fit legend
    legDistr = ROOT.TLegend(0.45, 0.75, 0.75, 0.90)
    legDistr.SetFillStyle(0)
    legDistr.SetBorderSize(0)
    legDistr.SetTextSize(0.045)
    legDistr.AddEntry(tf1D0, 'Linear fit', 'l')
    legDistr.AddEntry(hv2D0, '68% confidence level', 'f')
    legDistr.Draw()

    # Draw cut variation (top right)
    infile = ROOT.TFile.Open(cutvar_file)
    cutvar_dir = infile.Get(f'pt{pt_low}.0_{pt_high}.0')
    
    yield_fd = cutvar_dir.Get(f'hRawYieldFDVsCut_pT{pt_low}.0_{pt_high}.0')
    yield_p = cutvar_dir.Get(f'hRawYieldPromptVsCut_pT{pt_low}.0_{pt_high}.0')
    yield_tot = cutvar_dir.Get(f'hRawYieldsVsCutReSum_pT{pt_low}.0_{pt_high}.0')
    yield_data = cutvar_dir.Get(f'hRawYieldsVsCutPt_pT{pt_low}.0_{pt_high}.0')

    # Style cut variation plots
    SetObjectStyle(yield_data, linecolor=ROOT.kBlack, markercolor=ROOT.kBlack, markerstyle=ROOT.kFullCircle)
    SetObjectStyle(yield_p, color=ROOT.kRed+1, fillcolor=ROOT.kRed+1, fillalpha=0.3)
    SetObjectStyle(yield_fd, color=ROOT.kAzure+4, fillcolor=ROOT.kAzure+4, fillalpha=0.3)
    SetObjectStyle(yield_tot, linecolor=ROOT.kGreen+2)

    # Draw cut variation
    canvas.cd(2)
    yield_fd.DrawCopy('histsame')
    yield_p.DrawCopy('histsame')
    yield_tot.Draw('same')
    yield_data.Draw('same p')

    # Cut variation legend
    legDistr_cutvar = ROOT.TLegend(0.45, 0.5, 0.8, 0.90)
    legDistr_cutvar.SetFillStyle(0)
    legDistr_cutvar.SetBorderSize(0)
    legDistr_cutvar.SetTextSize(0.045)
    legDistr_cutvar.AddEntry(yield_p, 'Prompt #Lambda_{c}^{+}', 'f')
    legDistr_cutvar.AddEntry(yield_fd, 'Non-prompt #Lambda_{c}^{+}', 'f')
    legDistr_cutvar.AddEntry(yield_data, 'Data', 'lpe')
    legDistr_cutvar.AddEntry(yield_tot, 'Total', 'l')
    legDistr_cutvar.Draw('same')

    # Save canvas
    canvas.SaveAs(f'{OUT_DIR}/performance.pdf')
    infile.Close()

# Canvas helper function
def GetCanvas4sub(name, xmins, xmaxs, ymins_mass, ymaxs_mass, ymins_v2, ymaxs_v2, axisnametop, axisnamebottom):
    canvas = ROOT.TCanvas(name, name, *config['general']['canvas_size']['performance'])
    canvas.Divide(2, 2)
    frames = []

    for i in range(4):
        canvas.cd(i + 1)
        if i == 0:
            print(xmins, ymins_mass, xmaxs, ymaxs_mass, axisnametop)
            frame = canvas.DrawFrame(xmins, ymins_mass, xmaxs, ymaxs_mass, axisnametop)
        elif i == 1:
            frame = canvas.DrawFrame(0.5, 0, 20.5, 14900, ';BDT-based selection; raw yield')
        elif i == 2:
            frame = canvas.DrawFrame(xmins, ymins_v2, xmaxs, ymaxs_v2, axisnamebottom)
        elif i == 3:
            frame = canvas.DrawFrame(0, 0, 1.05, 0.3, ';Non-prompt fraction; #it{v}_{2}^{sig.}{SP, |#Delta#it{#eta}| > 1.3}')
        
        frame.SetTitle("")
        frames.append(frame)

    return canvas, frames


# Main execution
if __name__ == "__main__":
    # Default colors
    color_lc = get_root_constant(COLORS['lc']['default'])
    color_d0 = get_root_constant(COLORS['d0']['default'])

    # Run comparison plots
    compare_allD(color_lc=color_lc, color_d0=color_d0, no_Tamu=True, no_J_Psi=True)
    compare_dataWmodel_ncq(color_lc=color_lc, color_d0=color_d0)

    # Re-run model comparison with black colors
    color_lc_black = get_root_constant(COLORS['lc']['black'])
    color_d0_black = get_root_constant(COLORS['d0']['black'])
    # compare_with_model(color_lc=color_lc_black, color_d0=color_d0_black)
    compare_with_model(color_lc=color_lc, color_d0=color_d0)

    # Performance plots
    plot_performance()

    # Convert PDFs (optional)
    pdf_paths = [f'{OUT_DIR}/{f}' for f in ['ncq.pdf', 'compare-model.pdf', 'performance.pdf', 'LcVsAllD_woJpsi_woTamu.pdf']]
    # pdf2eps_imagemagick(pdf_paths, target_format='png')

    print("Done!")