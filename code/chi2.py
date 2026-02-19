"""
Chi-square calculation for Lc/D0/Ds v2 model comparison
"""
import ROOT
import yaml
import numpy as np
import array
from plot_untils import (
    preprocess, merge_asymmetric_errors, model_chi2
)

# Set ROOT batch mode to avoid GUI issues
ROOT.gROOT.SetBatch(True)

# ---------------------- Basic Helpers ----------------------
def load_config(config_path="plot_config.yaml"):
    """Load and validate config file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Basic validation
    required = ['data_paths']
    if not all(k in config for k in required):
        raise ValueError(f"Config missing keys: {required}")
    return config

def read_exp_data(file_path, particle):
    """Read experimental data from ROOT file"""
    f = ROOT.TFile.Open(file_path, "READ")
    if not f or f.IsZombie():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read experimental graphs
    data = {
        'stat': f.Get("gvn_prompt_stat"),
        'reso': f.Get("reso_syst"),
        'fit': f.Get("fit_syst"),
        'fd': f.Get("fd_syst")
    }
    # Merge total errors (stat + reso)
    data['total'] = merge_asymmetric_errors(data['stat'], data['reso'])
    f.Close()
    return data

# ---------------------- Interpolation (Graph-only) ----------------------
def get_interp_model(data_target, models_x, models_y, name='', no_ptshift=False):
    """
    Interpolate model to experimental graph points
    data_target: Experimental graph (TGraphAsymmErrors)
    models_x: Model x-values (numpy array)
    models_y: List of callable interpolation functions (1=value, 2=error band)
    no_ptshift: Use bin centers for interpolation
    """
    # Input validation
    if not isinstance(data_target, ROOT.TGraphAsymmErrors):
        raise TypeError("data_target must be TGraphAsymmErrors")
    if len(models_y) not in [1, 2]:
        raise ValueError("models_y needs 1 or 2 elements")
    
    n_points = data_target.GetN()
    x_target = np.array([data_target.GetX()[i] for i in range(n_points)])
    y_model = np.zeros(n_points)
    y_err_model = np.zeros(n_points)
    
    # Model x range
    x_max = np.max(models_x)
    # Interpolate to target points
    for i in range(n_points):
        # Calculate interpolation x point
        if no_ptshift and isinstance(data_target, ROOT.TGraphAsymmErrors):
            x_left = data_target.GetX()[i] - data_target.GetEXlow()[i]
            x_right = data_target.GetX()[i] + data_target.GetEXhigh()[i]
            pt_cent = (x_left + x_right) / 2
        else:
            pt_cent = x_target[i]
        
        # Skip points outside model range
        if not (pt_cent <= x_max):
            y_model[i] = 1e-10
            y_err_model[i] = 0
            continue
        
        # Calculate model value/error
        if len(models_y) == 1:
            y_model[i] = models_y[0](pt_cent)
            y_err_model[i] = 0
        else:
            val1, val2 = models_y[0](pt_cent), models_y[1](pt_cent)
            y_model[i] = np.mean([val1, val2])
            y_err_model[i] = np.abs(val1 - val2) / 2
    
    # Create output graph
    interp_graph = ROOT.TGraphAsymmErrors(n_points)
    interp_graph.SetNameTitle(name, f"Interpolated Model: {name}")
    
    # Populate graph
    for i in range(n_points):
        interp_graph.SetPoint(i, x_target[i], y_model[i])
        interp_graph.SetPointError(
            i,
            data_target.GetEXlow()[i],
            data_target.GetEXhigh()[i],
            y_err_model[i],
            y_err_model[i]
        )
    return interp_graph

# ---------------------- Main Chi2 Calculation ----------------------
def calculate_chi2_sys_correlation(debug=True, no_ptshift=False):
    """Calculate chi2 with systematic error correlation tests"""
    # 1. Load config and set paths
    config = load_config()
    DATA_PATHS = config['data_paths']
    
    # Model file paths (fixed key mapping for TAMU)
    model_files = {
        # TAMU models (d0/lc: low/high; ds: single)
        'tamu_lc': {'low': f'{DATA_PATHS["models"]}/arxivv1905.09216-tamu/lc-low.dat',
                    'high': f'{DATA_PATHS["models"]}/arxivv1905.09216-tamu/lc-up.dat'},
        'tamu_d0': {'low': f'{DATA_PATHS["models"]}/arxivv1905.09216-tamu/d0-low.dat',
                    'high': f'{DATA_PATHS["models"]}/arxivv1905.09216-tamu/d0-up.dat'},
        'tamu_ds': f'{DATA_PATHS["models"]}/arxivv1905.09216-tamu/PromptDs_TAMU_v2_5TeV_3050.txt',
        
        # Other models
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

    # 2. Read experimental data
    exp_data = {
        'd0': read_exp_data(DATA_PATHS['d0'], 'd0'),
        'ds': read_exp_data(DATA_PATHS['ds'], 'ds'),
        'lc': read_exp_data(DATA_PATHS['lc'], 'lc')
    }

    # 3. Interpolation function for models
    def interpolate_model(model_key, particle):
        """Interpolate single model to particle's experimental points"""
        data_target = exp_data[particle]['total']
        
        # TAMU models (special handling for d0/lc vs ds)
        if 'tamu' in model_key:
            if particle in ['d0', 'lc']:
                # d0/lc have low/high files
                x_low, y_low = preprocess(model_files[model_key]['low'], sep=',', do_interp=True)
                x_high, y_high = preprocess(model_files[model_key]['high'], sep=',', do_interp=True)
                return get_interp_model(data_target, x_low, [y_low, y_high], name=model_key, no_ptshift=no_ptshift)
            else:  # ds single file
                x, y_low, y_high = preprocess(model_files[model_key], header=0, sep=' ', do_interp=True, catania=True)
                return get_interp_model(data_target, x, [y_low, y_high], name=model_key, no_ptshift=no_ptshift)
        
        # Catania models (error band in single file)
        elif 'catania' in model_key:
            x, y_low, y_high = preprocess(model_files[model_key], sep=' ', do_interp=True, catania=True)
            return get_interp_model(data_target, x, [y_low, y_high], name=model_key, no_ptshift=no_ptshift)
        
        # POWLANG (HTL/latQCD)
        elif model_key in ['HTL_D0', 'HTL_Ds', 'HTL_Lc', 'latQCD_D0', 'latQCD_Ds', 'latQCD_Lc']:
            header_map = {'HTL_D0':1, 'HTL_Ds':41, 'HTL_Lc':61, 'latQCD_D0':81, 'latQCD_Ds':121, 'latQCD_Lc':141}
            x, y = preprocess(model_files['powlang'], header=header_map[model_key], sep=' ', nrows=18, do_interp=True)
            return get_interp_model(data_target, x, [y], name=model_key, no_ptshift=no_ptshift)
        
        # Other models (Langevin/LBT/EPOS4HQ)
        else:
            sep_map = {'langevin': '        ', 'lbt': '\t', 'epos4hq': ' '}
            header_map = {'langevin': None, 'lbt': None, 'epos4hq': 0}
            model_name = model_key.split('_')[0]
            header = header_map[model_name]
            sep = sep_map[model_name]
            if debug:
                print(f"Interpolating {model_key} with separator '{sep}' and header={header}")
            x, y = preprocess(model_files[model_key], header=header, sep=sep, do_interp=True)
            return get_interp_model(data_target, x, [y], name=model_key, no_ptshift=no_ptshift)

    # 4. Generate model graphs
    model_graphs = {
        'd0': [
            interpolate_model('tamu_d0', 'd0'),
            interpolate_model('catania_d0', 'd0'),
            interpolate_model('langevin_d0', 'd0'),
            interpolate_model('HTL_D0', 'd0'),
            interpolate_model('latQCD_D0', 'd0'),
            interpolate_model('lbt_d0', 'd0'),
            interpolate_model('epos4hq_d0', 'd0')
        ],
        'ds': [
            interpolate_model('tamu_ds', 'ds'),
            interpolate_model('HTL_Ds', 'ds'),
            interpolate_model('latQCD_Ds', 'ds')
        ],
        'lc': [
            interpolate_model('tamu_lc', 'lc'),
            interpolate_model('catania_lc', 'lc'),
            interpolate_model('langevin_lc', 'lc'),
            interpolate_model('HTL_Lc', 'lc'),
            interpolate_model('latQCD_Lc', 'lc'),
            interpolate_model('lbt_lc', 'lc'),
            interpolate_model('epos4hq_lc', 'lc')
        ]
    }

    # 5. Chi2 calculation for correlation scenarios
    def test_sys_correlation(data_graph, model_graph, stat_graph, sys_graphs, ndf):
        """Calculate chi2 for different sys error correlation scenarios"""
        reso, fit, fd = sys_graphs
        results = {}

        # Scenario 1: Baseline (reso correlated, fit/fd uncorrelated)
        _, _, chi2_ndf, _ = model_chi2(
            data_asymm=data_graph, h_model=model_graph, stat_err_data=stat_graph,
            sys_corr_components=[reso], sys_uncorr_components=[fit, fd],
            rho_sys_uncorr=np.eye(2), ndf=ndf
        )
        results['Baseline'] = chi2_ndf

        # Scenario 2: All uncorrelated
        _, _, chi2_ndf, _ = model_chi2(
            data_asymm=data_graph, h_model=model_graph, stat_err_data=stat_graph,
            sys_corr_components=[], sys_uncorr_components=[reso, fit, fd],
            rho_sys_uncorr=np.eye(3), ndf=ndf
        )
        results['All_Uncorrelated'] = chi2_ndf

        # Scenario 3: All correlated
        _, _, chi2_ndf, _ = model_chi2(
            data_asymm=data_graph, h_model=model_graph, stat_err_data=stat_graph,
            sys_corr_components=[reso, fit, fd], sys_uncorr_components=[],
            rho_sys_uncorr=None, ndf=ndf
        )
        results['All_Correlated'] = chi2_ndf

        # Scenario 4: Reso only
        _, _, chi2_ndf, _ = model_chi2(
            data_asymm=data_graph, h_model=model_graph, stat_err_data=stat_graph,
            sys_corr_components=[reso], sys_uncorr_components=[],
            rho_sys_uncorr=None, ndf=ndf
        )
        results['Reso_Only'] = chi2_ndf

        if debug:
            print(f"\nModel: {model_graph.GetName()} | NDF: {ndf}")
            for scen, val in results.items():
                print(f"  {scen:15} | chi2/ndf = {val:.4f}")
        
        return {model_graph.GetName(): results}

    # 6. Calculate chi2 for all particles/models
    chi2_results = {'D0': {}, 'Ds': {}, 'LambdaC': {}}
    ndf_map = {'d0':12, 'ds':8, 'lc':5}

    # D0
    for graph in model_graphs['d0']:
        chi2_results['D0'].update(test_sys_correlation(
            exp_data['d0']['total'], graph, exp_data['d0']['stat'],
            [exp_data['d0']['reso'], exp_data['d0']['fit'], exp_data['d0']['fd']], ndf_map['d0']
        ))

    # Ds
    for graph in model_graphs['ds']:
        chi2_results['Ds'].update(test_sys_correlation(
            exp_data['ds']['total'], graph, exp_data['ds']['stat'],
            [exp_data['ds']['reso'], exp_data['ds']['fit'], exp_data['ds']['fd']], ndf_map['ds']
        ))

    # LambdaC
    for graph in model_graphs['lc']:
        chi2_results['LambdaC'].update(test_sys_correlation(
            exp_data['lc']['total'], graph, exp_data['lc']['stat'],
            [exp_data['lc']['reso'], exp_data['lc']['fit'], exp_data['lc']['fd']], ndf_map['lc']
        ))

    # 7. Summary output
    if debug:
        out_name = f'{config["general"]["out_dir"]}/ch2.root'
        out_file = ROOT.TFile(out_name, "RECREATE")
        for particles in model_graphs.keys():
            for graph in model_graphs[particles]:
                graph.Write()
        out_file.Close()
        print("\n=== CHI-SQUARE SUMMARY ===")
        for p, models in chi2_results.items():
            print(f"\n{p}:")
            for m, res in models.items():
                print(f"  {m:15} | Baseline: {res['Baseline']:.4f} | All_Uncorr: {res['All_Uncorrelated']:.4f}")

    return chi2_results

import numpy as np

def save_chi2_results_to_txt(results, no_ptshift, output_path="chi2_results.txt"):
    """
    Save chi2 results dictionary to text file with readable format
    results: Chi2 results dict (particle → model → scenario → value)
    no_ptshift: Use bin centers for interpolation
    output_path: Path for output text file
    """
    # Open file in write mode
    with open(output_path, "w") as f:
        # Write header
        f.write("=== Chi-Square Calculation Results ===\n")
        f.write("Format: Particle → Model → Scenario: chi2/ndf\n\n")
        if no_ptshift:
            f.write("Note: Interpolation used bin centers (no_ptshift=True)\n\n")
        else:
            f.write("Note: Interpolation used shifted points (no_ptshift=False)\n\n")
        
        # Iterate through each particle (D0/Ds/LambdaC)
        for particle, models in results.items():
            f.write(f"[{particle}]\n")
            
            # Iterate through each model for the particle
            for model, scenarios in models.items():
                f.write(f"  {model}:\n")
                
                # Iterate through each correlation scenario
                for scenario, value in scenarios.items():
                    # Convert numpy float to regular float for readability
                    f.write(f"    {scenario}: {float(value):.6f}\n")
            
            f.write("\n")  # Add blank line between particles


# ---------------------- Run ----------------------
if __name__ == "__main__":
    # no_ptshift: Use bin centers for interpolation (True) vs. shifted points (False)
    no_ptshift = False
    results = calculate_chi2_sys_correlation(debug=False, no_ptshift=no_ptshift)
    save_chi2_results_to_txt(results, no_ptshift=no_ptshift, output_path="chi2_results.txt")