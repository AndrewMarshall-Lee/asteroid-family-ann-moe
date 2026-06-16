#!/usr/bin/env python3
# Builds the asteroid database with Gaia data.
# Author: Andrew Marshall-Lee

import os
import requests
import pandas as pd
import pdr
from astropy.table import Table
import numpy as np
import re
import glob
from bs4 import BeautifulSoup
import gzip
import shutil
from astropy.io import ascii
from scipy.stats import linregress
from csaps import csaps
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

PDS_BASE_URL = "https://sbnarchive.psi.edu/pds4/non_mission"
PDS_DATA_DIR = "data"
LOCAL_DATA_DIR = "data"

MPC_URL = "https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT"

# Updated data sources: name: (subdir, xml_filename, data_filename, table_key, format)
PDS_DATA_SOURCES = {
    "proper_orbits": ("ast.nesvorny.families_V2_0", "proper_catalog24.xml", "proper_catalog24.tab", "table", "tab"),
    "taxonomy": ("ast_taxonomy", "taxonomy10.xml", "taxonomy10.tab", "TABLE", "tab"),
    "colour": ("gbo.sdss-moc.phot", "sdssmocadr4.xml", "sdssmocadr4.tab", "table", "tab"),
    "albedo_mb": ("neowise_diameters_albedos_V2_0", "neowise_mainbelt.xml", "neowise_mainbelt.csv", "TABLE", "csv"),
    "albedo_ambos": ("neowise_diameters_albedos_V2_0", "neowise_ambos.xml", "neowise_ambos.csv", "TABLE", "csv"),
    "albedo_centaurs": ("neowise_diameters_albedos_V2_0", "neowise_centaurs.xml", "neowise_centaurs.csv", "TABLE", "csv"),
    "albedo_fixed_diameter_fits": ("neowise_diameters_albedos_V2_0", "neowise_fixed_diameter_fits.xml", "neowise_fixed_diameter_fits.csv", "TABLE", "csv"),
    "albedo_hildas": ("neowise_diameters_albedos_V2_0", "neowise_hildas.xml", "neowise_hildas.csv", "TABLE", "csv"),
    "albedo_irreg_sat": ("neowise_diameters_albedos_V2_0", "neowise_irreg_sat.xml", "neowise_irreg_sat.csv", "TABLE", "csv"),
    "albedo_jupiter_trojans": ("neowise_diameters_albedos_V2_0", "neowise_jupiter_trojans.xml", "neowise_jupiter_trojans.csv", "TABLE", "csv"),
    "albedo_neos": ("neowise_diameters_albedos_V2_0", "neowise_neos.xml", "neowise_neos.csv", "TABLE", "csv"),
    "spectra": ("gbo.ast.primass-l.spectra_V2_0", "primassl_visible_spectral_parameters.xml", "primassl_visible_spectral_parameters.csv", "TABLE", "csv"),
}

PDS_FAMILY_DATA = "ast.nesvorny.families_V2_0"


def download_file_if_needed(url, local_path, verbose=True):
    if not os.path.exists(local_path):
        if verbose:
            print(f"Downloading {os.path.basename(local_path)} from {url}")
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(response.content)
    else:
        if verbose:
            print(f"{os.path.basename(local_path)} already exists. Skipping download.")


def download_pds_dataset(subdir, xml_filename, data_filename):
    base_data_url = f"{PDS_BASE_URL}/{subdir}/{PDS_DATA_DIR}"
    xml_url = f"{base_data_url}/{xml_filename}"
    data_url = f"{base_data_url}/{data_filename}"

    xml_path = os.path.join(LOCAL_DATA_DIR, xml_filename)
    data_path = os.path.join(LOCAL_DATA_DIR, data_filename)

    download_file_if_needed(xml_url, xml_path)
    download_file_if_needed(data_url, data_path)

    return xml_path, data_path


def parse_mpcorb(filename):
    colspecs = [
        (0, 7), (8, 13), (14, 19), (20, 25), (26, 35), (37, 46), (48, 57),
        (59, 68), (70, 79), (80, 91), (92, 103), (105, 106), (107, 116),
        (117, 122), (123, 133), (134, 139), (140, 146), (147, 157),
        (158, 162), (166, 194)
    ]
    colnames = [
        "desig", "H", "G", "epoch", "M", "peri", "node", "incl", "e", "n", "a",
        "ref", "n_obs", "n_opp", "arc", "rms", "perts", "computer", "hex_flags", "name"
    ]

    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip() and line[:5].isdigit():
                first_data = i
                break

    df = pd.read_fwf(filename, colspecs=colspecs, names=colnames, skiprows=first_data)
    return df


def split_name(name):
    name = str(name).strip()
    match = re.match(r'\((\d+)\)\s+(.*)', name)
    if match:
        return pd.Series({'number': match.group(1), 'name_clean': match.group(2).strip()})
    else:
        return pd.Series({'number': 0, 'name_clean': name})


def create_df_base(
    mpc_url=MPC_URL,
    local_data_dir=LOCAL_DATA_DIR,
    output_path=os.path.join(LOCAL_DATA_DIR, 'df_base.csv')
):
    local_path = os.path.join(local_data_dir, 'MPCORB.DAT')
    download_file_if_needed(mpc_url, local_path)

    df_mpc = parse_mpcorb(local_path)
    df_base = df_mpc[['desig', 'name']].copy()
    df_base[['number', 'name_clean']] = df_base['name'].apply(split_name)
    df_base.to_csv(output_path, index=False)
    print(f"df_base saved to {output_path}")
    return df_base


def ensure_families_subdirs_downloaded():
    for subdir in ['families_2015', 'families_2024']:
        local_dir = os.path.join(PDS_DATA_DIR, subdir)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)
            url = f"{PDS_BASE_URL}/{PDS_FAMILY_DATA}/{PDS_DATA_DIR}/{subdir}/"
            print(f"Fetching file list from {url}")
            resp = requests.get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            links = [a['href'] for a in soup.find_all('a', href=True) if not a['href'].startswith('?')]
            for fname in links:
                if fname.endswith('/'):
                    continue
                file_url = url + fname
                local_path = os.path.join(local_dir, fname)
                download_file_if_needed(file_url, local_path, verbose=False)
        else:
            print(f"Family subdirectory {local_dir} already exists.")


def knn_bidirectional_impute(df, col_x, col_y, n_neighbors=5, flag_x_col=None, flag_y_col=None):
    """
    Perform bidirectional KNN imputation between two correlated columns.
    Also prints summary statistics showing how many values were imputed.
    Optional boolean flag columns can be added to record which rows were imputed.
    """
    df = df.copy()

    # Initialise any requested flag columns
    for flag_col in [flag_x_col, flag_y_col]:
        if flag_col is not None:
            if flag_col in df.columns:
                df[flag_col] = df[flag_col].fillna(False).astype(bool)
            else:
                df[flag_col] = False

    # Record initial availability
    initial_x = df[col_x].notna().sum()
    initial_y = df[col_y].notna().sum()
    initial_overlap = df[col_x].notna() & df[col_y].notna()
    initial_overlap_count = initial_overlap.sum()

    # Fit KNN model from col_x to predict col_y
    df_xy = df[[col_x, col_y]].dropna()
    if not df_xy.empty:
        X_xy = df_xy[[col_x]]
        y_xy = df_xy[col_y]
        knn_xy = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_xy, y_xy)

        mask_y_missing = df[col_y].isna() & df[col_x].notna()
        if mask_y_missing.any():
            X_pred = df.loc[mask_y_missing, [col_x]]
            df.loc[mask_y_missing, col_y] = knn_xy.predict(X_pred)

            if flag_y_col is not None:
                df.loc[mask_y_missing, flag_y_col] = True

    # Fit KNN model from col_y to predict col_x
    df_yx = df[[col_y, col_x]].dropna()
    if not df_yx.empty:
        X_yx = df_yx[[col_y]]
        y_yx = df_yx[col_x]
        knn_yx = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_yx, y_yx)

        mask_x_missing = df[col_x].isna() & df[col_y].notna()
        if mask_x_missing.any():
            X_pred = df.loc[mask_x_missing, [col_y]]
            df.loc[mask_x_missing, col_x] = knn_yx.predict(X_pred)

            if flag_x_col is not None:
                df.loc[mask_x_missing, flag_x_col] = True

    # Record final availability
    final_x = df[col_x].notna().sum()
    final_y = df[col_y].notna().sum()
    final_overlap = df[col_x].notna() & df[col_y].notna()
    final_overlap_count = final_overlap.sum()

    print(f"\nKNN bidirectional imputation summary for {col_x} ↔ {col_y}:")
    print(f"  Initial {col_x} count: {initial_x}")
    print(f"  Initial {col_y} count: {initial_y}")
    print(f"  Initial overlap:       {initial_overlap_count}")
    print(f"  Final {col_x} count:   {final_x}")
    print(f"  Final {col_y} count:   {final_y}")
    print(f"  Final overlap:         {final_overlap_count}")
    print(f"  Added {final_x - initial_x} new {col_x} values via imputation.")
    print(f"  Added {final_y - initial_y} new {col_y} values via imputation.")
    print(f"  Overlap increased by {final_overlap_count - initial_overlap_count}.")

    return df


def plot_knn_imputation_effect(df_before, df_after, col_x, col_y):
    """Plot original vs post-imputation data (SDSS → Gaia, no title)."""

    FEATURE_LABELS = {
        "A_MAG": r"$a^*$",
        "I_MAG_minus_Z_MAG": r"$(i-z)_{\mathrm{SDSS}}$",
        "gaia_slope": r"$S_{\mathrm{Gaia}}$",
        "gaia_z_minus_i": r"$(z-i)_{\mathrm{Gaia}}$",
    }

    xlab = FEATURE_LABELS.get(col_x, col_x)
    ylab = FEATURE_LABELS.get(col_y, col_y)

    plt.figure(figsize=(7, 6))

    # Observed (both present originally)
    mask_orig = df_before[col_x].notna() & df_before[col_y].notna()
    plt.scatter(
        df_before.loc[mask_orig, col_x],
        df_before.loc[mask_orig, col_y],
        s=18,
        alpha=0.9,
        label="Observed"
    )

    # Newly imputed points
    mask_new = (~mask_orig) & df_after[col_x].notna() & df_after[col_y].notna()
    plt.scatter(
        df_after.loc[mask_new, col_x],
        df_after.loc[mask_new, col_y],
        s=15,
        alpha=0.5,
        label="Imputed"
    )

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"Figures_final/knn_imputation_sdss_to_gaia_{col_x}.pdf", bbox_inches="tight")

# Ensure local directory exists os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

# Dictionary to hold DataFrames for each data source to merge
dataframes = {}

# Create base DataFrame from MPCORB if it doesn't exist
df_base_path = os.path.join(LOCAL_DATA_DIR, 'df_base.csv')
if os.path.exists(df_base_path):
    print(f"Loading df_base from {df_base_path}")
    df_base = pd.read_csv(df_base_path)
else:
    df_base = create_df_base()

# Download and load PDS data sources
for name, (subdir, xml_file, data_file, table_key, file_format) in PDS_DATA_SOURCES.items():
    try:
        xml_path, data_path = download_pds_dataset(subdir, xml_file, data_file)
        pds = pdr.read(xml_path)
        df = pds[table_key]
        dataframes[name] = df
        print(f"{name} data loaded: {len(df)} records.")
    except Exception as e:
        print(f"Failed to load {name}: {e}")

df_proper_orbits = dataframes.get("proper_orbits", None)
df_colour = dataframes.get("colour", None)

# Concatenate all albedo_* DataFrames into one
albedo_keys = [k for k in dataframes if k.startswith('albedo_')]
df_albedo = pd.concat([dataframes[k] for k in albedo_keys], ignore_index=True)
df_spectra = dataframes.get("spectra", None)

# Clean and preprocess DataFrames
df_colour[['U_MAG', 'G_MAG', 'R_MAG', 'I_MAG', 'Z_MAG', 'A_MAG']] = (
    df_colour[['U_MAG', 'G_MAG', 'R_MAG', 'I_MAG', 'Z_MAG', 'A_MAG']].replace(99.99, np.nan)
)
df_colour['AST_NUMBER'] = df_colour['AST_NUMBER'].replace(0, np.nan)
df_colour['PROV_ID'] = df_colour['PROV_ID'].replace('-                   ', np.nan)
df_colour['AST_NUMBER'] = df_colour['AST_NUMBER'].astype('Int64')
df_spectra.replace(-999999.0, np.nan, inplace=True)

# Strip whitespace from all string columns
for df in [df_base, df_proper_orbits, df_colour, df_albedo, df_spectra]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()

# Counts for reporting
n_proper_orbits = len(df_proper_orbits)
n_colour = len(df_colour)
n_albedo = len(df_albedo)
n_spectra = len(df_spectra)

# Prepare/rename columns for merging around packed_mpc_name df_base.rename(columns={'desig': 'packed_mpc_name', 'number': 'asteroid_number', 'name_clean': 'prov_id'}, inplace=True)

df_proper_orbits_sub = df_proper_orbits.rename(columns={'PACKED_MPC_NAME': 'packed_mpc_name'})[
    ['packed_mpc_name', 'PROPER_SEMIMAJOR_AXIS', 'PROPER_ECCENTRICITY', 'SINE_OF_PROPER_INCLINATION', 'ABSOLUTE_MAGNITUDE']
]

df_colour_sub = df_colour.rename(columns={'AST_NUMBER': 'asteroid_number', 'PROV_ID': 'prov_id'})[
    ['asteroid_number', 'prov_id', 'U_MAG', 'G_MAG', 'R_MAG', 'I_MAG', 'Z_MAG', 'A_MAG']
]

# Merge colour on asteroid_number
df_colour_num = pd.merge(
    df_colour_sub,
    df_base[['asteroid_number', 'packed_mpc_name']],
    on='asteroid_number',
    how='left'
).rename(columns={'packed_mpc_name': 'packed_mpc_name_num'})

# Merge colour on prov_id
df_colour_prov = pd.merge(
    df_colour_sub,
    df_base[['prov_id', 'packed_mpc_name']],
    on='prov_id',
    how='left'
).rename(columns={'packed_mpc_name': 'packed_mpc_name_prov'})

# Combine the two packed_mpc_name columns, preferring asteroid_number match
df_colour_sub['packed_mpc_name'] = df_colour_num['packed_mpc_name_num'].combine_first(
    df_colour_prov['packed_mpc_name_prov']
)
df_colour_sub.drop(columns=['asteroid_number', 'prov_id'], inplace=True)

df_albedo_sub = df_albedo.rename(columns={'MPC_packed_name': 'packed_mpc_name'})[
    ['packed_mpc_name', 'Diameter', 'V_albedo']
]

df_spectra_sub = df_spectra.rename(columns={'Number': 'asteroid_number', 'Slope': 'spectral_slope'})[
    ['asteroid_number', 'spectral_slope']
]

df_spectra_sub = pd.merge(
    df_spectra_sub,
    df_base[['asteroid_number', 'packed_mpc_name']],
    on='asteroid_number',
    how='left'
).drop(columns=['asteroid_number'])

# Aggregate each PDS DataFrame before merging
df_albedo_sub_agg = df_albedo_sub.groupby('packed_mpc_name', as_index=False).mean(numeric_only=True)
df_colour_sub_agg = df_colour_sub.groupby('packed_mpc_name', as_index=False).mean(numeric_only=True)
df_spectra_sub_agg = df_spectra_sub.groupby('packed_mpc_name', as_index=False).mean(numeric_only=True)

# Aggregated counts
n_albedo_agg = len(df_albedo_sub_agg)
n_colour_agg = len(df_colour_sub_agg)
n_spectra_agg = len(df_spectra_sub_agg)

print("Merging data sources into unified DataFrame...")

# Create the unified DataFrame starting from the base DataFrame
unified_df = df_base.copy()

# Merge by packed_mpc_name
unified_df = pd.merge(unified_df, df_proper_orbits_sub, on='packed_mpc_name', how='left')
unified_df = pd.merge(unified_df, df_albedo_sub_agg, on='packed_mpc_name', how='left')
unified_df = pd.merge(unified_df, df_colour_sub_agg, on='packed_mpc_name', how='left')
unified_df = pd.merge(unified_df, df_spectra_sub_agg, on='packed_mpc_name', how='left')

# Add derived columns
unified_df['I_MAG_minus_Z_MAG'] = unified_df['I_MAG'] - unified_df['Z_MAG']

n_orbiting_colors = unified_df['PROPER_SEMIMAJOR_AXIS'].notna() & unified_df['A_MAG'].notna()
n_orbiting_spectra = unified_df['PROPER_SEMIMAJOR_AXIS'].notna() & unified_df['spectral_slope'].notna()
n_orbiting_albedo = unified_df['PROPER_SEMIMAJOR_AXIS'].notna() & unified_df['V_albedo'].notna()

print("")
print("Summary of data:")
print(f"Total identified asteroids from MPC: {len(df_base)}")
print(f"Total proper orbits from Nesvorny: {n_proper_orbits}")
print(f"Total color measurements from SDSS: {n_colour} ({n_colour_agg} with IDs)")
print(f"Total albedo measurements from NeoWISE: {n_albedo} ({n_albedo_agg} with IDs)")
print(f"Total spectra measurements from PRIMASS: {n_spectra} ({n_spectra_agg} with IDs)")
print(f"Total orbiting asteroids with colour: {n_orbiting_colors.sum()}")
print(f"Total orbiting asteroids with albedo: {n_orbiting_albedo.sum()}")
print(f"Total orbiting asteroids with spectra: {n_orbiting_spectra.sum()}")

# Call this before processing family files
ensure_families_subdirs_downloaded()

# Add empty columns for family membership
unified_df['family_2015'] = pd.Series([pd.NA] * len(unified_df), dtype='str')
unified_df['family_2015_2'] = pd.Series([pd.NA] * len(unified_df), dtype='str')
unified_df['family_2024'] = pd.Series([pd.NA] * len(unified_df), dtype='str')
unified_df['family_2024_2'] = pd.Series([pd.NA] * len(unified_df), dtype='str')

# Add 2015 families
for xml_path in glob.glob('data/families_2015/*.xml'):
    try:
        fam = pdr.read(xml_path)
        df_fam = fam['table']
        if 'FAMILY_NAME' not in df_fam.columns:
            continue
        fam_name = str(df_fam['FAMILY_NAME'].iloc[0])
        mask = unified_df['asteroid_number'].isin(df_fam['AST_NUMBER'])
        already_filled = mask & unified_df['family_2015'].notna()
        unified_df.loc[mask & ~already_filled, 'family_2015'] = fam_name
        unified_df.loc[already_filled, 'family_2015_2'] = fam_name
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")

# Add 2024 families
for xml_path in glob.glob('data/families_2024/*.xml'):
    try:
        fam = pdr.read(xml_path)
        df_fam = fam['TABLE']
        if 'MPC_PACKED_NAME' not in df_fam.columns:
            continue

        fname = os.path.basename(xml_path)
        match = re.match(r'.*?_(\d+)_([a-z0-9]+)_fam3\.xml', fname, re.IGNORECASE)
        if match:
            fam_name = f"{match.group(1)} {match.group(2).capitalize()}"
        else:
            fam_name = fname

        mask = unified_df['packed_mpc_name'].isin(df_fam['MPC_PACKED_NAME'])
        already_filled = mask & unified_df['family_2024'].notna()
        unified_df.loc[mask & ~already_filled, 'family_2024'] = fam_name
        unified_df.loc[already_filled, 'family_2024_2'] = fam_name
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")

# Add a column indicating if the asteroid is in any family
family_cols = ["family_2015", "family_2015_2", "family_2024", "family_2024_2"]
unified_df["in_family"] = unified_df[family_cols].notna().any(axis=1).astype(int)

# Clean family columns before save
unified_df['family_2015'] = unified_df['family_2015'].astype(str).str.strip()
unified_df['family_2015_2'] = unified_df['family_2015_2'].astype(str).str.strip()
unified_df['family_2024'] = unified_df['family_2024'].astype(str).str.strip()
unified_df['family_2024_2'] = unified_df['family_2024_2'].astype(str).str.strip()

print("Adding Gaia data...")

BASE_GAIA_URL = "https://cdn.gea.esac.esa.int/Gaia/gdr3/Solar_system/sso_reflectance_spectrum/"
LOCAL_GAIA_DIR = "data/gaia"

os.makedirs(LOCAL_GAIA_DIR, exist_ok=True)

# Scrape the directory for .csv.gz files
resp = requests.get(BASE_GAIA_URL)
resp.raise_for_status()
soup = BeautifulSoup(resp.text, "html.parser")
links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.csv.gz')]

# Download missing files
for fname in links:
    local_gz = os.path.join(LOCAL_GAIA_DIR, fname)
    if not os.path.exists(local_gz):
        print(f"Downloading {fname}...")
        with requests.get(BASE_GAIA_URL + fname, stream=True) as r:
            r.raise_for_status()
            with open(local_gz, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        print(f"{fname} already exists, skipping download.")

    local_csv = local_gz[:-3]
    if not os.path.exists(local_csv):
        print(f"Unzipping {fname}...")
        with gzip.open(local_gz, 'rb') as f_in, open(local_csv, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        print(f"{os.path.basename(local_csv)} already exists, skipping unzip.")

# Read and concatenate all CSVs
csv_files = [os.path.join(LOCAL_GAIA_DIR, f) for f in os.listdir(LOCAL_GAIA_DIR) if f.endswith('.csv')]
dfs = []
for csv_file in csv_files:
    print(f"Reading {csv_file}...")
    table = ascii.read(csv_file, format="ecsv", fill_values=[('null', 'nan')])
    df = table.to_pandas()
    dfs.append(df)

full_spectra_df = pd.concat(dfs, ignore_index=True)

# Define wavelength range for slope calculation
slope_min = 450.0
slope_max = 750.0
norm_wavelength = 550.0

# Filter for RSF == 0 and wavelength range
mask = (
    (full_spectra_df['wavelength'] >= slope_min) &
    (full_spectra_df['wavelength'] <= slope_max) &
    (full_spectra_df['reflectance_spectrum_flag'] == 0)
)
df_filtered = full_spectra_df[mask]

results = []

print("Calculating slopes for Gaia reflectance spectra...")

for number_mp, group in df_filtered.groupby('number_mp'):
    wavelengths = group['wavelength'].values
    reflectance = group['reflectance_spectrum'].values

    reflectance_550 = np.interp(norm_wavelength, wavelengths, reflectance)
    reflectance_norm = reflectance / reflectance_550

    wavelengths_um = wavelengths / 1000.0
    slope, intercept, r_value, p_value, std_err = linregress(wavelengths_um, reflectance_norm)

    slope_pct_per_0_1um = slope * 10

    results.append({
        'asteroid_number': number_mp,
        'slope_pct_per_0.1um': slope_pct_per_0_1um,
        'r_squared': r_value**2,
        'n_points': len(wavelengths)
    })

slope_df = pd.DataFrame(results)
slope_df_renamed = slope_df.rename(columns={'slope_pct_per_0.1um': 'gaia_slope'})

# Calculate z - i color index using reflectance at 748 nm (i) and 893.2 nm (z)
print("Calculating z - i color index from gaia data...")

z_nm = 893.2
i_nm = 748.0

z_minus_i_results = []

for number_mp, group in full_spectra_df.groupby('number_mp'):
    group0 = group[group['reflectance_spectrum_flag'] == 0]
    if len(group0) < 4:
        z_minus_i_results.append({'asteroid_number': number_mp, 'gaia_z_minus_i': np.nan})
        continue

    wavelengths = group0['wavelength'].values
    reflectance = group0['reflectance_spectrum'].values

    try:
        spline = csaps(wavelengths, reflectance, smooth=5e-7)
        Rz = float(spline(z_nm))
        Ri = float(spline(i_nm))
        if Rz > 0 and Ri > 0:
            z_minus_i = 2.5 * np.log10(Rz / Ri)
        else:
            z_minus_i = np.nan
    except Exception as e:
        z_minus_i = np.nan
        print(f"Error calculating z - i for asteroid {number_mp}")
        print(f"Error: {e}")

    z_minus_i_results.append({'asteroid_number': number_mp, 'gaia_z_minus_i': z_minus_i})

z_minus_i_df = pd.DataFrame(z_minus_i_results)

# Merge z_minus_i into slope_df_renamed
slope_df_renamed = pd.merge(slope_df_renamed, z_minus_i_df, on='asteroid_number', how='left')

# Merge Gaia-derived features into unified_df
unified_df = pd.merge(
    unified_df,
    slope_df_renamed[['asteroid_number', 'gaia_slope', 'gaia_z_minus_i']],
    on='asteroid_number',
    how='left'
)

# Add imputation flag columns for each feature that can be filled by KNN. # These are feature-specific flags, so later plots can colour points by the # imputation status of the quantity actually being plotted.
imputation_flag_cols = [
    'A_MAG_imputed',
    'gaia_slope_imputed',
    'I_MAG_minus_Z_MAG_imputed',
    'gaia_z_minus_i_imputed',
]

for flag_col in imputation_flag_cols:
    unified_df[flag_col] = False

# Backwards-compatible alias used by some older scripts.
unified_df['slope_imputed'] = False

# Apply KNN bidirectional imputation between SDSS a* and Gaia slope.
df_before = unified_df.copy()
unified_df = knn_bidirectional_impute(
    unified_df,
    'A_MAG',
    'gaia_slope',
    flag_x_col='A_MAG_imputed',
    flag_y_col='gaia_slope_imputed'
)
unified_df['slope_imputed'] = unified_df['gaia_slope_imputed']
plot_knn_imputation_effect(df_before, unified_df,  'gaia_slope','A_MAG')

# Apply KNN bidirectional imputation between SDSS (i-z) and Gaia (z-i).
df_before = unified_df.copy()
unified_df = knn_bidirectional_impute(
    unified_df,
    'I_MAG_minus_Z_MAG',
    'gaia_z_minus_i',
    flag_x_col='I_MAG_minus_Z_MAG_imputed',
    flag_y_col='gaia_z_minus_i_imputed'
)
plot_knn_imputation_effect(df_before, unified_df, 'gaia_z_minus_i','I_MAG_minus_Z_MAG')

output_fits_path = os.path.join(PDS_DATA_DIR, 'unified_asteroid_data_with_gaia.fits')
table = Table.from_pandas(unified_df)
table.write(output_fits_path, overwrite=True)

print(unified_df.count())
for flag_col in imputation_flag_cols + ['slope_imputed']:
    print(f"\n{flag_col}:")
    print(unified_df[flag_col].value_counts(dropna=False))
print(f"Unified asteroid data saved to {output_fits_path}")
