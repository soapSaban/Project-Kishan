# agronomy_config.py

"""
PROJECT KISHAN - CROP PHENOLOGY DATABASE
Defines biological yield potentials (relative to Rice) and stress categories.
"""

CROP_SPECS = {
    # --- HEAVY BIOMASS / TROPICAL ---
    'Sugarcane': {'bio_factor': 2.50, 'type': 'wet_tropical'}, # 100+ tons
    'Oil Palm':  {'bio_factor': 1.80, 'type': 'wet_tropical'},
    'Coconut':   {'bio_factor': 1.20, 'type': 'tropical'},
    'Rubber':    {'bio_factor': 0.90, 'type': 'tropical'},
    
    # --- CEREAL STAPLES ---
    'Rice':      {'bio_factor': 1.00, 'type': 'wet_tropical'}, # Baseline
    'Maize':     {'bio_factor': 0.95, 'type': 'standard'},
    'Wheat':     {'bio_factor': 0.85, 'type': 'cool_season'},
    'Barley':    {'bio_factor': 0.80, 'type': 'cool_season'},
    
    # --- MILLETS & SORGHUM (Hardy) ---
    'Jowar':     {'bio_factor': 0.80, 'type': 'dryland'},
    'Sorghum':   {'bio_factor': 0.80, 'type': 'dryland'},
    'Bajra':     {'bio_factor': 0.70, 'type': 'dryland'},
    'Ragi':      {'bio_factor': 0.70, 'type': 'dryland'},
    'Millets':   {'bio_factor': 0.70, 'type': 'dryland'},
    
    # --- PULSES & OILSEEDS ---
    'Soybean':   {'bio_factor': 0.75, 'type': 'standard'},
    'Groundnut': {'bio_factor': 0.70, 'type': 'dryland'},
    'Sunflower': {'bio_factor': 0.65, 'type': 'standard'},
    'Mustard':   {'bio_factor': 0.60, 'type': 'cool_season'},
    'Linseed':   {'bio_factor': 0.60, 'type': 'cool_season'},
    'Sesame':    {'bio_factor': 0.55, 'type': 'dryland'},
    'Pulses':    {'bio_factor': 0.60, 'type': 'dryland'},
    
    # --- CASH CROPS ---
    'Cotton':    {'bio_factor': 0.65, 'type': 'dry_finish'},
    'Tobacco':   {'bio_factor': 0.70, 'type': 'standard'},
    'Tea':       {'bio_factor': 0.65, 'type': 'hill'},
    'Coffee':    {'bio_factor': 0.60, 'type': 'shade'},
    'Cashew Nut':{'bio_factor': 0.85, 'type': 'tropical'},
    
    # --- SPICES & SPECIALTY ---
    'Turmeric':  {'bio_factor': 0.85, 'type': 'standard'},
    'Ginger':    {'bio_factor': 0.80, 'type': 'standard'},
    'Black Pepper':{'bio_factor': 0.40, 'type': 'tropical'},
    'Cardamom':  {'bio_factor': 0.30, 'type': 'tropical'},
    'Saffron':   {'bio_factor': 0.05, 'type': 'cool_season'}, # Tiny yield
    
    # --- DEFAULT ---
    'default':   {'bio_factor': 0.80, 'type': 'standard'}
}