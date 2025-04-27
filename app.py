# working app.py streamlit

import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
import os
from PIL import Image # To display PNG images
import numpy as np # For potential numeric checks

# --- Page Configuration ---
st.set_page_config(
    page_title="Sepsis SOFA Equity Analyzer | Team PEARS", # Added team name
    page_icon="🍐", # Changed icon :)
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
DATA_DIR = "data"
MIN_GROUP_SIZE_INTERSECTIONAL = 30 # Define minimum size for intersectional table display

# Define clearer model names mapping
MODEL_NAME_MAP = {
    'SOFA_Only': 'SOFA Score Only',
    'SOFA_plus_Controls': 'SOFA + Controls (Age, Charlson)',
    'SOFA_plus_Controls_plus_SDoH_ALL': 'SOFA + Controls + SDoH Proxies'
}


# --- Caching Data Loading ---
@st.cache_data # Cache results to improve performance
def load_analysis_results(data_dir=DATA_DIR):
    """Loads all necessary analysis results from files."""
    results = {}
    files_to_load = {
        # Original Files
        'eda_results': 'eda_results.pkl',
        'auc_results': 'equity_auc_results.pkl',
        'roc_results': 'equity_roc_results.pkl',
        'survival_results': 'survival_analysis_results.pkl',
        'table1_overall_df': 'table1_overall.csv', # Load as DF
        'table1_mortality_df': 'table1_stratified_mortality.csv', # Load as DF
        'table1_ethnicity_df': 'table1_stratified_ethnicity.csv', # Load as DF
        'cox_main_df': 'cox_model_main_summary.csv', # Load as DF
        'cox_interact_df': 'cox_model_interaction_summary.csv', # Load as DF
        'final_cohort_df': 'final_prepared_cohort_with_survival.csv', # Load cohort for EDA plots

        # New Files from Part 6 / Secondary Analysis
        'model_comp_df': 'model_performance_comparison.csv', # Assuming this holds the AUC comparison
        'intersectional_df_min30_raw': f'intersectional_stats_filtered_MIN{MIN_GROUP_SIZE_INTERSECTIONAL}.csv' # Load raw for processing
        # Note: Removed 'intersectional_df_min30' key initially, will create it after processing
    }
    all_files_found = True
    essential_files_found = True # Track essential files specifically

    essential_keys = ['auc_results', 'roc_results', 'final_cohort_df'] # Files needed for core functionality

    print("--- Loading Data ---")
    for key, filename in files_to_load.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            # Only show sidebar warning for potentially expected files
            if key in essential_keys or key.startswith('table1') or key.startswith('km_plot') or key.startswith('cox') or key in ['model_comp_df', 'intersectional_df_min30_raw']:
                 st.sidebar.warning(f"Warning: File not found - {filename}")
            else:
                 print(f"Optional file not found, skipping: {filename}")

            results[key] = None
            if key in essential_keys:
                essential_files_found = False
            continue

        try:
            if filename.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    results[key] = pickle.load(f)
            elif filename.endswith('.csv'):
                # Load CSVs into DataFrames
                try:
                     # Try loading with index detection first
                     results[key] = pd.read_csv(filepath, index_col=0)
                     # Clean common 'Unnamed: 0' index name
                     if results[key].index.name == 'Unnamed: 0':
                          results[key].index.name = None
                except (ValueError, IndexError): # If index_col=0 fails
                    print(f"Index col 0 failed for {filename}, trying without index...")
                    try:
                        results[key] = pd.read_csv(filepath)
                    except Exception as e_inner:
                        st.sidebar.error(f"Error loading {filename}: {e_inner}")
                        results[key] = None
                        if key in essential_keys: essential_files_found = False
                        all_files_found = False
                except Exception as csv_load_err:
                     st.sidebar.error(f"Error loading {filename}: {csv_load_err}")
                     results[key] = None
                     if key in essential_keys: essential_files_found = False
                     all_files_found = False


                 # Special handling for intersectional multi-index structure (after load)
                if key.startswith('intersectional') and isinstance(results[key], pd.DataFrame) and 'age_group' in results[key].columns:
                    potential_index_cols = ['age_group', 'gender', 'ethnicity', 'insurance', 'language']
                    index_cols_present = [col for col in potential_index_cols if col in results[key].columns]
                    if len(index_cols_present) > 1:
                        try:
                            # If index is not already set correctly, try setting it
                            if not isinstance(results[key].index, pd.MultiIndex) or set(results[key].index.names) != set(index_cols_present):
                                 # Reload without auto index and set manually if needed
                                 # Check if the index cols are actual cols or already index
                                 if all(col in results[key].columns for col in index_cols_present):
                                     results[key] = results[key].set_index(index_cols_present)
                                     print(f"Set multi-index for {key}")
                                 else:
                                     print(f"Could not set multi-index for {key}, columns missing or already index.")
                            # else: index is already correct
                        except Exception as multi_idx_e:
                            print(f"Could not set multi-index for {key}: {multi_idx_e}. Using loaded structure.")

            print(f"Loaded: {filename} (Key: {key})")

        except Exception as e:
            st.sidebar.error(f"Error processing {filename}: {e}")
            results[key] = None
            if key in essential_keys: essential_files_found = False
            all_files_found = False

     # --- Post-processing after loading ---

    # --- Process Intersectional Data ---
    results['intersectional_df_min30'] = None # Initialize key
    if results.get('intersectional_df_min30_raw') is not None and isinstance(results['intersectional_df_min30_raw'], pd.DataFrame):
        print("Processing intersectional data...")
        df_intersectional = results['intersectional_df_min30_raw'].copy() # Work on a copy

        # Define original column names expected from the notebook output
        original_col_map = {
            'Group Size': 'Group Size',
            'Observed Mortality (%)': 'Observed Mortality',
            'Avg Pred Risk (SOFA_Only) (%)': f'Avg Pred Risk ({MODEL_NAME_MAP["SOFA_Only"]})',
            'Avg Pred Risk (SOFA_plus_Controls) (%)': f'Avg Pred Risk ({MODEL_NAME_MAP["SOFA_plus_Controls"]})',
            'Avg Pred Risk (SOFA_plus_Controls_plus_SDoH_ALL) (%)': f'Avg Pred Risk ({MODEL_NAME_MAP["SOFA_plus_Controls_plus_SDoH_ALL"]})'
        }

        # Find which original columns actually exist in the loaded DataFrame
        cols_to_rename = {old: new for old, new in original_col_map.items() if old in df_intersectional.columns}
        df_intersectional = df_intersectional.rename(columns=cols_to_rename)
        print(f"Renamed columns: {list(cols_to_rename.values())}")

        # Identify columns intended to be percentages (based on ORIGINAL names)
        percent_cols_original = [col for col in original_col_map.keys() if 'Mortality' in col or 'Risk' in col]
        # Get the NEW names of these columns after renaming
        percent_cols_new = [original_col_map[col] for col in percent_cols_original if col in cols_to_rename] # Use the new names

        print(f"Attempting conversion for percentage columns: {percent_cols_new}")
        for col in percent_cols_new:
             if col in df_intersectional.columns:
                original_dtype = df_intersectional[col].dtype
                try:
                    # Step 1: Convert to string, remove '%' if present
                    series = df_intersectional[col].astype(str).str.replace('%', '', regex=False).str.strip()
                    # Step 2: Convert to numeric, coercing errors (makes non-numbers NaN)
                    numeric_series = pd.to_numeric(series, errors='coerce')

                    # Step 3: If it was percentage (0-100), convert to proportion (0-1)
                    # Check if *any* non-NaN value was > 1 before dividing
                    if not numeric_series.isna().all() and numeric_series.max() > 1:
                         print(f"Column '{col}' appears to be 0-100 scale, converting to proportion.")
                         df_intersectional[col] = numeric_series / 100.0
                    else:
                         df_intersectional[col] = numeric_series # Assume it was already proportion or became NaN

                    # Step 4: Check final dtype
                    final_dtype = df_intersectional[col].dtype
                    if pd.api.types.is_numeric_dtype(final_dtype):
                        nan_count = df_intersectional[col].isna().sum()
                        print(f"Successfully converted column '{col}' to numeric (dtype: {final_dtype}). Original dtype: {original_dtype}. NaN count: {nan_count}.")
                    else:
                         print(f"Warning: Column '{col}' FAILED to convert to numeric (remains {final_dtype}). Original dtype: {original_dtype}. Check source data for non-numeric values.")

                except Exception as e:
                    print(f"ERROR during conversion of column '{col}'. Error: {e}. Column left as is (dtype: {original_dtype}).")
             else:
                 print(f"Warning: Expected percentage column '{col}' not found after renaming.")

        results['intersectional_df_min30'] = df_intersectional # Store the processed DataFrame
        print("Finished processing intersectional columns.")
        # print(results['intersectional_df_min30'].head()) # Debug print
        # print(results['intersectional_df_min30'].dtypes) # Debug print

    # Rename models in comparison table
    if results.get('model_comp_df') is not None and isinstance(results['model_comp_df'], pd.DataFrame):
         df_comp = results['model_comp_df']
         if 'Model' in df_comp.columns:
              df_comp['Model'] = df_comp['Model'].map(MODEL_NAME_MAP).fillna(df_comp['Model']) # Replace known names, keep others
              results['model_comp_df'] = df_comp
              print("Renamed models in comparison table.")


    # Load KM plot image paths
    results['km_plots'] = {}
    for factor in ['ethnicity', 'insurance', 'language']:
        img_filename = f'km_plot_{factor}.png'
        img_filepath = os.path.join(data_dir, img_filename)
        if os.path.exists(img_filepath):
            results['km_plots'][factor] = img_filepath
        else:
            results['km_plots'][factor] = None

    if not essential_files_found:
         st.sidebar.error("Some ESSENTIAL result files were missing. Core functionality may be broken.")

    return results

# --- Load Data ---
results = load_analysis_results()

# --- App Title & Intro ---
col_title, col_logo = st.columns([4, 1])
with col_title:
    st.title("🍐 Sepsis & SOFA Equity Analyzer")
    st.caption("Project by Team PEARS | Duke Datathon 2025")
with col_logo:
    st.write("") # Just adds some space

st.markdown("""
This dashboard presents findings from an analysis of MIMIC-IV data for ICU patients meeting Sepsis-3 criteria (~33k stays).
We investigated potential disparities related to Social Determinants of Health (SDoH) by analyzing cohort characteristics, assessing the equity of a First-Day SOFA-based mortality prediction model, exploring survival patterns, and comparing different predictive models.
""")
st.markdown("---")

# --- Check if essential results loaded ---
if results.get('auc_results') is None or results.get('roc_results') is None or results.get('final_cohort_df') is None:
     st.error("Essential analysis results (AUC/ROC pickles or final cohort CSV) could not be loaded. Cannot display visualizations.")
     st.stop() # Stop execution if core data is missing

# --- Define SDoH Factors & Other Vars ---
race_col = 'ethnicity' # Assuming 'ethnicity' is the definite column name now
sdoh_factors_available = [col for col in [race_col, 'insurance', 'language'] if col in results['final_cohort_df'].columns]
numeric_vars = ['age', 'sofa', 'charlson_comorbidity_index']
categorical_vars = ['gender'] + sdoh_factors_available

# --- Sidebar ---
st.sidebar.header("⚙️ Navigation & Options")
sidebar_options = ["Cohort Overview (EDA)",
                   "SOFA Equity Analysis",
                   "Survival Analysis",
                   "Model Summaries (Cox)",
                   "Model Comparison & Intersectional Risk"] # Added new section
selected_section = st.sidebar.radio("Select Analysis Section:",
                                   options=sidebar_options,
                                   index=4) # Default to the new section for testing
st.sidebar.markdown("---")

# Dropdown for Equity and KM plots (only show if results available)
if results.get('auc_results') and results.get('roc_results') and results.get('km_plots'):
    st.sidebar.header("Stratification Options")
    # Check if sdoh_factors_available is not empty before creating selectbox
    if sdoh_factors_available:
        default_sdoh_index = 0
        sdoh_display_factor = st.sidebar.selectbox(
            "Stratify Equity/Survival by:",
            options=sdoh_factors_available,
            index=default_sdoh_index
        )
        st.sidebar.info(f"Visualizations in 'SOFA Equity' and 'Survival Analysis' tabs will be stratified by **{sdoh_display_factor}** where applicable.")
    else:
        st.sidebar.warning("No SDoH factor columns (ethnicity, insurance, language) found in the cohort data.")
        sdoh_display_factor = None
else:
    sdoh_display_factor = None # Set to None if results are missing


# ===========================================================
# === Main Panel Display ====================================
# ===========================================================

# --- Tab 1: Cohort Overview (EDA) ---
if selected_section == "Cohort Overview (EDA)":
    st.header("📊 Cohort Overview (Exploratory Data Analysis)")
    st.markdown("Baseline characteristics and distributions for the final sepsis cohort.")

    # --- Display Table 1 Summaries ---
    st.subheader("Cohort Characteristics (Table 1 Style)")
    tab1_overall, tab1_mort, tab1_sdoh = st.tabs(["Overall", "By Mortality", f"By {sdoh_display_factor if sdoh_display_factor else 'Ethnicity'}"])

    with tab1_overall:
        if results.get('table1_overall_df') is not None:
            st.markdown("**Overall Cohort:**")
            st.dataframe(results['table1_overall_df'], use_container_width=True)
        else:
            st.warning("Overall Table 1 CSV not found.")

    with tab1_mort:
        if results.get('table1_mortality_df') is not None:
            st.markdown("**Stratified by 30-Day Mortality:**")
            st.dataframe(results['table1_mortality_df'], use_container_width=True)
        else: st.warning("Mortality-stratified Table 1 CSV not found.")

    with tab1_sdoh:
        if sdoh_display_factor:
            if sdoh_display_factor == race_col and results.get('table1_ethnicity_df') is not None:
                st.markdown(f"**Stratified by {sdoh_display_factor}:**")
                st.dataframe(results['table1_ethnicity_df'], use_container_width=True)
            elif sdoh_display_factor != race_col:
                 st.info(f"Pre-computed Table 1 stratified by '{sdoh_display_factor}' not available (only ethnicity was generated).")
            else: # Ethnicity selected but file missing
                 st.warning(f"Ethnicity-stratified Table 1 CSV not found.")
        else:
             st.warning("No SDoH factor selected for stratification in the sidebar.")

    st.markdown("---")
    # --- Interactive Data Distributions ---
    st.subheader("Interactive Data Distributions")
    st.markdown("Explore distributions of key variables.")

    eda_col1, eda_col2 = st.columns(2)
    with eda_col1:
        dist_var_type = st.radio("Select Variable Type:", ["Numeric", "Categorical"], horizontal=True, key="dist_var_type")
        if dist_var_type == "Numeric":
            dist_var_select = st.selectbox("Select Numeric Variable:", options=numeric_vars, key="dist_num_var")
        else:
            if categorical_vars:
                 dist_var_select = st.selectbox("Select Categorical Variable:", options=categorical_vars, key="dist_cat_var")
            else:
                 st.warning("No categorical variables available for selection.")
                 dist_var_select = None

    with eda_col2:
        stratification_options = ["None"] + [var for var in categorical_vars if var in results['final_cohort_df'].columns]
        if 'mortality_30day' in results['final_cohort_df'].columns:
            stratification_options.append('mortality_30day')

        if stratification_options:
             dist_stratify_select = st.selectbox("Stratify plot by (Optional):", options=stratification_options, index=0, key="dist_stratify")
        else:
             st.warning("No variables available for stratification.")
             dist_stratify_select = "None"


    # Generate plot based on selections
    if dist_var_select and dist_var_select in results['final_cohort_df'].columns:
        df_plot = results['final_cohort_df']
        stratify_col = None if dist_stratify_select == "None" else dist_stratify_select

        if stratify_col and stratify_col not in df_plot.columns:
            st.warning(f"Stratification column '{stratify_col}' not found in data. Plotting without stratification.")
            stratify_col = None

        try:
            if dist_var_type == "Numeric":
                st.markdown(f"**Distribution of {dist_var_select}**")
                fig_dist = px.histogram(
                    df_plot.dropna(subset=[dist_var_select]), # Drop NaNs for numeric plotting
                    x=dist_var_select,
                    color=stratify_col,
                    marginal="box",
                    barmode='overlay',
                    opacity=0.7,
                    title=f"Distribution of {dist_var_select}{f' by {stratify_col}' if stratify_col else ''}"
                )
                fig_dist.update_layout(xaxis_title=dist_var_select)
                st.plotly_chart(fig_dist, use_container_width=True)

            else: # Categorical
                st.markdown(f"**Distribution of {dist_var_select}**")
                if stratify_col:
                     count_df = df_plot[[dist_var_select, stratify_col]].dropna().groupby([dist_var_select, stratify_col], observed=True).size().reset_index(name='Count') # Use observed=True
                     fig_dist = px.bar(
                          count_df,
                          x=dist_var_select,
                          y='Count',
                          color=stratify_col,
                          barmode='group',
                          title=f"Counts of {dist_var_select} by {stratify_col}",
                          text_auto=True
                     )
                else:
                     count_df = df_plot[dist_var_select].dropna().value_counts().reset_index()
                     count_df.columns = [dist_var_select, 'Count']
                     fig_dist = px.bar(
                         count_df,
                         x=dist_var_select,
                         y='Count',
                         title=f"Counts of {dist_var_select}",
                         text_auto=True
                     )
                fig_dist.update_layout(xaxis_title=dist_var_select, yaxis_title="Number of Stays")
                st.plotly_chart(fig_dist, use_container_width=True)

        except Exception as plot_err:
            st.error(f"An error occurred while generating the plot for '{dist_var_select}': {plot_err}")
            columns_to_show = [dist_var_select] + ([stratify_col] if stratify_col else [])
            st.dataframe(df_plot[columns_to_show].head())
    elif dist_var_select:
         st.warning(f"Selected variable '{dist_var_select}' not found in the dataset.")


# --- Tab 2: SOFA Equity Analysis ---
elif selected_section == "SOFA Equity Analysis":
    st.header("⚖️ SOFA Equity Analysis: Stratified Model Performance")
    if not sdoh_display_factor:
        st.warning("Please select an SDoH factor in the sidebar to view stratified results.")
    else:
        st.markdown(f"""
        Assessing the performance of a Logistic Regression model (using First-Day SOFA + Controls: Age, Charlson Index)
        for predicting 30-day mortality **within** different patient groups defined by **{sdoh_display_factor.replace('_', ' ').title()}**.
        """)

        auc_data_all = results.get('auc_results', {})
        roc_data_all = results.get('roc_results', {})

        if not isinstance(auc_data_all, dict):
            st.error("Error: AUC results are not in the expected format (dictionary).")
            auc_data_all = {}
        if not isinstance(roc_data_all, dict):
             st.error("Error: ROC results are not in the expected format (dictionary).")
             roc_data_all = {}

        factor_auc_data = auc_data_all.get(sdoh_display_factor, {})
        factor_roc_data = roc_data_all.get(sdoh_display_factor, {})

        if not factor_auc_data or not factor_roc_data:
             st.warning(f"No equity analysis results found for factor: **{sdoh_display_factor}**. Check if the analysis was run for this factor.")
        else:
            plot_categories = list(factor_auc_data.keys())
            plot_aucs = [factor_auc_data.get(cat, None) for cat in plot_categories]
            valid_plot_data = {cat: auc for cat, auc in zip(plot_categories, plot_aucs) if auc is not None and isinstance(auc, (float, np.number))}

            if not valid_plot_data:
                 st.warning(f"No valid AUC data found for factor '{sdoh_display_factor}'.")
            else:
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("Model Discrimination (AUC)")
                    auc_df = pd.DataFrame(valid_plot_data.items(), columns=['Category', 'AUC Score']).sort_values('AUC Score', ascending=True)
                    fig_bar = px.bar(auc_df,
                                     x='AUC Score',
                                     y='Category',
                                     orientation='h',
                                     title=f"AUC by {sdoh_display_factor.replace('_', ' ').title()}", text_auto='.3f',
                                     color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_bar.update_layout(yaxis_title=sdoh_display_factor.replace('_', ' ').title(),
                                          xaxis_title="Area Under ROC Curve (AUC)",
                                          xaxis_range=[0.5, 1.0],
                                          height=max(400, 50 + len(valid_plot_data)*30))
                    fig_bar.update_traces(textposition='outside')
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.caption("Higher AUC = better model discrimination *within* that specific group.")

                with col2:
                    st.subheader("ROC Curves Comparison")
                    fig_roc = go.Figure()
                    fig_roc.add_shape(type='line', line=dict(dash='dash', color='grey'), x0=0, x1=1, y0=0, y1=1)
                    colors = px.colors.qualitative.Plotly

                    plot_counter = 0
                    for category, auc_val in valid_plot_data.items():
                        if category in factor_roc_data:
                             roc_info = factor_roc_data[category]
                             if isinstance(roc_info, dict) and 'fpr' in roc_info and 'tpr' in roc_info:
                                 if isinstance(roc_info['fpr'], (list, np.ndarray)) and isinstance(roc_info['tpr'], (list, np.ndarray)):
                                     color_idx = plot_counter % len(colors)
                                     fig_roc.add_trace(go.Scatter(x=roc_info['fpr'], y=roc_info['tpr'],
                                                                  name=f"{category} (AUC={auc_val:.3f})",
                                                                  line=dict(color=colors[color_idx], width=2), mode='lines'))
                                     plot_counter += 1
                                 else:
                                     st.warning(f"ROC data for category '{category}' has 'fpr' or 'tpr' that are not lists or arrays.")
                             else:
                                 st.warning(f"ROC data for category '{category}' is missing expected 'fpr' or 'tpr' keys, or is not a dictionary.")
                        else:
                             st.warning(f"ROC data structure missing for category '{category}'.")


                    fig_roc.update_layout(xaxis_title='False Positive Rate (1 - Specificity)',
                                          yaxis_title='True Positive Rate (Sensitivity)',
                                          yaxis=dict(scaleanchor="x", scaleratio=1),
                                          xaxis=dict(constrain='domain'),
                                          title=f"ROC Curves by {sdoh_display_factor.replace('_', ' ').title()}",
                                          legend_title_text='Category', height=500, hovermode="x unified")
                    st.plotly_chart(fig_roc, use_container_width=True)
                    st.caption("Curves closer to top-left = better performance.")


# --- Tab 3: Survival Analysis ---
elif selected_section == "Survival Analysis":
    st.header("⏳ Survival Analysis")
    if not sdoh_display_factor:
        st.warning("Please select an SDoH factor in the sidebar to view stratified results.")
    else:
        st.subheader("Kaplan-Meier Survival Curves")
        st.markdown(f"""
        Comparing estimated 30-day survival probabilities based on **{sdoh_display_factor.replace('_', ' ').title()}**.
        These curves show the proportion of patients surviving over time for each group.
        """)
        km_plot_path = results.get('km_plots', {}).get(sdoh_display_factor)
        if km_plot_path and os.path.exists(km_plot_path):
            try:
                 image = Image.open(km_plot_path)
                 st.image(image, caption=f"Kaplan-Meier Curves stratified by {sdoh_display_factor.replace('_', ' ').title()}", use_column_width='always')
                 survival_results = results.get('survival_results', {})
                 if isinstance(survival_results, dict):
                    km_results = survival_results.get('km_results', {})
                    logrank_key = f'logrank_{sdoh_display_factor}'
                    if isinstance(km_results, dict) and logrank_key in km_results:
                        st.write(f"**Log-rank Test Results (Comparison vs Baseline Group for {sdoh_display_factor}):**")
                        logrank_data = km_results[logrank_key]
                        if isinstance(logrank_data, dict):
                            formatted_logrank = {k: f"{v:.3g}" if isinstance(v, (float, np.number)) else v for k, v in logrank_data.items()}
                            st.json(formatted_logrank)
                        elif isinstance(logrank_data, (float, np.number)):
                            st.metric("Overall Log-rank Test p-value", f"{logrank_data:.3g}")
                        else:
                             st.write(f"Log-rank data format not recognized: {type(logrank_data)}")
                        st.caption("Log-rank test assesses if there's a statistically significant difference between the survival curves (p < 0.05 indicates significance).")
                    else:
                        st.info("Log-rank test results not found in the loaded survival analysis data for this factor.")
                 else:
                      st.warning("Survival results structure not as expected (should be a dictionary).")

            except Exception as img_e:
                st.error(f"Error displaying KM plot image '{km_plot_path}': {img_e}")
        else:
            st.warning(f"Kaplan-Meier plot image file not found for **{sdoh_display_factor}**. Expected at: `{os.path.join(DATA_DIR, f'km_plot_{sdoh_display_factor}.png')}`")

        st.markdown("---")
        st.subheader("Cox Proportional Hazards Model: SOFA Interaction Effects")
        st.markdown(f"""
        Does the **impact of First-Day SOFA score on mortality risk *differ* across the selected SDoH groups ({sdoh_display_factor.replace('_', ' ').title()})?**
        This table shows results from a Cox model including interaction terms between SOFA and the categories of **{sdoh_display_factor}**.
        Significant p-values (`p < 0.05`) for the interaction terms (e.g., `sofa_x_{sdoh_display_factor}[T.SOME_GROUP]`) suggest that the effect of SOFA on hazard of death is *not the same* for that group compared to the reference group.
        """)
        if results.get('cox_interact_df') is not None and isinstance(results['cox_interact_df'], pd.DataFrame):
            cox_summary_interact = results['cox_interact_df'].copy()
            if 'covariate' not in cox_summary_interact.columns:
                 cox_summary_interact = cox_summary_interact.reset_index().rename(columns={'index':'covariate'})

            sofa_row = cox_summary_interact[cox_summary_interact['covariate'] == 'sofa']
            interaction_rows = cox_summary_interact[cox_summary_interact['covariate'].str.contains(f'sofa_x_{sdoh_display_factor}', na=False)]

            if not interaction_rows.empty or not sofa_row.empty:
                display_df = pd.concat([sofa_row, interaction_rows]).drop_duplicates(subset=['covariate']).set_index('covariate')

                def highlight_p_val(p):
                    try:
                        p_float = float(p)
                        if pd.isna(p_float): return ''
                        if p_float < 0.001: return '***'
                        elif p_float < 0.01: return '**'
                        elif p_float < 0.05: return '*'
                    except (ValueError, TypeError):
                        pass
                    return ''

                cols_to_show_base = ['coef', 'exp(coef)', 'se(coef)', 'p', 'Signif.']
                if 'p' in display_df.columns:
                    display_df['Signif.'] = display_df['p'].apply(highlight_p_val)
                else:
                    st.warning("P-value column ('p') not found in Cox interaction summary. Significance stars cannot be shown.")
                    cols_to_show_base.remove('Signif.')
                    if 'p' in cols_to_show_base: cols_to_show_base.remove('p')

                cols_to_show = [col for col in cols_to_show_base if col in display_df.columns]
                formatters = {'coef': '{:.3f}', 'exp(coef)': '{:.3f}', 'se(coef)': '{:.3f}', 'p': '{:.3g}'}
                valid_formatters = {k: v for k, v in formatters.items() if k in display_df.columns}

                st.dataframe(display_df[cols_to_show].style.format(valid_formatters))
                st.caption("`exp(coef)` = Hazard Ratio (HR). HR > 1 indicates increased risk, HR < 1 indicates decreased risk. `p` < 0.05 (*) suggests statistical significance. Interaction term significance implies SOFA's effect varies by group.")
            else:
                 st.info(f"No SOFA main effect or interaction terms found for **{sdoh_display_factor}** in the Cox interaction model summary.")
        else:
            st.warning("Cox Interaction Model summary CSV (`cox_model_interaction_summary.csv`) not found or not loaded as a DataFrame.")


# --- Tab 4: Model Summaries (Cox) ---
elif selected_section == "Model Summaries (Cox)":
     st.header("📄 Detailed Cox Model Summaries")
     st.markdown("Full statistical output from the survival regression models (Cox Proportional Hazards).")

     exp_main = st.expander("View Main Effects Cox Model Summary", expanded=False)
     with exp_main:
          if results.get('cox_main_df') is not None and isinstance(results['cox_main_df'], pd.DataFrame):
                st.markdown("This model assesses the independent effect of each variable (including SOFA, controls, and SDoH factors) on 30-day mortality risk, assuming the effect of SOFA is the same across groups.")
                st.dataframe(results['cox_main_df'], use_container_width=True)
          else: st.warning("Main Effects Cox Model summary CSV (`cox_model_main_summary.csv`) not found or not loaded correctly.")

     exp_int = st.expander("View Interaction Effects Cox Model Summary", expanded=True)
     with exp_int:
          if results.get('cox_interact_df') is not None and isinstance(results['cox_interact_df'], pd.DataFrame):
                st.markdown("This model tests if the effect of SOFA score on mortality differs across various SDoH groups by including interaction terms (e.g., `sofa_x_ethnicityGROUP`). It shows all terms included in the model.")
                st.dataframe(results['cox_interact_df'], use_container_width=True)
          else: st.warning("Interaction Effects Cox Model summary CSV (`cox_model_interaction_summary.csv`) not found or not loaded correctly.")

# --- Tab 5: Model Comparison & Intersectional Risk --- CORRECTED SECTION ---
elif selected_section == "Model Comparison & Intersectional Risk":
    st.header("🔄 Model Comparison & Intersectional Risk")
    st.markdown("""
    This section compares the overall performance of different logistic regression models for predicting 30-day mortality and examines how well these models predict risk across specific intersectional patient groups.
    """)

    # --- Model Performance Comparison ---
    st.subheader("Overall Model Performance (AUC)")
    st.markdown(f"""
    Comparing the Area Under the ROC Curve (AUC) for models with different sets of predictors. A higher AUC indicates better overall discrimination between patients who died and those who survived within 30 days.
    - **{MODEL_NAME_MAP['SOFA_Only']}**: Uses only the first-day SOFA score.
    - **{MODEL_NAME_MAP['SOFA_plus_Controls']}**: Adds Age and Charlson Comorbidity Index.
    - **{MODEL_NAME_MAP['SOFA_plus_Controls_plus_SDoH_ALL']}**: Adds Gender, Ethnicity, Insurance, and Language proxies.
    """)
    if results.get('model_comp_df') is not None and isinstance(results['model_comp_df'], pd.DataFrame):
        model_perf_df = results['model_comp_df'].copy()
        auc_col_name = next((col for col in model_perf_df.columns if 'AUC' in col), None)

        if auc_col_name and 'Model' in model_perf_df.columns:
             model_perf_df = model_perf_df.sort_values(auc_col_name, ascending=True)
             fig_comp_auc = px.bar(model_perf_df,
                                  x=auc_col_name,
                                  y='Model',
                                  orientation='h',
                                  title="Model Comparison by Test Set AUC",
                                  text_auto='.4f',
                                  color_discrete_sequence=px.colors.qualitative.Set2)
             fig_comp_auc.update_layout(xaxis_title="Area Under ROC Curve (AUC) on Test Set",
                                        yaxis_title="Model",
                                        xaxis_range=[0.5, max(0.85, model_perf_df[auc_col_name].max() * 1.05)],
                                        height=max(300, 50 + len(model_perf_df)*40))
             fig_comp_auc.update_traces(textposition='outside')
             st.plotly_chart(fig_comp_auc, use_container_width=True)
             st.caption("Comparison based on performance on the held-out test set.")

        elif not model_perf_df.empty:
             st.markdown("**Model Performance Table:**")
             st.dataframe(model_perf_df, use_container_width=True)
             st.caption(f"Could not automatically detect 'Model' and/or an 'AUC' column for plotting. Displaying raw table.")
        else:
            st.warning("Model performance comparison data frame is empty.")

    else:
        st.warning("Model Performance Comparison CSV (`model_performance_comparison.csv`) not found or not loaded correctly.")

    st.markdown("---")

    # --- Intersectional Risk Assessment ---
    st.subheader(f"Intersectional Group Risk Assessment (Groups >= {MIN_GROUP_SIZE_INTERSECTIONAL} Patients)")
    st.markdown(f"""
    Comparing the **observed** 30-day mortality rate within specific intersectional groups (defined by Age Group, Gender, Ethnicity, and Insurance) against the **average predicted** mortality risk from the different models for patients in that group.
    Discrepancies between observed and predicted risk can highlight groups where models may under- or over-estimate risk.
    *Note: Results are filtered to show only groups with at least {MIN_GROUP_SIZE_INTERSECTIONAL} patients to improve reliability.*
    """)
    # Use the processed DataFrame directly from results
    intersectional_df = results.get('intersectional_df_min30')

    if intersectional_df is not None and isinstance(intersectional_df, pd.DataFrame):
        display_df = intersectional_df.copy()

        # Prepare for formatting: Identify NUMERIC columns intended for percentage display
        # These columns should have been converted to numeric in load_analysis_results
        numeric_percent_cols = []
        potential_percent_cols = [col for col in display_df.columns if 'Mortality' in col or 'Risk' in col]
        for col in potential_percent_cols:
             if pd.api.types.is_numeric_dtype(display_df[col]):
                 numeric_percent_cols.append(col)
             else:
                 # Inform user if a column expected to be numeric isn't
                 st.warning(f"Column '{col}' intended for percentage formatting is not numeric (dtype: {display_df[col].dtype}). It will not be formatted as percent. Check data loading logs.")

        format_dict = {col: "{:.1%}" for col in numeric_percent_cols} # Format only numeric ones as XX.X%

        # Display the dataframe with formatting. Reset index if it's a multi-index for better display
        if isinstance(display_df.index, pd.MultiIndex):
            display_df_styled = display_df.reset_index()
        else:
             display_df_styled = display_df

        # Reorder columns for better display
        cols_ordered = []
        if 'Group Size' in display_df_styled.columns:
            cols_ordered.append('Group Size')
        if isinstance(display_df.index, pd.MultiIndex): # Add index columns if they were reset
            cols_ordered.extend(display_df.index.names)
        # Add remaining columns (including the correctly named & potentially formatted ones)
        cols_ordered.extend([col for col in display_df_styled.columns if col not in cols_ordered])
        display_df_styled = display_df_styled[cols_ordered]


        # Apply the style formatting only to the columns that are actually numeric
        st.dataframe(display_df_styled.style.format(format_dict, na_rep="-"), use_container_width=True)
        st.caption(f"""
        Table shows: Group characteristics (Index), Group Size, Actual Observed Mortality (%), and Average Predicted Risk from the '{MODEL_NAME_MAP['SOFA_Only']}', '{MODEL_NAME_MAP['SOFA_plus_Controls']}', and '{MODEL_NAME_MAP['SOFA_plus_Controls_plus_SDoH_ALL']}' models.
        Look for groups with high observed mortality but lower predicted risk (underestimation) or vice-versa (overestimation). All percentages are based on the test set. Columns that could not be converted to numeric are shown without percentage formatting.
        """)

    elif results.get('intersectional_df_min30_raw') is not None:
        # This case means the initial file was loaded but processing failed somehow
         st.error("Intersectional data was loaded but could not be processed correctly. Please check the data loading logs in the console/terminal for specific errors regarding column conversion.")
         st.markdown("**Raw Loaded Intersectional Data:**")
         st.dataframe(results['intersectional_df_min30_raw'], use_container_width=True)
    else:
        st.warning(f"Filtered Intersectional Stats CSV (`intersectional_stats_filtered_MIN{MIN_GROUP_SIZE_INTERSECTIONAL}.csv`) not found or not loaded correctly.")


# --- Footer ---
st.markdown("---")
st.markdown("Created by **Team PEARS** | Duke Datathon 2025 | Analysis based on MIMIC-IV data")