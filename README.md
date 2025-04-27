# Sepsis SOFA Equity Analyzer

Live on: https://duke-datathon-25-pears-team.streamlit.app/

**Overview**: Streamlit app by Team PEARS for Duke Datathon 2025, analyzing sepsis outcomes in MIMIC-IV ICU data (~33k stays) for SDoH disparities, SOFA-based mortality predictions, survival patterns, and intersectional risks.

**Features**:
- Cohort stats and distributions
- SOFA model performance (AUC, ROC) by SDoH
- Kaplan-Meier survival curves and Cox models
- Model comparison and intersectional risk tables (age, gender, ethnicity, insurance)

**Requirements**:
- Python 3.8+
- Install: `pip install streamlit pandas plotly pillow numpy`

**Setup & Run**:
1. Place files in `~/Documents/code/MIT/duke_datathon_2025`
2. Ensure `data/` has required CSVs, PKLs, and PNGs
3. Run: `streamlit run app.py`
4. Open: `http://localhost:8501`

**Usage**:
- Use sidebar to navigate: EDA, Equity, Survival, Summaries, Intersectional Risk
- Select SDoH factor for stratification
- View model AUCs and risk tables