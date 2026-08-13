"""
ShmooPreprocessor
-----------------
Loads CSV / Excel, validates required columns, normalizes string formatting,
auto-detects single vs. multi-die datasets, and engineers features for the ML model.

Time complexity : O(N)  for all operations (N = number of rows)
Space complexity: O(N)  – one copy of the dataframe with extra columns
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Column contracts ──────────────────────────────────────────────────────────
REQUIRED_COLS = {'Point_ID', 'VDD_V', 'Frequency_GHz', 'Test_Result', 'Failure_Code'}
OPTIONAL_COLS = {
    'Lot_ID', 'Wafer_ID', 'Die_ID', 'Temperature_C',
    'Current_mA', 'Timing_ns', 'Leakage_mA',
    'Test_Time_ms', 'Pattern_ID', 'Test_ID', 'Margin_GHz',
    'March_Algorithm', 'Memory_Instance', 'Memory_Address',  # M-BIST columns
}
DIE_ID_COLS = ['Lot_ID', 'Wafer_ID', 'Die_ID']


class ShmooPreprocessor:
    def __init__(self):
        self.is_multi_die: bool = False
        self.die_groups: list  = []
        self.df_raw: pd.DataFrame | None = None
        self.df_processed: pd.DataFrame | None = None

    def load(self, filepath: str) -> dict:
        """Load file, validate schema, return metadata dict."""
        path = Path(filepath)
        suffix = path.suffix.lower()

        if suffix in ('.xlsx', '.xls'):
            df = pd.read_excel(filepath)
        elif suffix == '.csv':
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: '{suffix}'. Use CSV or Excel.")

        self._validate_columns(df)
        self.df_raw = df
        return self._detect_structure(df)

    def process(self) -> pd.DataFrame:
        """Engineer features; call after load(). Returns processed DataFrame."""
        if self.df_raw is None:
            raise RuntimeError("Call load() before process().")

        df = self.df_raw.copy()

        # ── Robust String Normalization ───────────────────────────────────────
        df['Test_Result']  = df['Test_Result'].astype(str).str.strip().str.upper()
        df['Failure_Code'] = df['Failure_Code'].fillna('NA').astype(str).str.strip().str.upper()
        df['Failure_Code'] = df['Failure_Code'].replace({'NAN': 'NA', 'NONE': 'NA', '': 'NA', 'NULL': 'NA', 'N/A': 'NA'})

        pass_mask = df['Test_Result'].isin(['PASS', 'PASSED', '1', 'TRUE', 'P'])
        df['Test_Result'] = np.where(pass_mask, 'PASS', 'FAIL')
        df['label']       = pass_mask.astype(int)

        # ── Core engineered features ──────────────────────────────────────────
        df['vdd_freq_product'] = df['VDD_V'] * df['Frequency_GHz']
        df['freq_per_volt']    = df['Frequency_GHz'] / df['VDD_V']
        df['vdd_squared']      = df['VDD_V'] ** 2
        df['freq_squared']     = df['Frequency_GHz'] ** 2

        vdd_min, vdd_max   = float(df['VDD_V'].min()), float(df['VDD_V'].max())
        freq_min, freq_max = float(df['Frequency_GHz'].min()), float(df['Frequency_GHz'].max())

        denom_v = (vdd_max  - vdd_min)  or 1.0
        denom_f = (freq_max - freq_min) or 1.0

        df['vdd_norm']  = (df['VDD_V']           - vdd_min)  / denom_v
        df['freq_norm'] = (df['Frequency_GHz']   - freq_min) / denom_f

        # ── Optional numerical features – median-impute if missing values ─────
        for col in ('Margin_GHz', 'Timing_ns', 'Current_mA', 'Leakage_mA'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

        # ── Generalized Fault-Rate Feature (Works for Scan or M-BIST) ─────────
        if 'Pattern_ID' in df.columns:
            group_key = 'Pattern_ID'
        elif {'March_Algorithm', 'Memory_Instance'}.issubset(df.columns):
            group_key = ['March_Algorithm', 'Memory_Instance']
        else:
            group_key = None

        if group_key is not None:
            fail_rate = 1.0 - df.groupby(group_key)['label'].transform('mean')
            df['fault_rate'] = fail_rate
            df['pattern_fail_rate'] = fail_rate

        self.df_processed = df
        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}\n"
                f"Found columns: {list(df.columns)}"
            )

    def _detect_structure(self, df: pd.DataFrame) -> dict:
        """Auto-detect single vs. multi-die and build metadata."""
        id_cols = [c for c in DIE_ID_COLS if c in df.columns]

        if id_cols:
            n_groups = df.groupby(id_cols).ngroups
        else:
            n_groups = 1

        self.is_multi_die = n_groups > 1
        self.die_groups   = id_cols if self.is_multi_die else []

        test_result_clean = df['Test_Result'].astype(str).str.strip().str.upper()
        pass_mask = test_result_clean.isin(['PASS', 'PASSED', '1', 'TRUE', 'P'])
        fail_code_clean = df['Failure_Code'].fillna('NA').astype(str).str.strip().str.upper().replace({'NAN': 'NA', 'NONE': 'NA', '': 'NA', 'NULL': 'NA', 'N/A': 'NA'})

        return {
            'n_points':      len(df),
            'n_dies':        n_groups,
            'is_multi_die':  self.is_multi_die,
            'die_id_cols':   id_cols,
            'vdd_range':     [float(df['VDD_V'].min()),          float(df['VDD_V'].max())],
            'freq_range':    [float(df['Frequency_GHz'].min()),  float(df['Frequency_GHz'].max())],
            'pass_rate':     float(pass_mask.mean()),
            'n_pass':        int(pass_mask.sum()),
            'n_fail':        int((~pass_mask).sum()),
            'failure_codes': fail_code_clean.value_counts().to_dict(),
            # Lot / wafer / die info if present
            'lot_id':   str(df['Lot_ID'].iloc[0])   if 'Lot_ID'   in df.columns else 'N/A',
            'wafer_id': str(df['Wafer_ID'].iloc[0]) if 'Wafer_ID' in df.columns else 'N/A',
            'die_id':   str(df['Die_ID'].iloc[0])   if 'Die_ID'   in df.columns else 'N/A',
            'temp_c':   float(df['Temperature_C'].iloc[0]) if 'Temperature_C' in df.columns else None,
        }
