# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from openpyxl import load_workbook
from openpyxl.utils import range_boundaries, get_column_letter

# ✅ Page config FIRST
st.set_page_config(page_title="CSI Coder using AI")

# Load the hidden CSI reference table from an Excel Table
@st.cache_data
def load_reference():
    file_path = "Master_CSI.xlsx"
    wb = load_workbook(filename=file_path, data_only=True)
    table_name = "CSI_Integration_Tool"
    sheet_name = None
    table_ref = None
    for ws in wb.worksheets:
        if table_name in ws.tables:
            sheet_name = ws.title
            table_ref = ws.tables[table_name].ref
            break
    if not table_ref:
        raise ValueError(f"Table '{table_name}' not found in {file_path}")

    start_col, start_row, end_col, end_row = range_boundaries(table_ref)
    start_col_letter = get_column_letter(start_col)
    end_col_letter = get_column_letter(end_col)
    usecols = f"{start_col_letter}:{end_col_letter}"
    header_row = start_row - 1

    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=header_row,
        usecols=usecols,
        engine="openpyxl"
    )
    df = df.iloc[: (end_row - start_row + 1)]
    df = df.dropna(how="all")
    df.columns = [col.strip() for col in df.columns]
    return df

reference_df = load_reference()
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("🧠 CSI Coder using AI")
st.markdown("Upload your BOQ item list and get AI-matched CSI codes with full reference data from our hidden Excel table.")

# File uploader for BOQ items
uploaded_file = st.file_uploader(
    "📎 Upload BOQ Item List (Excel or CSV)",
    type=["xlsx", "csv"]
)

if uploaded_file:
    # Read BOQ file and auto-detect header
    if uploaded_file.name.lower().endswith("csv"):
        raw_df = pd.read_csv(uploaded_file, header=None)
    else:
        raw_df = pd.read_excel(uploaded_file, header=None)
    # Drop completely empty rows
    raw_df = raw_df.dropna(how='all')
    # Use first non-empty row as header
    new_header = raw_df.iloc[0].astype(str).tolist()
    df_data = raw_df[1:].copy()
    # Assign detected header
    boq_df = df_data
    boq_df.columns = new_header
    selected_column = st.selectbox(
        "🔽 Select the BOQ column to code:",
        boq_df.columns
    )

    if st.button("🚀 Match Items to CSI Codes"):
        with st.spinner("Matching using AI..."):
            texts = boq_df[selected_column].astype(str).tolist()
            emb_texts = model.encode(texts, convert_to_numpy=True)
            emb_texts = np.array(emb_texts, dtype=float)
            if emb_texts.ndim == 1:
                emb_texts = emb_texts.reshape(1, -1)

            ref_keywords = reference_df.iloc[:, 0].astype(str).tolist()
            emb_refs = model.encode(ref_keywords, convert_to_numpy=True)
            emb_refs = np.array(emb_refs, dtype=float)
            if emb_refs.ndim == 1:
                emb_refs = emb_refs.reshape(1, -1)

            results = []
            for i, txt in enumerate(texts):
                a = emb_texts[i].reshape(1, -1)
                sim_scores = cosine_similarity(a, emb_refs)[0]
                top_idx = int(np.argmax(sim_scores))
                score = float(sim_scores[top_idx])

                # Determine confidence
                if score >= 0.75:
                    confidence = 'High'
                elif score >= 0.5:
                    confidence = 'Medium'
                else:
                    confidence = 'Low'

                # Build entry with full reference row
                ref_row = reference_df.iloc[top_idx].to_dict()
                entry = {"BOQ Description": txt}
                entry.update(ref_row)
                entry["Confidence"] = confidence
                results.append(entry)

            result_df = pd.DataFrame(results)
            cols = ["BOQ Description"] + reference_df.columns.tolist() + ["Confidence"]
            result_df = result_df[cols]

            st.success("✅ Matching complete with confidence levels!")
            st.dataframe(result_df)

            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                result_df.to_excel(writer, index=False, sheet_name='Matched Results')
            output.seek(0)
            st.download_button(
                label="⬇️ Download Matched Results as Excel",
                data=output,
                file_name="CSI_Matched_Output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
