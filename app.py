# app.py

import streamlit as st
import pandas as pd
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openpyxl import load_workbook
from openpyxl.utils import range_boundaries, get_column_letter

# ✅ Page config FIRST
st.set_page_config(page_title="CSI Coder using AI")

# Load the hidden CSI reference table from an Excel Table
@st.cache_data
def load_reference():
    file_path = "Master_CSI.xlsx"
    # Load workbook (tables available in normal mode)
    wb = load_workbook(filename=file_path, data_only=True)
    table_name = "CSI_Integration_Tool"
    # Find the table in any sheet
    sheet_name = None
    table_ref = None
    for ws in wb.worksheets:
        if table_name in ws.tables:
            sheet_name = ws.title
            table_ref = ws.tables[table_name].ref  # e.g. 'A1:D100'
            break
    if not table_ref:
        raise ValueError(f"Table '{table_name}' not found in {file_path}")

    # Parse table_ref to boundaries
    start_col, start_row, end_col, end_row = range_boundaries(table_ref)
    start_col_letter = get_column_letter(start_col)
    end_col_letter = get_column_letter(end_col)
    usecols = f"{start_col_letter}:{end_col_letter}"
    header_row = start_row - 1

    # Read exactly the table into pandas
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
    # Load user BOQ DataFrame
    if uploaded_file.name.lower().endswith("csv"):
        boq_df = pd.read_csv(uploaded_file)
    else:
        boq_df = pd.read_excel(uploaded_file)

    # Ensure column names are strings
    boq_df.columns = [str(c) for c in boq_df.columns]
    # User selects the description column
    selected_column = st.selectbox(
        "🔽 Select the BOQ column to code:",
        boq_df.columns
    )

    if st.button("🚀 Match Items to CSI Codes"):
        with st.spinner("Matching using AI..."):
            # Compute embeddings
            texts = boq_df[selected_column].astype(str).tolist()
            emb_texts = model.encode(texts, convert_to_numpy=True)
            emb_texts = np.array(emb_texts, dtype=float)
            if emb_texts.ndim == 1:
                emb_texts = emb_texts.reshape(1, -1)

            # Reference embeddings from table
            ref_keywords = reference_df.iloc[:, 0].astype(str).tolist()
            emb_refs = model.encode(ref_keywords, convert_to_numpy=True)
            emb_refs = np.array(emb_refs, dtype=float)
            if emb_refs.ndim == 1:
                emb_refs = emb_refs.reshape(1, -1)

            # Matching
            results = []
            for i, txt in enumerate(texts):
                sim_scores = cosine_similarity(
                    emb_texts[i].reshape(1, -1),
                    emb_refs
                )[0]
                top_idx = int(np.argmax(sim_scores))

                # Build result entry with full reference row
                ref_row = reference_df.iloc[top_idx].to_dict()
                entry = {"BOQ Description": txt}
                entry.update(ref_row)
                results.append(entry)

            # Display and download
            result_df = pd.DataFrame(results)
            cols = ["BOQ Description"] + reference_df.columns.tolist()
            result_df = result_df[cols]

            st.success("✅ Matching complete!")
            st.dataframe(result_df)

            # Excel download
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
