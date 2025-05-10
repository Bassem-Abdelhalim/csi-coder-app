# app.py

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from io import BytesIO
import torch

# ✅ Page config FIRST
st.set_page_config(page_title="CSI Coder using AI")

# Load the hidden CSI reference table
@st.cache_data
def load_reference():
    df = pd.read_excel("Master_CSI.xlsx")
    df = df.dropna()
    df.columns = [col.strip() for col in df.columns]
    return df

reference_df = load_reference()
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("🧠 CSI Coder using AI")
st.markdown("Upload your BOQ item list and get AI-matched CSI codes with full reference data.")

# File uploader for BOQ
uploaded_file = st.file_uploader("📎 Upload BOQ Item List (Excel or CSV)", type=["xlsx", "csv"])

if uploaded_file:
    # Read BOQ file
    if uploaded_file.name.endswith("csv"):
        boq_df = pd.read_csv(uploaded_file)
    else:
        boq_df = pd.read_excel(uploaded_file)

    # Ensure column names are strings
    boq_df.columns = [str(col) for col in boq_df.columns]
    # Let user select which BOQ column contains descriptions
    selected_column = st.selectbox("🔽 Select the BOQ column to code:", boq_df.columns)

    if st.button("🚀 Match Items to CSI Codes"):
        with st.spinner("Matching using AI..."):
            # Prepare embeddings
            boq_texts = boq_df[selected_column].astype(str).tolist()
            boq_embeddings = model.encode(boq_texts, convert_to_tensor=True, device='cpu')

            # Reference keywords (first column) for similarity
            ref_keywords = reference_df.iloc[:, 0].astype(str).tolist()
            ref_embeddings = model.encode(ref_keywords, convert_to_tensor=True, device='cpu')

            # Match and compile results
            results = []
            for idx, text in enumerate(boq_texts):
                scores = util.cos_sim(boq_embeddings[idx], ref_embeddings)[0]
                top_idx = scores.argmax().item()
                # Fetch full reference row
                ref_row = reference_df.iloc[top_idx].to_dict()
                # Build result entry
                entry = {"BOQ Description": text}
                entry.update(ref_row)
                results.append(entry)

            # Create DataFrame and reorder columns
            result_df = pd.DataFrame(results)
            cols = ["BOQ Description"] + reference_df.columns.tolist()
            result_df = result_df[cols]

            st.success("✅ Matching complete!")
            st.dataframe(result_df)

            # Export to Excel
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
