# app.py

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from io import BytesIO

# ✅ Page config FIRST
st.set_page_config(page_title="CSI Coder using AI")

# Load the hidden CSI reference table
@st.cache_data
def load_reference():
    df = pd.read_excel("Master_CSI.xlsx")
    df = df.dropna()
    df.columns = [col.strip() for col in df.columns]  # Clean column names
    return df

reference_df = load_reference()
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("🧠 CSI Coder using AI")
st.markdown("Upload your BOQ item list and get AI-matched CSI codes based on our internal classification reference.")

# File uploader for BOQ
uploaded_file = st.file_uploader("📎 Upload BOQ Item List (Excel or CSV)", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith("csv"):
        boq_df = pd.read_csv(uploaded_file)
    else:
        boq_df = pd.read_excel(uploaded_file)

    boq_df.columns = [str(col) for col in boq_df.columns]  # Ensure column names are strings
    selected_column = st.selectbox("🔽 Select the column to be coded:", boq_df.columns)

    if st.button("🚀 Match Items to CSI Codes"):
        with st.spinner("Matching using AI..."):
            boq_descriptions = boq_df[selected_column].astype(str).tolist()
            boq_embeddings = model.encode(boq_descriptions, convert_to_tensor=True)

            reference_keywords = reference_df.iloc[:, 0].astype(str).tolist()
            reference_codes = reference_df.iloc[:, 1].astype(str).tolist()
            ref_embeddings = model.encode(reference_keywords, convert_to_tensor=True)

            results = []
            for i, boq_text in enumerate(boq_descriptions):
                cosine_scores = util.cos_sim(boq_embeddings[i], ref_embeddings)[0]
                top_idx = cosine_scores.argmax().item()
                matched_keyword = reference_keywords[top_idx]
                matched_code = reference_codes[top_idx]
                results.append({
                    "BOQ Description": boq_text,
                    "Matched CSI Code": matched_code,
                    "Matched Keyword": matched_keyword
                })

            result_df = pd.DataFrame(results)
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

