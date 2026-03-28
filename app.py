import streamlit as st
import pdfplumber
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import io
from collections import defaultdict

# -------------------------------
# 1. Extract Q&A from PDF
# -------------------------------
def extract_qa_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    # More flexible pattern: matches Q. / Question / Q No. etc. and Answer / Ans.
    qa_pairs = []
    lines = text.split('\n')
    current_q = None
    current_a = []
    in_answer = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for question marker
        if re.match(r'^(?:Q\.?\s*\d*\.?\s*|Question\s*\d*\.?\s*)', line, re.I):
            if current_q and current_a:
                qa_pairs.append((current_q, ' '.join(current_a)))
            current_q = re.sub(r'^(?:Q\.?\s*\d*\.?\s*|Question\s*\d*\.?\s*)', '', line)
            current_a = []
            in_answer = False
        
        # Check for answer marker
        elif re.match(r'^(?:Ans\.?|Answer|Solution)\s*:?', line, re.I):
            in_answer = True
            ans_part = re.sub(r'^(?:Ans\.?|Answer|Solution)\s*:?', '', line)
            current_a.append(ans_part)
        
        elif in_answer:
            current_a.append(line)
        
        elif current_q:
            # Might be continuation of question (no answer marker yet)
            current_q += ' ' + line
    
    if current_q and current_a:
        qa_pairs.append((current_q, ' '.join(current_a)))
    
    # Clean
    return [(q.strip(), a.strip()) for q, a in qa_pairs if q and a]

# -------------------------------
# 2. Group similar questions
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def group_questions(qa_pairs_list, threshold=0.75):
    all_qs = []
    doc_indices = []
    for doc_idx, pairs in enumerate(qa_pairs_list):
        for q, a in pairs:
            all_qs.append(q)
            doc_indices.append(doc_idx)
    
    if not all_qs:
        return {}
    
    model = load_model()
    embeddings = model.encode(all_qs, show_progress_bar=False)
    sim = cosine_similarity(embeddings)
    
    clusters = {}
    visited = set()
    for i in range(len(all_qs)):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(i+1, len(all_qs)):
            if j not in visited and sim[i][j] >= threshold:
                cluster.append(j)
                visited.add(j)
        clusters[i] = cluster
    
    cluster_data = {}
    for seed, members in clusters.items():
        doc_set = set()
        q_list = []
        a_list = []
        for idx in members:
            doc_set.add(doc_indices[idx])
            q_list.append(all_qs[idx])
            # Find corresponding answer
            doc_idx = doc_indices[idx]
            pair_index = [qa[0] for qa in qa_pairs_list[doc_idx]].index(all_qs[idx])
            a_list.append(qa_pairs_list[doc_idx][pair_index][1])
        cluster_data[seed] = {
            'questions': q_list,
            'answers': a_list,
            'doc_count': len(doc_set),
            'sample_q': q_list[0],
            'sample_a': a_list[0]
        }
    return cluster_data

# -------------------------------
# 3. Apply amendments
# -------------------------------
def apply_amendments(answer, mapping_df):
    if mapping_df is None or mapping_df.empty:
        return answer
    for _, row in mapping_df.iterrows():
        old = row['old_text']
        new = row['new_text']
        answer = answer.replace(old, new)
    return answer

# -------------------------------
# 4. Generate PDF
# -------------------------------
def create_pdf(repetitive, never_asked, amendments_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="CA Exam Concept Analysis", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="1. Highly Repetitive Concepts", ln=True)
    pdf.set_font("Arial", size=11)
    for i, item in enumerate(repetitive):
        pdf.multi_cell(0, 8, txt=f"Concept {i+1}: {item['sample_q']}")
        ans = apply_amendments(item['sample_a'], amendments_df)
        pdf.multi_cell(0, 8, txt=f"Sample Answer (amended): {ans[:500]}...")
        pdf.ln(4)
    
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="2. Rare / Never Asked Concepts", ln=True)
    pdf.set_font("Arial", size=11)
    for i, item in enumerate(never_asked):
        pdf.multi_cell(0, 8, txt=f"Concept {i+1}: {item['sample_q']}")
        ans = apply_amendments(item['sample_a'], amendments_df)
        pdf.multi_cell(0, 8, txt=f"Sample Answer (amended): {ans[:500]}...")
        pdf.ln(4)
    
    return pdf.output(dest='S').encode('latin1')

# -------------------------------
# 5. Streamlit UI
# -------------------------------
st.set_page_config(page_title="CA Concept Analyzer", layout="wide")
st.title("📘 CA Exam Concept Analyzer")
st.markdown("Upload past papers, mock tests, and RTPs (PDFs with Q&A). The tool groups similar questions and highlights repetitive vs. rare concepts.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Similarity threshold (higher = stricter grouping)", 0.5, 0.95, 0.75)
    rep_cutoff = st.slider("Repetitive cutoff (% of documents)", 0, 100, 30) / 100.0
    st.write("---")
    st.header("Amendments (optional)")
    amend_file = st.file_uploader("CSV with columns: old_text, new_text", type=['csv'])
    amend_df = None
    if amend_file:
        amend_df = pd.read_csv(amend_file)
        st.success(f"Loaded {len(amend_df)} amendment rules")

uploaded = st.file_uploader("Choose PDF files", type=['pdf'], accept_multiple_files=True)

if uploaded and st.button("Analyze"):
    with st.spinner("Extracting Q&A from PDFs..."):
        all_qa = []
        for f in uploaded:
            qa = extract_qa_from_pdf(f)
            if qa:
                all_qa.append(qa)
            else:
                st.warning(f"No Q&A found in {f.name}. Check formatting.")
        if not all_qa:
            st.error("No valid Q&A extracted. Ensure PDFs contain clearly marked Q. and Ans.")
            st.stop()
        st.success(f"Extracted from {len(all_qa)} documents.")
    
    with st.spinner("Grouping concepts..."):
        clusters = group_questions(all_qa, threshold)
        if not clusters:
            st.error("No groups formed. Try lowering the similarity threshold.")
            st.stop()
    
    total_docs = len(all_qa)
    repetitive = []
    never_asked = []
    for data in clusters.values():
        if data['doc_count'] / total_docs >= rep_cutoff:
            repetitive.append(data)
        else:
            never_asked.append(data)
    
    st.write(f"**{len(repetitive)} repetitive concepts** (≥ {rep_cutoff*100:.0f}% documents)")
    st.write(f"**{len(never_asked)} rare concepts**")
    
    with st.spinner("Generating PDF..."):
        pdf_bytes = create_pdf(repetitive, never_asked, amend_df)
    
    st.download_button("📥 Download Analysis PDF", data=pdf_bytes, file_name="ca_analysis.pdf", mime="application/pdf")
    
    with st.expander("Preview repetitive concepts"):
        for i, item in enumerate(repetitive[:5]):
            st.write(f"**{i+1}.** {item['sample_q']}")
            st.caption(f"Appears in {item['doc_count']}/{total_docs} documents")
