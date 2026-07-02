from __future__ import annotations

from dataclasses import asdict

import streamlit as st

from dmease import DMeasePipeline


@st.cache_resource
def load_pipeline() -> DMeasePipeline:
    return DMeasePipeline.from_config("configs/dmease.yaml")


st.set_page_config(page_title="DMease", layout="wide")
st.title("DMease")

pipeline = load_pipeline()
patient_id = st.selectbox("Patient ID", pipeline.patient_db.list_ids())

if st.button("Run inference", type="primary"):
    result = pipeline.infer(patient_id)
    st.subheader("Syndrome")
    st.write(asdict(result.syndrome))

    st.subheader("Targets")
    st.write(result.targets)

    st.subheader("Prescription")
    st.dataframe([asdict(step) for step in result.prescription], use_container_width=True)

    st.subheader("Ranked candidates")
    st.dataframe([asdict(candidate) for candidate in result.ranked_candidates], use_container_width=True)

    st.subheader("Clinical review")
    for note in result.notes:
        st.info(note)
