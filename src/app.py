import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Demonstration d'outils de NLP basés sur des LLM ! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Cette interaface streamlit à pour objectif de rendre aggréable l'utilisation des différentes démonstrations de NLP implémentées dans ce Repo.
    """
)

st.markdown(
    """
    Acuellemment disponible :
    - Traducteur Anglais/Français
    """
)