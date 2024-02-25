import streamlit as st
from transformers import TFAutoModelForSeq2SeqLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("my_awesome_opus_books_model")
model = TFAutoModelForSeq2SeqLM.from_pretrained("my_awesome_opus_books_model")
# Titre de l'application
st.title('Démo de Traduction Anglais vers Français avec T5')


# Entrée utilisateur pour le texte à traduire
text_to_translate = st.text_area("Texte à traduire", "Écrivez ici le texte en anglais que vous souhaitez traduire en français...")

# Bouton pour exécuter la traduction
if st.button('Traduire'):
    if text_to_translate:
        # Préparation du texte pour le modèle T5
        inputs = tokenizer(text_to_translate, return_tensors="tf").input_ids
        
        # Génération de la traduction
        outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Afficher le texte traduit
        st.write('Texte Traduit :', translated_text)
    else:
        st.write('Veuillez saisir du texte pour la traduction.')
