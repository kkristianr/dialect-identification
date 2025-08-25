############ 1. IMPORTING LIBRARIES ############

# Import streamlit, requests for API calls, and pandas and numpy for data manipulation

import streamlit as st
import requests
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from streamlit_tags import st_tags  # to add labels on the fly!
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

@st.cache_resource()
def load_model():
    model = torch.load('finetuned/pytorch_model.bin', map_location=torch.device('cpu'))
    return model

#model = torch.load('finetuned/pytorch_model.bin', map_location=torch.device('cpu'))
class BERTMultiClass(torch.nn.Module):
    def __init__(self):
        super(BERTMultiClass, self).__init__()
        self.l1 = AutoModel.from_pretrained(model_name)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 12)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

def perform_inference(sample_sentence, model):
    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("kkristianr/finetuned_bert_multilingual")

    # Tokenize and prepare input for the model
    inputs = tokenizer(sample_sentence, return_tensors='pt', truncation=True, padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Perform inference
    with torch.no_grad():
        model.eval()
        outputs = model(input_ids, attention_mask)

    # Get probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # Convert probabilities to a list
    probs_list = probabilities.squeeze().tolist()

    return probs_list


############ 2. SETTING UP THE PAGE LAYOUT AND TITLE ############

# `st.set_page_config` is used to display the default layout width, the title of the app, and the emoticon in the browser tab.

st.set_page_config(
    layout="centered", page_title="Dialect Classifier", page_icon="❄️"
)

############ CREATE THE LOGO AND HEADING ############

# We create a set of columns to display the logo and the heading next to each other.


c1, c2 = st.columns([0.32, 2])

# The snowflake logo will be displayed in the first column, on the left.

with c1:

    st.image(
        "images/loco.png",
        width=85,
    )


# The heading will be on the right.

with c2:

    st.caption("")
    st.title("Similar languages & dialects classifier")


# We need to set up session state via st.session_state so that app interactions don't reset the app.

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False



############ TABBED NAVIGATION ############

# First, we're going to create a tabbed navigation for the app via st.tabs()
# tabInfo displays info about the app.
# tabMain displays the main app.

MainTab, InfoTab = st.tabs(["Main", "Info"])

with InfoTab:

    st.subheader("Why this app?")
    st.markdown(
        "[Streamlit](https://streamlit.io) is a Python library that allows the creation of interactive, data-driven web applications in Python."
    )

    st.subheader("Resources")
    st.markdown(
        """
    - [Streamlit Documentation](https://docs.streamlit.io/)
    - [Cheat sheet](https://docs.streamlit.io/library/cheatsheet)
    - [Book](https://www.amazon.com/dp/180056550X) (Getting Started with Streamlit for Data Science)
    """
    )


with MainTab:
    model = load_model()
    # Then, we create a intro text for the app, which we wrap in a st.markdown() widget.

    st.write("")
    st.markdown(
        """

    Distinguish between dialects and similar languages

    """
    )

    st.write("")
    sample_sentence = st.text_input("Enter a sample sentence:", "Jelena moze pisati zadacu")


# Perform BERT inference when the user clicks the button
    if st.button("Run Inference"):
        # Call the inference function
        probs_list = perform_inference(sample_sentence, model)

        # Display probabilities in Streamlit app
        st.write("Probabilities for each class:")
        for i, prob in enumerate(probs_list):
            st.write(f"Class {i}: {prob:.4f}")