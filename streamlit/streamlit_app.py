import streamlit as st
import requests
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from streamlit_tags import st_tags  
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

@st.cache_resource()
def load_model():
    model = torch.load('streamlit/finetuned/pytorch_model.bin', map_location=torch.device('cpu'))
    return model

##The class of the model is needed for loading the model
class BERTMultiClass(torch.nn.Module):
    def __init__(self):
        super(BERTMultiClass, self).__init__()
        #self.l1 = AutoModel.from_pretrained(model)
        self.l1 = model_name
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


class_names = [
    'bs',
    'es-AR',
    'es-ES',
    'es-PE',
    'hr',
    'pt-BR',
    'pt-PT',
    'sr',
    'BS',
    'LU',
    'ZH',
    'BE'
]


def perform_inference(sample_sentence, model):
    tokenizer = AutoTokenizer.from_pretrained("kkristianr/finetuned_bert_multilingual")
    inputs = tokenizer(sample_sentence, return_tensors='pt', truncation=True, padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        model.eval()
        outputs = model(input_ids, attention_mask)

    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    selected_probs = [probabilities[0, class_names.index(class_name)].item() for class_name in selected_classes]

    #probs_list = probabilities.squeeze().tolist()

    return selected_probs

st.set_page_config(
    layout="centered", page_title="Dialect Classifier", page_icon="❄️"
)

## HEADER WITH TWO COLUMNS
c1, c2 = st.columns([0.32, 2])

with c1:
    st.image(
        "streamlit/images/loco.png",
        #"images/loco.png",
        width=85,
    )

with c2:
    st.caption("")
    st.title("Similar languages & dialects classifier")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

## BODY CONTENT
MainTab, InfoTab = st.tabs(["Main", "Info"])

with InfoTab:

    st.subheader("What is this app?")
    st.markdown(
        "This app shows the probabilities of a sentence being in a certain language or dialect. This is achieved by fine-tuning a multilingual BERT model on a dataset of 12 languages and dialects."
    )

    st.subheader("Resources")
    st.markdown(
        """
    - [GitHub code](https://github.com/kkristianr/dialect-identification)
    - [Finetuned tokenizer + model](https://huggingface.co/kkristianr/finetuned_bert_multilingual/)
    """
    )


with MainTab:
    model = load_model()
    st.write("")
    st.markdown(
    """
    Distinguish between dialects and similar languages
    """
    )
    selected_classes = st.multiselect("Select languages to include in the prediction:", class_names, default=class_names)

    st.write("")
    sample_sentence = st.text_input("Enter a sample sentence:", "Jelena moze pisati zadacu")

# Call inference function
    if st.button("Predict language"):
        selected_probs = perform_inference(sample_sentence, model)
        st.markdown("---")        
        st.write("Probabilities for selected languages:")
        table_data = {'Class': selected_classes, 'Probability': selected_probs}
        st.table(table_data)