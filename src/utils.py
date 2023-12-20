import torch
import itertools
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

def pipeline(model_path, vocab_path):
    """
    Create a pipeline for performing inference on a sentence.

    Parameters:
    - model_path (str): Path to the model.
    - vocab_path (str): Path to the vocabulary.

    Returns:
    - A pipeline object.
    """
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(vocab_path)

    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

    return classifier

if __name__ == "__main__":
    model_path = './models/pytorch_bertic_3.bin'
    vocab_path = './models/vocab_bertic_3.bin'

    bert_pipeline = pipeline(model_path, vocab_path)

    # Example: Perform inference on a croatian sentence
    test_sentence = "Rečenica je jedinica kojom se prenosi potpuna obavijest. Sastoji se od najmanje jedne riječi i s drugim rečenicama može tvoriti tekst. Riječi u rečenici povezane su gramatičkim odnosima. Zato riječi mačka i dvorište u lajati mijaukati pas ne tvore rečenicu, a riječi u dvorištu mačka mijauče i pas laje tvore."
    result = bert_pipeline(test_sentence)

    # Print the result
    print(result)



def grid_search(config, hyperparameters):
    """
    Perform grid search over the specified hyperparameters.

    Parameters:
    - config (dict): The base configuration dictionary.
    - hyperparameters (dict): A dictionary where keys are hyperparameter names
                             and values are lists of values to search over.

    Returns:
    - List of dictionaries, each containing a specific configuration.
    """
    # Generate all possible combinations of hyperparameters
    grid_search_configs = list(itertools.product(*hyperparameters.values()))

    # Perform grid search
    all_configs = []
    for hyperparam_values in grid_search_configs:
        # Create a copy of the base configuration
        current_config = config.copy()

        # Update the current configuration with the specific hyperparameter values
        for hyperparam_name, value in zip(hyperparameters.keys(), hyperparam_values):
            current_config[hyperparam_name] = value

        all_configs.append(current_config)

    return all_configs

