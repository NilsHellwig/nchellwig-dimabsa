import json
import torch
import gc
import os
import re
import numpy as np


def setup_gpu_environment():
    """Configure GPU environment for optimal performance."""
    print("Configuring GPU environment...")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()


def clear_memory(variables_to_clear=None, verbose=True):
    """
    Clear memory by deleting specified variables and freeing CUDA cache.

    Args:
        variables_to_clear (list, optional): List of variable names to remove from globals.
                                            Defaults to common ML variables.
        verbose (bool, optional): Whether to print memory status. Defaults to True.
    """
    # Default variables to clear if none specified
    if variables_to_clear is None:
        variables_to_clear = ["inputs", "model", "processor", "trainer",
                              "peft_model", "bnb_config"]

    # Delete specified variables if they exist in global scope
    g = globals()
    for var in variables_to_clear:
        if var in g:
            del g[var]

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Second garbage collection pass
        gc.collect()

        # Print memory status if verbose
        if verbose:
            print(
                f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(
                f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


def get_dataset(subtask=3, language="eng", domain="restaurant", split="train"):
    split_phrase = "alltasks" if split == "train" else f"task{subtask}"
    filepath = f"task-dataset/track_a/subtask_{subtask}/{language}/{language}_{domain}_{split}_{split_phrase}.jsonl"

    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())

            # Erstelle das finale Datenformat
            data_entry = {
                "id": json_obj["ID"],
                "text": json_obj["Text"]
            }

            # Wenn Quadruplet existiert, füge Label hinzu
            if "Quadruplet" in json_obj:
                labels = []
                for quad in json_obj["Quadruplet"]:
                    aspect = quad["Aspect"]
                    opinion = quad["Opinion"]
                    category = quad["Category"]
                    va_values = quad["VA"].split("#")
                    valence = va_values[0]
                    arousal = va_values[1]

                    # Subtask 3: aspect, category, opinion, valence, arousal
                    if subtask == 3:
                        labels.append({"aspect_term": aspect, "category": category,
                                      "opinion_term": opinion, "valence": valence, "arousal": arousal})
                    # Subtask 2: aspect, opinion, valence, arousal
                    elif subtask == 2:
                        labels.append(
                            {"aspect_term": aspect, "opinion_term": opinion, "valence": valence, "arousal": arousal})
                    # Subtask 1: aspect, category, opinion, valence, arousal (gleich wie 3)
                    else:
                        labels.append({"aspect_term": aspect, "category": category,
                                      "opinion_term": opinion, "valence": valence, "arousal": arousal})

                data_entry["label"] = labels

            data.append(data_entry)

    return data


aspect_categories = {
    "restaurant": {
        "entity": ["RESTAURANT", "FOOD", "DRINKS", "AMBIENCE", "SERVICE", "LOCATION"],
        "attributes": ["GENERAL", "PRICES", "QUALITY", "STYLE_OPTIONS", "MISCELLANEOUS"]
    },
    "laptop": {
        "entity": ["LAPTOP", "DISPLAY", "KEYBOARD", "MOUSE", "MOTHERBOARD", "CPU", "FANS_COOLING", "PORTS", "MEMORY", "POWER_SUPPLY", "OPTICAL_DRIVES", "BATTERY", "GRAPHICS", "HARD_DISK", "MULTIMEDIA_DEVICES", "HARDWARE", "SOFTWARE", "OS", "WARRANTY", "SHIPPING", "SUPPORT", "COMPANY"],
        "attributes": ["GENERAL", "PRICE", "QUALITY", "DESIGN_FEATURES", "OPERATION_PERFORMANCE", "USABILITY", "PORTABILITY", "CONNECTIVITY", "MISCELLANEOUS"]
    }
}


def get_prompt(text, subtask=3, language="eng", domain="restaurant"):
    prompt = """According to the following sentiment elements definition: 

- The 'aspect term' is the exact word or phrase in the text that represents a specific feature, attribute, or aspect of a product or service that a user may express an opinion about. The aspect term might be 'NULL' for implicit aspect."""

    if subtask == 3:
        entities = aspect_categories[domain]["entity"]
        attributes = aspect_categories[domain]["attributes"]
        entities_str = ", ".join(entities)
        attributes_str = ", ".join(attributes)

        prompt += f"""
- The 'aspect category' refers to the category that the aspect belongs to. It is a combination of an entity and an attribute in the format 'ENTITY#ATTRIBUTE'. The available entities are: {entities_str}. The available attributes are: {attributes_str}."""

    prompt += """
- The 'opinion term' is the exact word or phrase in the text that refers to the sentiment or attitude expressed by a user towards a particular aspect or feature of a product or service. 
- The 'valence' measures the degree of positivity or negativity.
- The 'arousal' measures the intensity of emotion.

A score of 1.00 indicates extremely negative valence or very low arousal, 9.00 indicates extremely positive valence or very high arousal, and 5.00 represents a neutral valence or medium arousal.

For the following text, recognize all sentiment elements with their corresponding aspect terms,"""

    if subtask == 3:
        prompt += " aspect categorys,"

    prompt += " valence, arousal, opinion terms in the following text in the form of a list of tuples"

    if subtask == 3:
        prompt += " [('aspect term', 'aspect category', 'opinion term', 'valence', 'arousal'), ...].\n"""
    else:
        prompt += " [('aspect term', 'opinion term', 'valence', 'arousal'), ...].\n"""

    prompt += f"Text: {text}\nSentiment Elements:"

    return prompt


def convert_tuples_to_output_format(tuples_list, example_id, subtask=3):
    """Convert tuples to the required output format for submission."""
    if subtask == 3:
        # Quadruplet format
        quadruplets = []
        for t in tuples_list:
            if len(t) == 5:
                aspect, category, opinion, valence, arousal = t
                quadruplets.append({
                    "Aspect": aspect,
                    "Category": category,
                    "Opinion": opinion,
                    "VA": f"{valence}#{arousal}"
                })
        return {"ID": example_id, "Quadruplet": quadruplets}
    elif subtask == 2:
        # Triplet format
        triplets = []
        for t in tuples_list:
            if len(t) == 4:
                aspect, opinion, valence, arousal = t
                triplets.append({
                    "Aspect": aspect,
                    "Opinion": opinion,
                    "VA": f"{valence}#{arousal}"
                })
        return {"ID": example_id, "Triplet": triplets}


def set_seed(seed=42):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    # When running on CuDNN backend, make operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed} for reproducibility")


def parse_label_string(label_string, subtask=3):
    """Parse LLM output string back to tuples."""
    label_string = label_string.strip()

    # Remove outer brackets if present
    if label_string.startswith("[") and label_string.endswith("]"):
        label_string = label_string[1:-1]

    # Check if array-based tuples or parentheses-based tuples
    if "[" in label_string:
        array_based = True
    else:
        array_based = False

    if array_based:
        tuples = label_string.split("], [")
    else:
        tuples = label_string.split("), (")

    tuples_list = []
    for t in tuples:
        t = t.strip()

        if array_based:
            if not t.startswith("["):
                t = "[" + t
            if not t.endswith("]"):
                t = t + "]"
        else:
            if not t.startswith("("):
                t = "(" + t
            if not t.endswith(")"):
                t = t + ")"

        if subtask == 2:
            if array_based:
                pattern = r"\['(.+?)', '(.+?)', '(.+?)', '(.+?)'\]"
            else:
                pattern = r"\('(.+?)', '(.+?)', '(.+?)', '(.+?)'\)"
        elif subtask == 3:
            if array_based:
                pattern = r"\['(.+?)', '(.+?)', '(.+?)', '(.+?)', '(.+?)'\]"
            else:
                pattern = r"\('(.+?)', '(.+?)', '(.+?)', '(.+?)', '(.+?)'\)"
        matches = re.match(pattern, t)
        if matches:
            tuples_list.append(matches.groups())

    # Convert to tuples
    tuples_list = [tuple(t) for t in tuples_list]

    return tuples_list


def convert_label_objects_to_tuples(labels, subtask=3):
    """Convert label objects to tuples for training."""
    tuples_list = []
    for label in labels:
        if subtask == 3:
            tuples_list.append((
                label["aspect_term"],
                label["category"],
                label["opinion_term"],
                label["valence"],
                label["arousal"]
            ))
        elif subtask == 2:
            tuples_list.append((
                label["aspect_term"],
                label["opinion_term"],
                label["valence"],
                label["arousal"]
            ))
    return tuples_list
