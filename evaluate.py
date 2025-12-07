from collections import defaultdict
import statistics
import json
import math
key_name = {1: "Aspect_VA", 2: "Triplet", 3: 'Quadruplet'}


def read_jsonl_file(file_path, task=3, data_type='pred'):
    """
    Reads a JSONL file from the specified path and processes each line.

    Args:
        file_path (str): The path to the JSONL file.
        type (str): pred or gold.

    Returns:
        list: A list of dictionaries containing all successfully parsed lines. 
              Returns an empty list if the file does not exist or cannot be read.
    """
    output_key = key_name[task]
    input_key = key_name[3] if (
        data_type == 'gold' and task == 2) else key_name[task]

    data = []
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return data  # Return empty list on failure instead of exiting

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, start=1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    # Parse JSON line
                    json_data = json.loads(line)

                    # Extract basic fields (ID, Text), which are usually required
                    entry = {
                        # Use line number if ID is missing
                        'ID': json_data.get('ID', f"Missing_ID_Line{line_num}"),
                        # Empty string if Text is missing
                        'Text': json_data.get('Text', ''),
                        'Aspect': json_data.get('Aspect', []),
                    }
                    if entry['ID'] == f"Missing_ID_Line{line_num}":
                        exit("Error: ID value is missing!")
                    # Handle Quadruplet field (might not exist or be an empty list)
                    quadruplets = json_data.get(
                        input_key, [])  # Default to empty list
                    if data_type == 'gold' and len(quadruplets) == 0:
                        quadruplets = json_data.get(output_key, [])

                    if isinstance(quadruplets, list):
                        # Process each quadruplet
                        parsed_quadruplets = []
                        for quad in quadruplets:
                            # Ensure quad is a dictionary
                            if not isinstance(quad, dict):
                                print(
                                    f"Warning: Quadruplet at line {line_num} contains non-dictionary item: {quad}")
                                continue

                            # Extract parts of the quadruplet, handle possible missing values
                            aspect = quad.get('Aspect', 'Unknown_Aspect')
                            category = quad.get('Category', 'Unknown_Category')
                            opinion = quad.get('Opinion', 'Unknown_Opinion')
                            # Default value if VA is missing
                            va = quad.get('VA', '0.00#0.00')
                            if va == '0.00#0.00':
                                exit("Error: VA value is missing!")
                            if aspect == 'Unknown_Aspect':
                                exit(
                                    f"Error: {input_key}-Aspect value is missing!")
                            if opinion == 'Unknown_Opinion' and (task == 2 or task == 3):
                                exit(
                                    f"Error: {input_key}-Opinion value is missing!")
                            if category == 'Unknown_Category' and task == 3:
                                exit(
                                    f"Error: {input_key}-Category value is missing!")

                            # Add parsed quadruplet to list
                            parsed_quadruplets.append({
                                'Aspect': aspect.lower(),
                                'Category': category.lower(),
                                'Opinion': opinion.lower(),
                                'VA': va
                            })
                        entry[output_key] = parsed_quadruplets
                    else:
                        # If Quadruplet exists but is not a list (e.g., null or other types), log warning and set to empty list
                        print(
                            f"Warning: Quadruplet at line {line_num} is not a list type: {type(quadruplets)}")
                        entry[output_key] = []

                    # Add parsed entry to data list
                    data.append(entry)

                except json.JSONDecodeError as e:
                    print(f"JSON parsing error at line {line_num}: {e}")
                    # Can choose to skip problematic lines or record errors
                    continue
                except Exception as e:
                    print(
                        f"An unknown error occurred while processing line {line_num}: {e}")
                    continue

    except Exception as e:
        print(f"An error occurred while reading file '{file_path}': {e}")
        return data  # Return empty list on failure instead of exiting

    return data


def print_data_summary(data, task=3):
    """
    Prints a brief summary of the loaded data.

    Args:
        data (list): The list of data entries obtained from the read_jsonl_file function.
    """
    print(f"\n--- Data Summary ---")
    print(f"Successfully loaded {len(data)} valid records.")

    if data:
        print(f"\nSample Data:")
        # Print the first 3 entries as examples
        for i, entry in enumerate(data[:3]):
            print(f"  Record {i+1}:")
            print(f"    ID: {entry['ID']}")
            print(f"    Text: {entry['Text']}")
            print(f"    Quadruplets ({len(entry[key_name[task]])}):")
            for quad in entry[key_name[task]]:
                print(f"      - Aspect: '{quad['Aspect']}', Category: '{quad['Category']}', "
                      f"Opinion: '{quad['Opinion']}', VA: '{quad['VA']}'")
            if i < 2 and len(data) > 3:  # Add separator if there are more records
                print("    ...")


def quadruplet_to_tuple(quad, key_fields):
    """
    Converts a quadruplet dictionary into an immutable tuple for set operations.
    Dynamically determines which fields to include based on the provided key_fields list.

    Args:
        quad (dict): A dictionary representing a quadruplet.
        key_fields (list of str): A list of field names to be used as the unique identifier, 
                                 e.g., ['Aspect', 'Category', 'Opinion'].

    Returns:
        tuple: A tuple containing the values of the specified key_fields.
    """
    # Handle potential missing fields by using placeholders
    values = []
    for field in key_fields:
        # If the field is missing, use 'Unknown_' + field as a placeholder
        # (using a string to maintain type consistency)
        values.append(quad.get(field, f"Unknown_{field}"))
    return tuple(values)


def evaluate_predictions(gold_data, pred_data, task=3):
    """
    Calculate TP, FP, FN, TN, Precision, Recall, F1.

    Args:
        gold_data (list): List of dictionaries containing the ground truth (gold standard) entries.
        pred_data (list): List of dictionaries containing the predicted entries.
        task (int): Task identifier (2 for triplet, 3 for quadruplet).

    Returns:
        dict: A dictionary containing TP, FP, FN, TN, Precision, Recall, F1.
    """
    warning1, warning2 = False, False
    key = key_name[task]
    if not gold_data or not pred_data:
        print("Error: Failed to load one or both data files. Cannot perform evaluation.")
        return None

    # Determine key fields for matching based on task
    key_fields = ['Aspect', 'Opinion'] if task == 2 else [
        'Aspect', 'Opinion', 'Category']

    # Index data by ID for efficient lookup
    gold_dict = {entry['ID']: entry[key] for entry in gold_data}
    pred_dict = {entry['ID']: entry[key] for entry in pred_data}

    # Initialize counters
    # Continuous True Positive (TP_cat minus the sum of thier VA error distances)
    cTP_total = 0.0
    TP_cat = 0         # True Positive (exact match for key fields)
    FP_cat = 0         # False Positive
    FN_cat = 0         # False Negative
    # tn = 0       # True Negatives - typically not calculated directly due to vast, undefined negative space

    # Iterate over all IDs present in either gold or prediction data
    all_ids = set(gold_dict.keys()).union(set(pred_dict.keys()))
    for id_ in all_ids:
        gold_quads = gold_dict.get(id_, [])
        pred_quads = pred_dict.get(id_, [])
        matched_pred_num = 0  # The number of matched predictions
        for gold_quad in gold_quads:
            # List to store all cTP scores for matching predictions for the current gold quadruplet
            all_cTP_scores = []  # Reset for each gold quadruplet
            gold_match_key = quadruplet_to_tuple(gold_quad, key_fields)

            for pred_quad in pred_quads:  # Iterate through predictions
                pred_match_key = quadruplet_to_tuple(pred_quad, key_fields)

                # Check if key fields match
                if gold_match_key == pred_match_key:
                    # Parse VA string
                    try:
                        gold_v_str, gold_a_str = gold_quad['VA'].split('#')
                        pred_v_str, pred_a_str = pred_quad['VA'].split('#')
                        gold_v, gold_a = float(gold_v_str), float(gold_a_str)
                        pred_v, pred_a = float(pred_v_str), float(pred_a_str)
                    except ValueError as e:
                        print(
                            f"Warning: Failed to parse VA values for ID '{id_}'. Error: {e}")
                        continue

                    if pred_a < 1.0 or pred_a > 9.0 or pred_v < 1.0 or pred_v > 9.0:
                        warning1 = True
                        all_cTP_scores.append(0)
                        continue

                    # --- Calculate Euclidean distance and cTP score ---
                    # Calculate Euclidean distance between (V, A) points
                    va_euclid = math.sqrt(
                        (pred_v - gold_v)**2 + (pred_a - gold_a)**2)
                    # cTP score = 1 - distance, but cannot be less than 0 (due to numerical error)
                    # The maximum possible distance in [1,9]x[1,9] space is sqrt(128), so we cap the distance used in score calculation.
                    D_max = math.sqrt(128)
                    cTP_t = max(0.0, 1.0 - (va_euclid / D_max))

                    # print("======================="*5)
                    # print("id: ",id_)
                    # print("gold: ",gold_quad)
                    # print("pred: ",pred_quad)
                    # print(pred_v - gold_v,pred_a - gold_a)
                    # print("dist:   ",va_euclid)
                    # print("cTP_t: ", cTP_t)
                    # print("======================="*5)
                    # input()
                    all_cTP_scores.append(cTP_t)

            if len(all_cTP_scores) > 1:
                warning2 = True
                FN_cat = FN_cat + 1
                cTP_total += 0
            elif len(all_cTP_scores) == 1:
                matched_pred_num += 1
                TP_cat += 1
                cTP_total += all_cTP_scores[0]
            else:
                # If no matching prediction, contribution is 0
                FN_cat = FN_cat + 1
        FP_cat += (len(pred_quads)-matched_pred_num)

    # Calculate cPrecision, cRecall, cF1 using cTP_total
    cPrecision = cTP_total / \
        (TP_cat + FP_cat) if (TP_cat + FP_cat) > 0 else 0.0
    cRecall = cTP_total / (TP_cat + FN_cat) if (TP_cat + FN_cat) > 0 else 0.0
    cF1 = 2 * cPrecision * cRecall / \
        (cPrecision + cRecall) if (cPrecision + cRecall) > 0 else 0.0
    if warning1:
        print(f"Warning: Some predicted values are out of the numerical range.")
    if warning2:
        print(f"Warning: Duplicate prediction exists.")

    return {
        'TP': cTP_total,
        'FP': FP_cat,
        'FN': FN_cat,
        'cPrecision': cPrecision,
        'cRecall': cRecall,
        'cF1': cF1
    }


################################################################################################################################################################################################################################################################################

from scipy import stats
from itertools import combinations
from helper import *
import numpy as np
import pandas as pd

N_SPLITS = 1  # Anzahl der 80/20 Splits für train_split
NUM_PRED_SC = 5 # Anzahl der Vorhersagen für Self-Consistency
RUN_SEED = 0  # allgemeine Seed für Reproduzierbarkeit
COLUMNS = ["no_sc_no_guided_12b", "sc_5_12b", "sc_15_12b", "no_sc_no_guided_27b", "sc_5_27b", "sc_15_27b"]

COLUMN_CONFIG = {
    "no_sc_no_guided_12b": ("unsloth/gemma-3-12b-it-bnb-4bit", None, "no_sc_no_guided"),
    "sc_5_12b": ("unsloth/gemma-3-12b-it-bnb-4bit", 5, "sc_no_guided"),
    "sc_15_12b": ("unsloth/gemma-3-12b-it-bnb-4bit", 15, "sc_no_guided"),
    "no_sc_no_guided_27b": ("unsloth/gemma-3-27b-it-bnb-4bit", None, "no_sc_no_guided"),
    "sc_5_27b": ("unsloth/gemma-3-27b-it-bnb-4bit", 5, "sc_no_guided"),
    "sc_15_27b": ("unsloth/gemma-3-27b-it-bnb-4bit", 15, "sc_no_guided"),
}

# Valid combinations of (language, domain) that have data
VALID_LANGUAGES_DOMAINS = [
    ("eng", "restaurant"),
    ("eng", "laptop"),
    ("jpn", "hotel"),
    ("rus", "restaurant"),
    ("tat", "restaurant"),
    ("ukr", "restaurant"),
    ("zho", "restaurant"),
    ("zho", "laptop"),
]


def load_predictions(subtask, language, domain, split_idx, llm, strategy="train_split", guidance=True, self_consistency=True, run_idx=0):
    llm_name_formatted = llm.replace("/", "_")

    guidance_str = "with_guidance" if guidance else "no_guidance"
    temp_str = "_temp0.8" if self_consistency else "_temp0"
    run_str = f"_run{run_idx}" if self_consistency else ""
    split_idx_str = f"_{split_idx}" if strategy == "train_split" else ""

    path = f"results/results_{strategy}/{llm_name_formatted}/{subtask}_{language}_{domain}_{RUN_SEED}{split_idx_str}{temp_str}_{guidance_str}{run_str}.jsonl"

    predictions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            predictions.append(data)
    return predictions


def load_ground_truth(subtask, language, domain):
    # task-dataset/track_a/subtask_2/eng/eng_laptop_train_alltasks.jsonl
    path = f"task-dataset/track_a/subtask_{subtask}/{language}/{language}_{domain}_train_alltasks.jsonl"
    ground_truth = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            ground_truth.append(data)
    return ground_truth


def filter_predictions(predictions, ground_truth):
    labels_filtered = []

    preds_dict = {pred['ID']: pred for pred in predictions}
    for label in ground_truth:
        if label['ID'] in preds_dict:
            labels_filtered.append(label)
    return labels_filtered


def merge_predictions(predictions, subtask, min_votes=3):
    counter = defaultdict(list)

    # Annahme: alle Runs haben dieselbe ID
    final_id = predictions[0]["ID"]

    for run in predictions:
        key_list = "Quadruplet" if subtask == 3 else "Triplet"
        for quad in run[key_list]:
            # Key abhängig vom Subtask
            if subtask == 3:
                key = (quad["Aspect"], quad["Category"], quad["Opinion"])
            else:  # subtask == 2
                key = (quad["Aspect"], quad["Opinion"])

            # Valence/Arousal parsen
            valence_str, arousal_str = quad["VA"].split("#")
            valence = float(valence_str)
            arousal = float(arousal_str)

            counter[key].append((valence, arousal))

    merged = []

    # Verarbeitung der aggregierten Quadruplets
    for key, values in counter.items():
        if len(values) >= min_votes:
            mean_valence = statistics.mean(v[0] for v in values)
            mean_arousal = statistics.mean(v[1] for v in values)

            if subtask == 3:
                aspect, category, opinion = key
                merged.append({
                    "Aspect": aspect,
                    "Category": category,
                    "Opinion": opinion,
                    "VA": f"{mean_valence:.2f}#{mean_arousal:.2f}"
                })
            else:
                aspect, opinion = key
                merged.append({
                    "Aspect": aspect,
                    "Opinion": opinion,
                    "VA": f"{mean_valence:.2f}#{mean_arousal:.2f}"
                })

    # Finaler Output mit passendem Key
    if subtask == 2:
        return {
            "ID": final_id,
            "Triplet": merged
        }
    else:
        return {
            "ID": final_id,
            "Quadruplet": merged
        }


def get_performance(language, domain, subtask, strategy, llm="unsloth/gemma-3-27b-it-bnb-4bit", num_preds_sc=NUM_PRED_SC):
    labels = load_ground_truth(subtask, language, domain)

    results = []
    n_splits = N_SPLITS if strategy == "train_split" else 1

    for split_idx in range(n_splits):
        preds_no_sc_guided = load_predictions(
            subtask, language, domain, split_idx=split_idx, llm=llm, strategy=strategy, guidance=True, self_consistency=False)
        preds_no_sc_no_guided = load_predictions(
            subtask, language, domain, split_idx=split_idx, llm=llm, strategy=strategy, guidance=False, self_consistency=False)

        preds_sc_guided = []
        all_preds_guided = [
            load_predictions(subtask, language, domain, split_idx=split_idx,
                             llm=llm, strategy=strategy, guidance=True, self_consistency=True, run_idx=i)
            for i in range(num_preds_sc)
        ]
        for k in range(len(all_preds_guided[0])):
            merged_quads = merge_predictions(
                [all_preds_guided[i][k] for i in range(num_preds_sc)], subtask=subtask, min_votes=num_preds_sc//2+1)
            preds_sc_guided.append(merged_quads)

        preds_sc_no_guided = []
        all_preds_no_guided = [
            load_predictions(subtask, language, domain, split_idx=split_idx,
                             llm=llm, strategy=strategy, guidance=False, self_consistency=True, run_idx=i)
            for i in range(num_preds_sc)
        ]
        for k in range(len(all_preds_no_guided[0])):
            merged_quads = merge_predictions(
                [all_preds_no_guided[i][k] for i in range(num_preds_sc)], subtask=subtask, min_votes=num_preds_sc//2+1)
            preds_sc_no_guided.append(merged_quads)

        labels_filtered = filter_predictions(preds_no_sc_guided, labels)
        
        # convert key Quadruplet to Triplet if subtask==2 for labels_filtered
        if subtask == 2:
            for entry in labels_filtered:
                quads = entry.pop("Quadruplet", [])
                triplets = []
                for quad in quads:
                    triplets.append({
                        "Aspect": quad["Aspect"],
                        "Opinion": quad["Opinion"],
                        "VA": quad["VA"]
                    })
                entry["Triplet"] = triplets

        results.append({
            "no_sc_guided": evaluate_predictions(labels_filtered, preds_no_sc_guided, task=subtask),
            "no_sc_no_guided": evaluate_predictions(labels_filtered, preds_no_sc_no_guided, task=subtask),
            "sc_guided": evaluate_predictions(labels_filtered, preds_sc_guided, task=subtask),
            "sc_no_guided": evaluate_predictions(labels_filtered, preds_sc_no_guided, task=subtask),
        })

    # calculate average over splits
    if strategy == "train_split":
        avg_results = {}
        for key in results[0].keys():
            avg_results[key] = {}
            for metric in results[0][key].keys():
                avg_results[key][metric] = statistics.mean(
                    result[key][metric] for result in results)
        return avg_results
    else:
        return results[0], {
            "no_sc_guided": preds_no_sc_guided,
            "no_sc_no_guided": preds_no_sc_no_guided,
            "sc_guided": preds_sc_guided,
            "sc_no_guided": preds_sc_no_guided,
        }

def get_key_of_best_strategy(lang, domain, df):
    strategies = {}
    for column in COLUMNS:
        strategies[column] = df.loc[(df["Language"] == language_mapping[lang]) & (df["Domain"] == domain_mapping[domain]), column].values[0]
    
    # throw error if any value is nan or np.float64(nan)
    for key in strategies:
        if pd.isna(strategies[key]):
            strategies[key] = None
    if all(value is None for value in strategies.values()):
        raise FileNotFoundError(
            f"No performance data found for language: {lang}, domain: {domain}")

    # Get strategy with highest score
    best_strategy = max(
        strategies, key=lambda k: strategies[k] if strategies[k] is not None else -1)

    return best_strategy

def get_performance_tabular(table_metric, table_subtask, strategy="train_split"):
    table = defaultdict(lambda: defaultdict(dict))

    for language, domain in VALID_LANGUAGES_DOMAINS:
        for col_name, (llm, num_preds_sc, result_key) in COLUMN_CONFIG.items():
            try:
                if num_preds_sc is None:
                    result = get_performance(language, domain, table_subtask, strategy, llm=llm)
                else:
                    result = get_performance(language, domain, table_subtask, strategy, llm=llm, num_preds_sc=num_preds_sc)
                table[language][domain][col_name] = result[result_key][table_metric]
            except FileNotFoundError:
                table[language][domain][col_name] = None

    df_rows = []
    for language, domain in VALID_LANGUAGES_DOMAINS:
        row = {"Language": language_mapping[language], "Domain": domain_mapping[domain]}
        row.update({col: table[language][domain][col] for col in COLUMNS})
        df_rows.append(row)

    df = pd.DataFrame(df_rows)

    # Add AVG row
    avg_row = {"Language": "AVG", "Domain": ""}
    avg_row.update({col: df[col].mean(skipna=True) for col in COLUMNS})

    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    return df

def run_significance_tests(df, subtask):
    from IPython.display import display
    
    print(f"\n{'='*60}")
    print(f"SIGNIFICANCE TESTS - Subtask {subtask}")
    print(f"{'='*60}")
    
    df_clean = df[df["Language"] != "AVG"].copy()
    groups = {col: df_clean[col].dropna().values for col in COLUMNS}
    
    print("\n--- Shapiro-Wilk Normalitätstests ---")
    normality_results = {}
    for col, values in groups.items():
        if len(values) >= 3:
            stat, p = stats.shapiro(values)
            normality_results[col] = p > 0.05
            print(f"{col}: W={stat:.4f}, p={p:.4f} {'(normal)' if p > 0.05 else '(nicht normal)'}")
    
    all_normal = all(normality_results.values())
    print(f"\nAlle normalverteilt: {all_normal}")
    
    print("\n--- Omnibus-Test (Unterschied zwischen allen 6 Gruppen) ---")
    valid_groups = [v for v in groups.values() if len(v) >= 2]
    
    if all_normal:
        stat, p = stats.f_oneway(*valid_groups)
        test_name = "ANOVA"
    else:
        stat, p = stats.friedmanchisquare(*valid_groups)
        test_name = "Friedman"
    
    print(f"{test_name}: Statistik={stat:.4f}, p={p:.6f}")
    print(f"Signifikanter Unterschied zwischen Gruppen: {'Ja' if p < 0.05 else 'Nein'}")
    
    print("\n--- Paarweise Vergleiche (Bonferroni-Holm) ---")
    pairs = list(combinations(COLUMNS, 2))
    pairwise_results = []
    
    for col1, col2 in pairs:
        vals1, vals2 = groups[col1], groups[col2]
        min_len = min(len(vals1), len(vals2))
        if min_len < 2:
            continue
        vals1, vals2 = vals1[:min_len], vals2[:min_len]
        
        if all_normal:
            stat, p = stats.ttest_rel(vals1, vals2)
        else:
            stat, p = stats.wilcoxon(vals1, vals2)
        
        pairwise_results.append((col1, col2, stat, p))
    
    pairwise_results.sort(key=lambda x: x[3])
    n_tests = len(pairwise_results)
    
    rows = []
    sig_stopped = False
    for i, (col1, col2, stat, p) in enumerate(pairwise_results):
        adj_alpha = 0.05 / (n_tests - i)
        if not sig_stopped and p >= adj_alpha:
            sig_stopped = True
        p_adj = p * (n_tests - i)
        rows.append({
            "Vergleich": f"{col1} vs {col2}",
            "p": p,
            "p adj.": min(p_adj, 1.0),
            "adj. α": adj_alpha,
            "Signifikant": "Nein" if sig_stopped else ("Ja" if p < adj_alpha else "Nein")
        })
    
    df_results = pd.DataFrame(rows)
    
    display(df_results)
    return df_results

def format_table_parameter_tuning_for_latex(df):
    df_latex = df.copy()

    # Format numbers as percentages with two decimal places and *100 multiplication
    for col in COLUMNS:
        df_latex[col] = df_latex[col].apply(lambda x: f"{x*100:.2f}" if pd.notna(x) else "N/A")
    
    # mark the best strategy in bold for each row
    for index, row in df_latex.iterrows():
        strategies = {}
        for column in COLUMNS:
            strategies[column] = float(row[column]) if row[column] != "N/A" else -1
        best_strategy = max(strategies, key=strategies.get)
        if strategies[best_strategy] != -1:
            df_latex.at[index, best_strategy] = f"\\textbf{{{df_latex.at[index, best_strategy]}}}"    
    
    return df_latex


# load muster/parameter_optimization.txt
def get_tabular_parameter_optimization(values):
    with open(os.path.join("plots", "muster", "parameter_optimization.txt"), "r") as f:
       parameter_optimization_content = f.read()
    # go from xxxx to xxxx and insert values
    for i, value in enumerate(values):
        parameter_optimization_content = parameter_optimization_content.replace(f"xxxx", f"{value}", 1)
    return parameter_optimization_content

