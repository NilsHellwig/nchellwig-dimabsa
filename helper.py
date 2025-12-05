import json
import torch
import gc
import os
import re
import numpy as np
import logging
from datetime import datetime

# Setup logger
logger = logging.getLogger('training')
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler('logs.txt', mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)

# Add handler to logger
if not logger.handlers:
    logger.addHandler(file_handler)


def setup_gpu_environment():
    """Configure GPU environment for optimal performance."""
    logger.info("Configuring GPU environment...")
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
                              "peft_model", "bnb_config", "fastmodel",
                              "tokenizer", "base_model", "llm"]

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
            logger.info(
                f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(
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
    },
    "hotel": {
        "entity": ["HOTEL", "ROOMS", "FACILITIES", "ROOM_AMENITIES", "SERVICE", "LOCATION", "FOOD_DRINKS"],
        "attributes": ["GENERAL", "PRICE", "COMFORT", "CLEANLINESS", "QUALITY", "DESIGN_FEATURES", "STYLE_OPTIONS", "MISCELLANEOUS"]
    },
    "finance": {
        "entity": ["MARKET", "COMPANY", "BUSINESS", "PRODUCT"],
        "attributes": ["GENERAL", "SALES", "PROFIT", "AMOUNT", "PRICE", "COST"]
    }
}


def get_unique_aspect_categories(domain):
    # return list of all combinations of entity#attribute for the given domain
    entities = aspect_categories[domain]["entity"]
    attributes = aspect_categories[domain]["attributes"]
    combinations = [
        f"{entity}#{attribute}" for entity in entities for attribute in attributes]
    # sort alphabetically
    combinations.sort()
    return combinations


def get_prompt(text, subtask=3, language="eng", domain="restaurant"):
    # Multilingual prompt templates
    prompts = {
        "eng": {
            "intro": "According to the following sentiment elements definition:",
            "aspect_term": "The 'aspect term' is the exact word or phrase in the text that represents a specific feature, attribute, or aspect of a product or service that a user may express an opinion about. The aspect term might be 'NULL' for implicit aspect.",
            "category_intro": "The 'aspect category' refers to the category that the aspect belongs to. It is a combination of an entity and an attribute in the format 'ENTITY#ATTRIBUTE'. The available entities are: {entities}. The available attributes are: {attributes}.",
            "opinion_term": "The 'opinion term' is the exact word or phrase in the text that refers to the sentiment or attitude expressed by a user towards a particular aspect or feature of a product or service. The opinion term might be 'NULL' if no explicit opinion expression is present.",
            "valence": "The 'valence score' measures the degree of positivity or negativity.",
            "arousal": "The 'arousal score' measures the intensity of emotion.",
            "scale": "A score of 1.00 indicates extremely negative valence or very low arousal, 9.00 indicates extremely positive valence or very high arousal, and 5.00 represents a neutral valence or medium arousal. Both valence score and arousal score must be rounded to exactly two decimal places.",
            "task_subtask3": "For the following text, recognize all sentiment elements with their corresponding aspect terms, aspect categorys, valence score, arousal score, opinion terms in the following text in the form of a list of tuples [('aspect term', 'ENTITY#ATTRIBUTE', 'opinion term', 'valence score', 'arousal score'), ...].",
            "task_subtask2": "For the following text, recognize all sentiment elements with their corresponding aspect terms, valence score, arousal score, opinion terms in the following text in the form of a list of tuples [('aspect term', 'opinion term', 'valence score', 'arousal score'), ...].",
            "text_label": "Text:",
            "elements_label": "Sentiment Elements:"
        },
        "jpn": {
            "intro": "次の感情要素の定義に従って:",
            "aspect_term": "「'aspect term'」は、ユーザーが意見を述べる可能性のある製品やサービスの特定の特徴、属性、または側面を表すテキスト内の正確な単語またはフレーズです。暗黙のアスペクトの場合、'aspect term'は「NULL」である可能性があります。",
            "category_intro": "「'aspect category'」は、アスペクトが属するカテゴリを指します。これは「ENTITY#ATTRIBUTE」の形式でエンティティと属性の組み合わせです。利用可能なエンティティは: {entities}。利用可能な属性は: {attributes}。",
            "opinion_term": "「'opinion term'」は、製品やサービスの特定の側面や特徴に対してユーザーが表現する感情や態度を指すテキスト内の正確な単語またはフレーズです。明示的な意見表現がない場合、'opinion term'は「NULL」である可能性があります。",
            "valence": "「'valence score'」は、ポジティブまたはネガティブの度合いを測定します。",
            "arousal": "「'arousal score'」は、感情の強度を測定します。",
            "scale": "1.00のスコアは非常にネガティブな'valence score'または非常に低い'arousal score'を示し、9.00は非常にポジティブな'valence score'または非常に高い'arousal score'を示し、5.00は中立の'valence score'または中程度の'arousal score'を表します。'valence score'と'arousal score'は両方とも正確に小数点以下2桁に四捨五入する必要があります。",
            "task_subtask3": "次のテキストについて、対応する'aspect term'、'aspect category'、'opinion term'、'valence score'、'arousal score'を含むすべての感情要素をタプルのリストの形式で認識してください [('aspect term', 'ENTITY#ATTRIBUTE', 'opinion term', 'valence score', 'arousal score'), ...]。",
            "task_subtask2": "次のテキストについて、対応する'aspect term'、'opinion term'、'valence score'、'arousal score'を含むすべての感情要素をタプルのリストの形式で認識してください [('aspect term', 'opinion term', 'valence score', 'arousal score'), ...]。",
            "text_label": "テキスト:",
            "elements_label": "感情要素:"
        },
        "rus": {
            "intro": "Согласно следующему определению элементов настроения:",
            "aspect_term": "«'aspect term'» - это точное слово или фраза в тексте, которое представляет собой конкретную особенность, атрибут или аспект продукта или услуги, о которых пользователь может выразить мнение. 'aspect term' может быть «NULL» для неявного аспекта.",
            "category_intro": "«'aspect category'» относится к категории, к которой принадлежит аспект. Это комбинация сущности и атрибута в формате «ENTITY#ATTRIBUTE». Доступные сущности: {entities}. Доступные атрибуты: {attributes}.",
            "opinion_term": "«'opinion term'» - это точное слово или фраза в тексте, которое относится к настроению или отношению, выраженному пользователем к определенному аспекту или особенности продукта или услуги. 'opinion term' может быть «NULL», если явное выражение мнения отсутствует.",
            "valence": "«'valence score'» измеряет степень позитивности или негативности.",
            "arousal": "«'arousal score'» измеряет интенсивность эмоции.",
            "scale": "Оценка 1.00 указывает на крайне негативный 'valence score' или очень низкий 'arousal score', 9.00 указывает на крайне позитивный 'valence score' или очень высокий 'arousal score', а 5.00 представляет собой нейтральный 'valence score' или средний 'arousal score'. И 'valence score', и 'arousal score' должны быть округлены до ровно двух десятичных знаков.",
            "task_subtask3": "Для следующего текста распознайте все элементы настроения с соответствующими 'aspect term', 'aspect category', 'opinion term', 'valence score', 'arousal score' в следующем тексте в виде списка кортежей [('aspect term', 'ENTITY#ATTRIBUTE', 'opinion term', 'valence score', 'arousal score'), ...].",
            "task_subtask2": "Для следующего текста распознайте все элементы настроения с соответствующими 'aspect term', 'opinion term', 'valence score', 'arousal score' в следующем тексте в виде списка кортежей [('aspect term', 'opinion term', 'valence score', 'arousal score'), ...].",
            "text_label": "Текст:",
            "elements_label": "Элементы настроения:"
        },
        "tat": {
            "intro": "Түбәндәге хис-кичерешләр элементлары билгеләмәсенә туры килеп:",
            "aspect_term": "«'aspect term'» - текстта кулланучы фикер белдерә алырлык продукт яки хезмәтнең билгеле бер үзенчәлеген, атрибутын яки аспектын күрсәтүче төгәл сүз яки фраза. 'aspect term' ачык булмаган аспект өчен «NULL» булырга мөмкин.",
            "category_intro": "«'aspect category'» аспект караган категориягә мөрәҗәгать итә. Бу «ENTITY#ATTRIBUTE» форматындагы субъект һәм атрибутның берләшмәсе. Мөмкин субъектлар: {entities}. Мөмкин атрибутлар: {attributes}.",
            "opinion_term": "«'opinion term'» - текстта кулланучының продукт яки хезмәтнең билгеле бер аспекты яки үзенчәлеге турында белдергән хисләренә яки мөнәсәбәтенә мөрәҗәгать итүче төгәл сүз яки фраза. Ачык фикер белдерү булмаса, 'opinion term' «NULL» булырга мөмкин.",
            "valence": "«'valence score'» уңай яки тискәре дәрәҗәне үлчи.",
            "arousal": "«'arousal score'» хисләр интенсивлыгын үлчи.",
            "scale": "1.00 балл бик тискәре 'valence score' яки бик түбән 'arousal score' күрсәтә, 9.00 бик уңай 'valence score' яки бик югары 'arousal score' күрсәтә, ә 5.00 нейтраль 'valence score' яки уртача 'arousal score' белдерә. 'valence score' һәм 'arousal score' икесе дә төгәл ике утырмалы саннарга түгәрәкләнергә тиеш.",
            "task_subtask3": "Түбәндәге текст өчен барлык хис-кичерешләр элементларын тиешле 'aspect term', 'aspect category', 'opinion term', 'valence score', 'arousal score' белән кортежлар исемлеге рәвешендә тану [('aspect term', 'ENTITY#ATTRIBUTE', 'opinion term', 'valence score', 'arousal score'), ...].",
            "task_subtask2": "Түбәндәге текст өчен барлык хис-кичерешләр элементларын тиешле 'aspect term', 'opinion term', 'valence score', 'arousal score' белән кортежлар исемлеге рәвешендә тану [('aspect term', 'opinion term', 'valence score', 'arousal score'), ...].",
            "text_label": "Текст:",
            "elements_label": "Хис-кичерешләр элементлары:"
        },
        "ukr": {
            "intro": "Відповідно до наступного визначення елементів настрою:",
            "aspect_term": "«'aspect term'» - це точне слово або фраза в тексті, яке представляє конкретну особливість, атрибут або аспект продукту чи послуги, про які користувач може висловити думку. 'aspect term' може бути «NULL» для неявного аспекту.",
            "category_intro": "«'aspect category'» відноситься до категорії, до якої належить аспект. Це комбінація сутності та атрибута у форматі «ENTITY#ATTRIBUTE». Доступні сутності: {entities}. Доступні атрибути: {attributes}.",
            "opinion_term": "«'opinion term'» - це точне слово або фраза в тексті, яке відноситься до настрою або ставлення, висловленого користувачем до певного аспекту або особливості продукту чи послуги. 'opinion term' може бути «NULL», якщо явне вираження думки відсутнє.",
            "valence": "«'valence score'» вимірює ступінь позитивності або негативності.",
            "arousal": "«'arousal score'» вимірює інтенсивність емоції.",
            "scale": "Оцінка 1.00 вказує на надзвичайно негативний 'valence score' або дуже низький 'arousal score', 9.00 вказує на надзвичайно позитивний 'valence score' або дуже високий 'arousal score', а 5.00 представляє нейтральний 'valence score' або середній 'arousal score'. І 'valence score', і 'arousal score' повинні бути округлені до рівно двох десяткових знаків.",
            "task_subtask3": "Для наступного тексту розпізнайте всі елементи настрою з відповідними 'aspect term', 'aspect category', 'opinion term', 'valence score', 'arousal score' у наступному тексті у вигляді списку кортежів [('aspect term', 'ENTITY#ATTRIBUTE', 'opinion term', 'valence score', 'arousal score'), ...].",
            "task_subtask2": "Для наступного тексту розпізнайте всі елементи настрою з відповідними 'aspect term', 'opinion term', 'valence score', 'arousal score' у наступному тексті у вигляді списку кортежів [('aspect term', 'opinion term', 'valence score', 'arousal score'), ...].",
            "text_label": "Текст:",
            "elements_label": "Елементи настрою:"
        },
        "zho": {
            "intro": "根据以下情感元素定义：",
            "aspect_term": "「'aspect term'」是文本中代表用户可能表达意见的产品或服务的特定特征、属性或方面的确切单词或短语。对于隐含方面，'aspect term'可能为'NULL'。",
            "category_intro": "「'aspect category'」是指方面所属的类别。它是'ENTITY#ATTRIBUTE'格式的实体和属性的组合。可用实体为：{entities}。可用属性为：{attributes}。",
            "opinion_term": "「'opinion term'」是文本中指用户对产品或服务的特定方面或特征所表达的情感或态度的确切单词或短语。如果没有明确的意见表达，'opinion term'可能为'NULL'。",
            "valence": "「'valence score'」衡量积极或消极的程度。",
            "arousal": "「'arousal score'」衡量情绪的强度。",
            "scale": "1.00的分数表示极度负面的'valence score'或非常低的'arousal score'，9.00表示极度正面的'valence score'或非常高的'arousal score'，5.00表示中性的'valence score'或中等的'arousal score'。'valence score'和'arousal score'都必须精确地四舍五入到两位小数。",
            "task_subtask3": "对于以下文本，识别具有相应'aspect term'、'aspect category'、'opinion term'、'valence score'、'arousal score'的所有情感元素，以元组列表的形式 [('aspect term', 'ENTITY#ATTRIBUTE', 'opinion term', 'valence score', 'arousal score'), ...]。",
            "task_subtask2": "对于以下文本，识别具有相应'aspect term'、'opinion term'、'valence score'、'arousal score'的所有情感元素，以元组列表的形式 [('aspect term', 'opinion term', 'valence score', 'arousal score'), ...]。",
            "text_label": "文本：",
            "elements_label": "情感元素："
        }
    }

    # Get language-specific prompts, fallback to English
    lang_prompts = prompts.get(language, prompts["eng"])

    prompt = lang_prompts["intro"] + "\n\n"
    prompt += "- " + lang_prompts["aspect_term"] + "\n"

    if subtask == 3:
        entities = aspect_categories[domain]["entity"]
        attributes = aspect_categories[domain]["attributes"]
        entities_str = ", ".join(entities)
        attributes_str = ", ".join(attributes)

        prompt += "- " + lang_prompts["category_intro"].format(
            entities=entities_str, attributes=attributes_str) + "\n"

    prompt += "- " + lang_prompts["opinion_term"] + "\n"
    prompt += "- " + lang_prompts["valence"] + "\n"
    prompt += "- " + lang_prompts["arousal"] + "\n\n"
    prompt += lang_prompts["scale"] + "\n\n"

    if subtask == 3:
        prompt += lang_prompts["task_subtask3"] + "\n"
    else:
        prompt += lang_prompts["task_subtask2"] + "\n"

    prompt += f"{lang_prompts['text_label']} {text}\n{lang_prompts['elements_label']}"

    return prompt


def convert_tuples_to_output_format(tuples_list, example_id, subtask=3, disable_null_aspect=True):
    """Convert tuples to the required output format for submission."""
    if subtask == 3:
        # Quadruplet format - deduplicate based on aspect_term, category, opinion_term
        quadruplets = []
        seen_keys = set()
        for t in tuples_list:
            if len(t) == 5:
                aspect, category, opinion, valence, arousal = t
                # Clamp valence and arousal to [1.00, 9.00] range
                valence = max(1.00, min(9.00, float(valence)))
                arousal = max(1.00, min(9.00, float(arousal)))
                # valence and arousal must have two decimal places
                valence = f"{valence:.2f}"
                arousal = f"{arousal:.2f}"

                aspect = aspect.replace("\\'", "'")
                opinion = opinion.replace("\\'", "'")

                # Skip tuples with NULL aspect or opinion if disable_null_aspect is True
                if disable_null_aspect and (aspect == "NULL" or opinion == "NULL"):
                    continue

                # Create deduplication key
                key = (aspect, category, opinion)
                if key not in seen_keys:
                    seen_keys.add(key)
                    quadruplets.append({
                        "Aspect": aspect,
                        "Category": category,
                        "Opinion": opinion,
                        "VA": f"{valence}#{arousal}"
                    })
        return {"ID": example_id, "Quadruplet": quadruplets}
    elif subtask == 2:
        # Triplet format - deduplicate based on aspect_term, opinion_term
        triplets = []
        seen_keys = set()
        for t in tuples_list:
            if len(t) == 4:
                aspect, opinion, valence, arousal = t
                aspect = aspect.replace("\\'", "'")
                opinion = opinion.replace("\\'", "'")

                # Skip tuples with NULL aspect or opinion if disable_null_aspect is True
                if disable_null_aspect and (aspect == "NULL" or opinion == "NULL"):
                    continue

                # Clamp valence and arousal to [1.00, 9.00] range
                valence = max(1.00, min(9.00, float(valence)))
                arousal = max(1.00, min(9.00, float(arousal)))
                # valence and arousal must have two decimal places
                valence = f"{valence:.2f}"
                arousal = f"{arousal:.2f}"
                # Create deduplication key
                key = (aspect, opinion)
                if key not in seen_keys:
                    seen_keys.add(key)
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
    logger.info(f"Random seed set to {seed} for reproducibility")


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

        # Define patterns that match get_regex_pattern_tuple
        if subtask == 2:
            # aspect_term, opinion_term, valence, arousal
            if array_based:
                # Allow NULL for aspect_term and opinion_term, specific pattern for VA (0, 1 or 2 decimal places)
                pattern = r"\['(NULL|.+?)', '(NULL|.+?)', '?([1-9](?:\.[0-9][0-9]?)?)'?, '?([1-9](?:\.[0-9][0-9]?)?)'?\]"
            else:
                pattern = r"\('(NULL|.+?)', '(NULL|.+?)', '?([1-9](?:\.[0-9][0-9]?)?)'?, '?([1-9](?:\.[0-9][0-9]?)?)'?\)"
        elif subtask == 3:
            # aspect_term, category, opinion_term, valence, arousal
            if array_based:
                # Allow NULL for aspect_term and opinion_term, specific pattern for VA (0, 1 or 2 decimal places)
                pattern = r"\['(NULL|.+?)', '(.+?)', '(NULL|.+?)', '?([1-9](?:\.[0-9][0-9]?)?)'?, '?([1-9](?:\.[0-9][0-9]?)?)'?\]"
            else:
                pattern = r"\('(NULL|.+?)', '(.+?)', '(NULL|.+?)', '?([1-9](?:\.[0-9][0-9]?)?)'?, '?([1-9](?:\.[0-9][0-9]?)?)'?\)"

        matches = re.match(pattern, t)
        if matches:
            tuples_list.append(matches.groups())

    # Convert to tuples
    tuples_list = [tuple(t) for t in tuples_list]
    
    if len(tuples_list) == 0:
        try:
            # Reconstruct the list format for eval
            eval_string = "[" + label_string + "]" if not label_string.startswith("[") else label_string
            parsed = eval(eval_string)
            if isinstance(parsed, list):
                for tup in parsed:
                    if not isinstance(tup, tuple):
                        continue
                    if subtask == 3 and len(tup) != 5:
                        continue
                    if subtask == 2 and len(tup) != 4:
                        continue
                    if all(isinstance(element, str) for element in tup):
                        tuples_list.append(tup)
        except:
            tuples_list = []

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


def escape_except_space(text):
    return re.sub(r'([.^$*+?{}\[\]\\|()])', r'\\\1', text)


def find_valid_phrases_list(text: str, max_characters_in_phrase: int | None = None) -> list[str]:
    """
    Extract all valid phrases from a given text based on punctuation and formatting rules.

    Args:
        text (str): The input text.
        max_characters_in_phrase (int, optional): Maximum number of characters allowed per phrase. 
                                              If None, no limit is applied.
    Returns:
        list[str]: List of cleaned, unique phrases.
    """
    text = f" {text.strip()} "
    phrases = []

    # Collect split positions
    split_positions = {0, len(text)}

    # Define patterns for splits (merged for clarity)
    split_patterns = [
        r'(?<=\w)(?=[,\.!?;:\-\(\)])',  # before punctuation
        r'\s+',                          # spaces
        r'[\$"\'“”\/…]',                 # before/after special chars
        r'(?<=[a-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u0430-\u044F\u0451])(?=[A-Z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u0410-\u042F\u0401])',  # camel-case and script boundaries
        r'[^\x00-\x7F]',                 # non-ASCII chars
    ]

    # Collect split indices
    for pattern in split_patterns:
        for match in re.finditer(pattern, text):
            split_positions.update({match.start(), match.end()})

    split_positions = sorted(split_positions)

    # Set token limit
    if max_characters_in_phrase is None:
        max_characters_in_phrase = len(text)

    # Generate phrases
    for i, start in enumerate(split_positions):
        for end in split_positions[i + 1:]:
            phrase = text[start:end].strip()
            if not phrase:
                continue
            if len(phrase) <= max_characters_in_phrase:
                phrases.append(phrase)

    # Escape quotes and remove duplicates
    clean_phrases = list(
        {p.replace("'", r"\'").replace('"', r'\"') for p in phrases})

    return clean_phrases


def get_regex_pattern_tuple(unique_aspect_categories, polarities, text, subtask=3, max_characters_in_phrase=64, disable_null_aspect=True):
    valid_phrases = find_valid_phrases_list(
        text, max_characters_in_phrase=max_characters_in_phrase)
    if len(valid_phrases) > 5000:
        return get_regex_pattern_tuple(unique_aspect_categories, polarities, text, subtask=subtask, max_characters_in_phrase=max_characters_in_phrase - 1, disable_null_aspect=disable_null_aspect)

    # Definiere considered_aspects basierend auf subtask
    if subtask == 3:
        # Subtask 3: aspect_term, category, opinion_term, valence, arousal
        considered_aspects = [
            "aspect term", "aspect category", "opinion term", "valence", "arousal"]
    elif subtask == 2:
        # Subtask 2: aspect_term, opinion_term, valence, arousal
        considered_aspects = ["aspect term",
                              "opinion term", "valence", "arousal"]
    else:
        # Subtask 1: aspect_term, category, opinion_term, valence, arousal (gleich wie 3)
        considered_aspects = [
            "aspect term", "aspect category", "opinion term", "valence", "arousal"]

    # Tuple Patternteile
    category_pattern = "|".join(cat for cat in unique_aspect_categories)
    polarity_pattern = "|".join(polarities)

    # Valence und Arousal Pattern - 0, 1 oder 2 Nachkommastellen
    va_pattern = r"[1-9](\.[0-9][0-9]?)?"

    # Tuple pattern
    tuple_pattern_parts = []
    for aspect in considered_aspects:
        if aspect == "aspect term":
            if disable_null_aspect:
                tuple_pattern_parts += [
                    rf"'({'|'.join(escape_except_space(p) for p in valid_phrases)})'"]
            else:
                tuple_pattern_parts += [
                    rf"'(NULL|({'|'.join(escape_except_space(p) for p in valid_phrases)}))'"]
        elif aspect == "aspect category":
            tuple_pattern_parts += [rf"'({category_pattern})'"]
        elif aspect == "sentiment polarity":
            tuple_pattern_parts += [rf"'({polarity_pattern})'"]
        elif aspect == "opinion term":
            if disable_null_aspect:
                tuple_pattern_parts += [
                    rf"'({'|'.join(escape_except_space(p) for p in valid_phrases)})'"]
            else:
                tuple_pattern_parts += [
                    rf"'(NULL|({'|'.join(escape_except_space(p) for p in valid_phrases)}))'"]
        elif aspect == "valence":
            tuple_pattern_parts += [rf"'({va_pattern})'"]
        elif aspect == "arousal":
            tuple_pattern_parts += [rf"'({va_pattern})'"]

    tuple_pattern_str = rf"\(" + rf", ".join(tuple_pattern_parts) + rf"\)"
    tuple_pattern_str = rf"\[{tuple_pattern_str}(, {tuple_pattern_str})*\]\n"

    return tuple_pattern_str


language_mapping = {
    "eng": "English",
    "jpn": "Japanese",
    "rus": "Russian",
    "tat": "Tatar",
    "ukr": "Ukrainian",
    "zho": "Chinese"
}

domain_mapping = {
    "restaurant": "Restaurant",
    "laptop": "Laptop",
    "hotel": "Hotel",
    "finance": "Finance"
}
