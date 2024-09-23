import os
import string
import numpy as np
from PIL import Image
from transformers.image_transforms import pad
from transformers import T5TokenizerFast
import evaluate
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib
import matplotlib.pyplot as plt
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import spacy
from nltk import sent_tokenize

def split_into_sentences(text, method='spacy'):
    if method == 'spacy':
        # Using Spacy
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]
    elif method == 'nltk':
        # Using NLTK
        sentences = sent_tokenize(text)
    else:
        raise ValueError("Invalid method. Supported methods: 'spacy' or 'nltk'")
    
    return sentences


def normalize_scores(scores, min_range, max_range):
    if len(scores) == 0:
        return scores.clone().detach()
    min_score = min(scores)
    max_score = max(scores)

    normalized_scores = [((score - min_score) / (max_score - min_score) * (max_range - min_range) + min_range).item() for score in scores]

    return normalized_scores

def colorize(words, color_array):
    cmap=matplotlib.cm.Blues
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        # print(color)
        # colored_string += template.format(color, '&nbsp' + word + '&nbsp')
        colored_string += template.format(color, word)
    return colored_string

def sdprint(obj):
    if obj is None:
        print('None')
    if hasattr(obj, 'shape'):
        print('shape', obj.shape)
    else:
        print('type', type(obj))


def compute_metrics(pred_str, label_str, use_bert = False, use_spice = False):
    # https://huggingface.co/spaces/evaluate-metric/bleu bleu default 4 grams
    metric_bleu = evaluate.load("bleu")
    metric_rouge = evaluate.load("rouge")
    metric_meteor = evaluate.load('meteor')
    

    bleu4 = metric_bleu.compute(predictions=pred_str, references=label_str)["bleu"]
    rougeL = metric_rouge.compute(predictions=pred_str, references=label_str)["rougeL"]
    meteor = metric_meteor.compute(predictions=pred_str, references=label_str)["meteor"]
    if use_bert:
        metric_bert = evaluate.load('bertscore')
        bert = metric_bert.compute(predictions=pred_str, references=label_str, lang="en")
        bertscore = np.mean(bert['f1'])
    else:
        bertscore = None

    references = {index: [value] for index, value in enumerate(label_str)}
    candidate = {index: [value] for index, value in enumerate(pred_str)}

    scorer = Cider()
    cider = scorer.compute_score(references, candidate)[0]

    spice = None
    if use_spice:
        scorer = Spice()
        spice = scorer.compute_score(references, candidate)[0]
    
    return {"bleu4": bleu4, "rougeL": rougeL, "meteor":meteor, 
            "bertscore": bertscore, "cider": cider, "spice": spice}

def remove_stopwords(text):
    # Tokenize the text into individual words
    tokens = word_tokenize(text)

    # Get the list of English stop words
    stop_words = set(stopwords.words('english'))

    # Remove stop words from the tokens
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Join the filtered tokens back into a single string
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text

def get_string(input_data):
    if isinstance(input_data, list):
        if len(input_data) > 0:  
            return input_data[0] 
        else:
            return None 
    else:
        return input_data  

def preprocess_sentence(sentence):
    sentence = sentence.lower()  # Convert to lowercase
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = set(sentence.split())  # Tokenize into words and convert to a set
    return words

def compute_jaccard_score(sentence1, sentence2):
    words1 = preprocess_sentence(sentence1)
    words2 = preprocess_sentence(sentence2)


    intersection = words1.intersection(words2)
    union = words1.union(words2)

    if len(union) == 0:
        jaccard_score  = 0
    else: 
        jaccard_score = len(intersection) / len(union)
    if len(words1) == 0:
        modified_jaccard  = -1

    else: 
        modified_jaccard = len(intersection) / len(words1)
    return jaccard_score, modified_jaccard

def compute_metrics_all(pred_str, label_str, mode = 'rouge'):
    if mode == 'rouge':
        # https://huggingface.co/spaces/evaluate-metric/bleu bleu default 4 grams
        metric_bleu = evaluate.load("bleu")
        metric_rouge = evaluate.load("rouge")
        # metric_meteor = evaluate.load('meteor')
        
        if pred_str == '' or pred_str == '\n':
            pred_str = '.'


        rouge_scores = metric_rouge.compute(predictions=pred_str, references=label_str)
        # print(rouge_scores)
        rougeL = rouge_scores["rougeL"]
        rouge1 = rouge_scores["rouge1"]
        rouge2 = rouge_scores["rouge2"]
        rougeLsum = rouge_scores["rougeLsum"]
        
        return {
                "rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL, "rougeLsum": rougeLsum}
    elif mode == 'jaccard':
        pred_str = get_string(pred_str)
        label_str = get_string(label_str)
        jaccard_score, modified_jaccard = compute_jaccard_score(pred_str, label_str)

        return {'jaccard': jaccard_score, 'modified_jaccard': modified_jaccard}
    else:
        print('Unknown Metrics')
        return None

def get_feature_path(img_path, save_string, PREP_DIR):
    relative_path = img_path[len(PREP_DIR):]
    relative_path = relative_path.replace('images', save_string) # better not to make path longer to avoid path too long error
    base, extension = os.path.splitext(relative_path)
    new_path = PREP_DIR + base + '.pt'
    return new_path

def get_save_string(model_name):
    if model_name == "openai/clip-vit-base-patch16":
        return "clipb16"
    elif model_name == "openai/clip-vit-large-patch14":
        return "clipl14"
    elif model_name == "openai/clip-vit-base-patch32":
        return "clipb32"
    
def pad_to_square(image):
    width, height = image.size
    # Determine the target size (the larger dimension)
    target_size = max(width, height)
    square_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    x_offset = (target_size - width) // 2
    y_offset = (target_size - height) // 2

    # Paste the original image into the center of the square image
    square_image.paste(image, (x_offset, y_offset))
    return square_image

def pad_to_square_np(raw_image):
    h, w, _ = raw_image.shape
    if h > w:
        pad_before = (h - w)//2
        pad_after = h - w - pad_before
        raw_image = pad(raw_image, ((0, 0), (pad_before, pad_after)), constant_values = 0)
    elif h < w:
        pad_before = (w - h)//2
        pad_after = w - h - pad_before
        raw_image = pad(raw_image, ((pad_before, pad_after), (0, 0)), constant_values = 0)

    return raw_image

# Based on https://github.com/neural-dialogue-metrics/Distinct-N/blob/main/distinct_n/metrics.py
from itertools import chain
def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence

def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    # return n-grams as an iterator
    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]

def word_level_ngrams(text, n, pad_left=False, pad_right=False,
                      left_pad_symbol='<s>', right_pad_symbol='</s>'):
    words = text.split()  # Split the text into words
    return ngrams(words, n, pad_left, pad_right, left_pad_symbol, right_pad_symbol)

def preprocess(text):
    # Convert to lower case
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize (split into words)
    return text.split()

def distinct_n_sentence_level(sentence, n):
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level_avg(sentences, n):
    """
    This is actually just average of utterance level
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)


def distinct_n_corpus_level(sentences, n):
    all_ngrams = []
    total_length = 0

    for sentence in sentences:
        if len(sentence.split()) >= n:  # Only process sentences long enough for n-grams
            all_ngrams.extend(word_level_ngrams(sentence, n))
            total_length += len(sentence)

    if total_length == 0:  # Prevent division by zero
        return 0.0


    distinct_ngrams = set(all_ngrams)
    return len(distinct_ngrams) / total_length