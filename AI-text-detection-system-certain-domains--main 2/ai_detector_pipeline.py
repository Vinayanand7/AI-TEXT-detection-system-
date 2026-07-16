import torch
import math
import numpy as np
import nltk
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    GPT2TokenizerFast,
    GPT2LMHeadModel
)



try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# ----------------------------
# LOAD CLASSIFIER MODEL
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier_tokenizer = RobertaTokenizerFast.from_pretrained("saved_model1")
classifier_model = RobertaForSequenceClassification.from_pretrained("saved_model1")

classifier_model.to(device)
classifier_model.eval()

# ----------------------------
# LOAD PERPLEXITY MODEL
# ----------------------------

ppl_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
ppl_model = GPT2LMHeadModel.from_pretrained("gpt2")

ppl_model.to(device)
ppl_model.eval()

# ----------------------------
# CLASSIFIER SCORE
# ----------------------------
import torch.nn.functional as F

def calibrated_probability(logits, temperature=2.0):

    probs = F.softmax(logits / temperature, dim=-1)

    return probs[0][1].item()
def classifier_score(text):

    try:
        inputs = classifier_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = classifier_model(**inputs)

        logits = outputs.logits

        probs = F.softmax(logits / 2.0, dim=-1)
        score = probs[0][1].item()

        # prevent overconfidence
        score = min(score, 0.85)

        return float(score)

    except Exception as e:
        print("Classifier error:", e)
        return None
def split_into_chunks(text, chunk_size=120):

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

# ----------------------------
# PERPLEXITY SCORE
# ----------------------------
def perplexity_to_score(perplexity):

    midpoint = 60
    steepness = 0.08

    score = 1 / (1 + math.exp(steepness * (perplexity - midpoint)))

    return score

def perplexity_score(text):

    encodings = ppl_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = ppl_model(input_ids, labels=input_ids)

    loss = outputs.loss
    perplexity = math.exp(loss.item())

    score = perplexity_to_score(perplexity)

    return score, perplexity

# ----------------------------
# STYLOMETRY FEATURES
# ----------------------------

def stylometry_score(text):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)

    if len(sentences) < 2:
        return 0.5

    # Sentence lengths
    sentence_lengths = [len(s.split()) for s in sentences]
    burstiness = np.std(sentence_lengths)

    # Normalize burstiness (AI = low variance)
    burstiness_score = 1 / (1 + burstiness)

    # Clean words
    import re
    words = [w.lower() for w in words if re.match(r'^[a-zA-Z]+$', w)]

    vocab_richness = len(set(words)) / max(len(words), 1)

    # AI tends to lower diversity
    vocab_score = 1 - vocab_richness

    # Final stylometry score (AI-likeness)
    style_score = (
        0.6 * burstiness_score +
        0.4 * vocab_score
    )

    return style_score

# ----------------------------
# FINAL AI DETECTOR
# ----------------------------

def detect_ai(text):

    # ignore very short text (avoid garbage predictions)
    if len(text.split()) < 40:
        return {
            "classifier_score": 0.0,
            "perplexity": 0.0,
            "perplexity_score": 0.0,
            "stylometry_score": 0.0,
            "final_ai_probability": 0.0,
            "final_score": 0.0,
            "num_chunks": 0
        }

    chunks = split_into_chunks(text, chunk_size=120)
    clf_scores = []
    ppl_scores = []
    style_scores = []
    perplexities = []

    for chunk in chunks:

        clf = classifier_score(chunk)
        ppl_score, ppl = perplexity_score(chunk)
        style = stylometry_score(chunk)

        #print("DEBUG:", clf, ppl_score, style)

        # 🚨 skip broken chunks
        if clf is None:
            continue

        clf_scores.append(clf)
        ppl_scores.append(ppl_score)
        style_scores.append(style)
        perplexities.append(ppl)
    
    clf_var = np.std(clf_scores)
    ppl_var = np.std(ppl_scores)
    style_var = np.std(style_scores)
    
    if len(clf_scores) == 0:
            return {
        "classifier_score": 0.0,
        "perplexity": 0.0,
        "perplexity_score": 0.0,
        "stylometry_score": 0.0,
        "final_ai_probability": 0.0,
        "num_chunks": 0,
        "clf_variance": clf_var,
        "ppl_variance": ppl_var,
        "style_variance": style_var,
    }
    #print("DEBUG:", clf, ppl_score, style)

    # smarter aggregation (not just average)
    avg_clf = 0.7 * np.mean(clf_scores) + 0.3 * max(clf_scores)
    avg_ppl = 0.7 * np.mean(ppl_scores) + 0.3 * max(ppl_scores)
    avg_style = np.mean(style_scores)
    avg_perplexity = np.mean(perplexities)

    final_score = (
        0.3 * avg_clf +
        0.5 * avg_ppl +
        0.2 * avg_style
    )

    return {
        "classifier_score": avg_clf,
        "perplexity": avg_perplexity,
        "perplexity_score": avg_ppl,
        "stylometry_score": avg_style,
        "final_ai_probability": final_score,
        "final_score": final_score,
        "num_chunks": len(chunks)
    }
def agreement_score(classifier, perplexity_score_value, stylometry):
    values = [classifier, perplexity_score_value, stylometry]
    return 1 - np.std(values)

def length_score(text):
    words = len(text.split())
    return min(words / 500, 1)   # cap at 500 words

def certainty_score(final_ai_probability):
    return abs(final_ai_probability - 0.5) * 2

import numpy as np

def confidence_score(detection, text):
    final = detection["final_score"]

    # chunk agreement (LOW variance = HIGH confidence)
    clf_var = detection.get("clf_variance", 0)
    ppl_var = detection.get("ppl_variance", 0)
    style_var = detection.get("style_variance", 0)

    variance = np.mean([clf_var, ppl_var, style_var])
    agreement = 1 / (1 + variance)

    # text length factor
    words = len(text.split())
    length = min(words / 500, 1)

    # certainty (distance from 0.5)
    certainty = abs(final - 0.5) * 2

    return (
        0.5 * agreement +
        0.3 * length +
        0.2 * certainty
    )

# ----------------------------
# TEST TEXT
# ----------------------------

if __name__ == "__main__":

    text = """Biodiversity—the variety of life on Earth, from the smallest bacteria to the largest whales—is the foundation of our planet's health. It is not merely a collection of individual species; it is a complex, functional network where every organism plays a role in maintaining the balance of ecosystems. Protecting this diversity is essential for human survival and global stability.

The benefits of a biodiverse world are often referred to as "ecosystem services." For instance, diverse plant life ensures soil fertility and provides the oxygen we breathe. Insects, birds, and bats act as pollinators for the crops that feed billions of people. Furthermore, the natural world is a vast pharmacy; a significant percentage of modern medicines, including treatments for malaria and heart disease, are derived from wild plants and fungi. When we lose a species, we may be losing a future cure for a terminal illness.

Today, however, biodiversity is under threat at an alarming rate due to habitat destruction, pollution, and climate change. The "extinction crisis" is not just a tragedy for wildlife; it is a direct threat to human food security and economic stability. Monoculture—the practice of growing only one type of crop—makes our food supply vulnerable to pests and diseases that a diverse ecosystem would naturally keep in check.

To preserve the Earth's biological heritage, we must move beyond small-scale conservation toward systemic change. This includes protecting vast tracts of wilderness, restoring degraded lands, and shifting toward sustainable agricultural practices. Biodiversity is the life-support system of our planet. By protecting it, we are not just saving "nature"—we are ensuring the continued existence of the human race and a vibrant, livable world for future generations.  """

    result = detect_ai(text)

    print("\n---- AI Detection Report ----")

    for k, v in result.items():
        print(k, ":", round(v, 4))
    
    conf = confidence_score(result, text)


    print("Confidence Score:", round(conf, 4))

    if result["final_ai_probability"] > 0.75:
        print("\nPrediction: AI Generated")
    else:
        print("\nPrediction: Human Written")