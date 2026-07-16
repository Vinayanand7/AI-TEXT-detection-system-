# AI-text-detection-system-certain-domains-

# AI Text Detection System

## Overview

The AI Text Detection System is a web-based application designed to identify whether a given text is AI-generated or human-written. The system combines transformer-based deep learning models, perplexity analysis, and stylometric feature extraction to improve detection accuracy.



---

## Features

- Detects AI-generated and Human-written text
- Transformer-based classification using RoBERTa
- Perplexity score analysis
- Stylometric feature extraction
- Weighted ensemble scoring mechanism
- Interactive web interface
- Visualization of detection results
- File upload support
- Detailed analysis reports

---

## Technologies Used

### Frontend

- HTML
- CSS
- JavaScript


### Backend

- Python
- Flask

### Machine Learning

- PyTorch
- Transformers
- RoBERTa
- Scikit-Learn

### NLP Libraries

- NLTK
- TextStat

---

## System Architecture

1. User submits text.
2. Text preprocessing is performed.
3. RoBERTa classifier predicts AI probability.
4. Perplexity score is calculated.
5. Stylometric features are extracted.
6. Weighted ensemble combines all scores.
7. Final prediction is generated.

---

## Detection Formula

Final Score =

0.60 × Classifier Score +
0.25 × Perplexity Support +
0.15 × Stylometric Score

Based on the final score, the system predicts whether the text is AI-generated or human-written.

---

## Dataset

The project uses a combination of:

- Human-written text samples
- AI-generated text samples from modern LLMs

Examples include:

- Essays
- Articles
- Academic writing

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/AI-Text-Detection-System.git
```

Move into project directory:

```bash
cd AI-Text-Detection-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

## Project Outcomes

- Successfully integrated multiple NLP techniques.

- Developed an intuitive user interface.
- Achieved effective detection of AI-generated text.

---

## Future Enhancements

- Detection of outputs from newer LLMs.
- Larger and more diverse datasets.
- Continuous model retraining.
- Multilingual text detection.
- Real-time API integration.

---


## Authors

Bhanu Prakash

---

## License

This project is developed for academic and educational purposes.
