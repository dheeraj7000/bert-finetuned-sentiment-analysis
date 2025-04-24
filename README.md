# 🧠 BERT Fine-Tuned Sentiment Analysis

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

A fine-tuned BERT model for sentiment analysis tasks, achieving state-of-the-art performance on customer review datasets.

## 📌 Key Features

- **Pre-trained BERT Base Model** leveraged for transfer learning
- **Fine-tuned on custom dataset** for domain-specific sentiment analysis
- **Achieves 92%+ accuracy** on validation set
- **Easy-to-use prediction pipeline** for new text inputs
- **Lightweight deployment** options with ONNX conversion

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/dheeraj7000/bert-finetuned-sentiment-analysis.git
cd bert-finetuned-sentiment-analysis
pip install -r requirements.txt
```

### Basic Usage

```python
from sentiment_analyzer import SentimentAnalyzer

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Predict sentiment
text = "This product works amazingly well!"
result = analyzer.predict(text)
print(result)  # Output: {'label': 'POSITIVE', 'score': 0.98}
```

## 🏋️ Training Details

| Hyperparameter       | Value          |
|----------------------|----------------|
| Base Model           | bert-base-uncased |
| Epochs               | 4              |
| Batch Size           | 32             |
| Learning Rate        | 2e-5           |
| Max Sequence Length  | 256            |
| Warmup Steps         | 500            |

## 📊 Performance Metrics

| Metric        | Training | Validation |
|---------------|----------|------------|
| Accuracy      | 94.2%    | 92.7%      |
| Precision     | 94.5%    | 92.9%      |
| Recall        | 94.1%    | 92.6%      |
| F1 Score      | 94.3%    | 92.7%      |

## 🗂️ Project Structure

```
bert-finetuned-sentiment-analysis/
├── data/                   # Training datasets
│   ├── train.csv           # Training samples
│   └── test.csv            # Validation samples
├── notebooks/              # Jupyter notebooks
│   └── sentiment_analysis.ipynb  # Training notebook
├── models/                 # Saved model weights
├── sentiment_analyzer.py   # Prediction pipeline
├── train.py                # Training script
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## 🌐 Deployment Options

### Option 1: Local REST API
```bash
python api.py  # Starts FastAPI server on port 8000
```

### Option 2: ONNX Runtime
```python
# Convert to ONNX (see convert_to_onnx.py)
onnx_model = ONNXRuntime(model_path="model.onnx")
```

### Option 3: HuggingFace Hub
```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="dheeraj7000/bert-sentiment")
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📚 References

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Developed with ❤️ by <a href="https://github.com/dheeraj7000">Dheeraj</a>
</div>
```
