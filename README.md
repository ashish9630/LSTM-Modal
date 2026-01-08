# LSTM Text Generation Project

A production-ready implementation of LSTM-based text generation using TensorFlow and Keras.

## Features

- **Complete Data Pipeline**: Text loading, preprocessing, tokenization, and sequence creation
- **Robust Model Architecture**: Multi-layer LSTM with embedding and dropout layers
- **Training Optimization**: Early stopping, model checkpointing, and validation splitting
- **Text Generation**: Configurable text generation with temperature sampling
- **Model Persistence**: Save/load trained models and tokenizers
- **Parameter Experiments**: Easy experimentation with different configurations

## Project Structure

```
LSTM/
├── lstm_text_generator.py    # Main implementation
├── requirements.txt          # Dependencies
├── README.md                # This file
├── sample_text.txt          # Generated sample dataset
├── best_lstm_model.h5       # Best model checkpoint (after training)
├── lstm_model.h5            # Final trained model (after training)
└── tokenizer.pkl            # Saved tokenizer (after training)
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the project:
```bash
python lstm_text_generator.py
```

## Usage

### Basic Usage
The script will automatically create a sample dataset and train the model:

```python
python lstm_text_generator.py
```

### Custom Dataset
Replace `sample_text.txt` with your own text file, or modify the `dataset_path` variable in the `main()` function.

### Configuration
Modify these parameters in the `main()` function:
- `SEQUENCE_LENGTH`: Length of input sequences (default: 30)
- `EMBEDDING_DIM`: Embedding layer dimension (default: 100)
- `LSTM_UNITS`: Number of LSTM units (default: 128)
- `EPOCHS`: Training epochs (default: 20)

## Model Architecture

1. **Embedding Layer**: Converts tokens to dense vectors
2. **LSTM Layers**: Two stacked LSTM layers with dropout
3. **Dense Output**: Softmax activation for probability distribution

## Key Features Implemented

✅ **Dataset Handling**: Loads and preprocesses text data  
✅ **Text Preprocessing**: Lowercase conversion, punctuation removal  
✅ **Tokenization**: Keras Tokenizer with sequence creation  
✅ **Model Architecture**: Sequential API with embedding and LSTM layers  
✅ **Training**: Train/validation split with callbacks  
✅ **Text Generation**: Iterative prediction with temperature sampling  
✅ **Model Persistence**: Save/load functionality  
✅ **Parameter Experiments**: Temperature and architecture variations  

## Performance Tips

- **Sequence Length**: 50-100 for better context (requires more memory)
- **Training Data**: More diverse text improves generation quality
- **Epochs**: 50-100 epochs for production models
- **Temperature**: 0.5-0.8 for coherent text, 1.0+ for creativity

## Example Output

```
Seed 1: 'to be or not to be'
Generated: to be or not to be that is the question whether tis nobler in the mind to suffer the slings and arrows...

Seed 2: 'the question whether tis nobler'
Generated: the question whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune...
```

## Advanced Usage

### Custom Text Generation
```python
generator = LSTMTextGenerator(sequence_length=50, lstm_units=256)
# ... train model ...
text = generator.generate_text("your seed text", num_words=100, temperature=0.7)
```

### Load Pretrained Model
```python
generator = LSTMTextGenerator()
generator.load_model_and_tokenizer('lstm_model.h5', 'tokenizer.pkl')
text = generator.generate_text("seed text", num_words=50)
```

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- NumPy 1.21+
- scikit-learn 1.0+

## License

This project is open source and available under the MIT License.
