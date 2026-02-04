# machine_learning_protein_classifier

A machine learning tool for bioinformatics that predicts whether a protein sequence is an enzyme based on amino acid composition analysis.

## Overview

This project demonstrates the application of machine learning to protein classification, a common task in bioinformatics. The classifier uses Random Forest algorithm to distinguish between enzymes and non-enzyme structural proteins based on their amino acid composition features.

## Features

- **Amino Acid Composition Analysis**: Extracts frequency-based features from protein sequences
- **Random Forest Classification**: Robust ensemble learning approach
- **Cross-Validation**: 5-fold cross-validation for reliable performance estimation
- **Feature Importance Analysis**: Identifies which amino acids are most predictive
- **Easy-to-Use API**: Simple interface for training and prediction

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone https://github.com/andrewdavis3/machine_learning_protein_classifier.git
cd machine_learning_protein_classifier
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the demonstration script:

```bash
python protein_classifier.py
```

This will:
- Generate sample protein sequences
- Train a Random Forest classifier
- Evaluate performance on test set
- Perform cross-validation
- Display feature importance

### Using as a Library

```python
from protein_classifier import ProteinClassifier

# Initialize classifier
classifier = ProteinClassifier()

# Prepare your data
sequences = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPIL...",
    "ARTKQTARKSTGGKAPRKQLATKAARKSAPATGGVKK...",
]
labels = [1, 0]  # 1 = enzyme, 0 = non-enzyme

# Train the model
classifier.train(sequences, labels)

# Make predictions
new_sequence = "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPF..."
prediction = classifier.predict([new_sequence])[0]
probabilities = classifier.predict_proba([new_sequence])[0]

print(f"Prediction: {'Enzyme' if prediction == 1 else 'Non-enzyme'}")
print(f"Confidence: {max(probabilities):.2%}")
```

## Model Details

### Features

The classifier extracts 21 features from each protein sequence:
- **Amino Acid Frequencies**: Proportion of each of the 20 standard amino acids (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)
- **Sequence Length**: Total number of amino acids

### Algorithm

- **Model**: Random Forest Classifier
- **Estimators**: 100 decision trees
- **Training/Test Split**: 75/25 with stratification
- **Validation**: 5-fold cross-validation

### Performance

On the demonstration dataset:
- **Cross-validation accuracy**: ~65% (±24%)
- Note: Performance on real-world data will vary depending on the quality and size of training data

## Example Output

```
============================================================
Protein Sequence Classifier for Bioinformatics
============================================================

📊 Generating sample protein sequences...
Total sequences: 20
Enzymes: 10, Non-enzymes: 10
Training set: 15 sequences
Test set: 5 sequences

🧬 Training Random Forest classifier...
✓ Training complete!

📈 Evaluating model performance...

Confusion Matrix:
[[1 2]
 [1 1]]

Classification Report:
              precision    recall  f1-score   support

  Non-enzyme       0.50      0.33      0.40         3
      Enzyme       0.33      0.50      0.40         2

    accuracy                           0.40         5

🔄 Performing 5-fold cross-validation...
Mean CV accuracy: 65.00% (+/- 24.49%)

🔍 Top 5 most important features:
1. G: 0.0866
2. F: 0.0781
3. D: 0.0745
4. H: 0.0696
5. R: 0.0634
```

## Real-World Applications

To use this classifier with real protein data:

### 1. Load Data from FASTA Files

```python
from Bio import SeqIO

def load_fasta(filepath):
    sequences = []
    for record in SeqIO.parse(filepath, "fasta"):
        sequences.append(str(record.seq))
    return sequences

# Load enzyme sequences
enzymes = load_fasta("enzymes.fasta")
# Load non-enzyme sequences
non_enzymes = load_fasta("structural_proteins.fasta")

# Create labels
labels = [1] * len(enzymes) + [0] * len(non_enzymes)
sequences = enzymes + non_enzymes
```

### 2. Use Real Databases

- **UniProt**: Download curated enzyme and non-enzyme sequences
- **Pfam**: Use protein family classifications
- **ENZYME**: EC-number annotated enzyme database
- **PDB**: Structural protein databases

### 3. Enhance Features

Consider adding:
- **k-mer frequencies**: Capture sequence patterns (e.g., dipeptides, tripeptides)
- **Physicochemical properties**: Hydrophobicity, charge, molecular weight
- **Secondary structure predictions**: Alpha-helix, beta-sheet content
- **Domain information**: Pfam domains, conserved motifs
- **Sequence embeddings**: Use pre-trained protein language models (ProtBERT, ESM)

## Extending the Classifier

### Try Different Algorithms

```python
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

# SVM
classifier.model = SVC(kernel='rbf', probability=True)

# Neural Network
classifier.model = MLPClassifier(hidden_layer_sizes=(100, 50))

# Gradient Boosting
classifier.model = GradientBoostingClassifier()
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    classifier.model, 
    param_grid, 
    cv=5, 
    scoring='accuracy'
)

X, y = classifier.prepare_data(sequences, labels)
grid_search.fit(X, y)
print(f"Best parameters: {grid_search.best_params_}")
```

## Limitations

- **Simple Features**: Uses only amino acid composition; doesn't capture sequence order or structure
- **Binary Classification**: Currently only distinguishes enzymes vs. non-enzymes
- **Sample Data**: Demonstration uses synthetic/simplified sequences
- **Small Dataset**: Real applications need hundreds or thousands of sequences

## Future Improvements

- [ ] Multi-class classification (enzyme sub-types, protein families)
- [ ] Add k-mer features for sequence pattern recognition
- [ ] Integrate with BioPython for FASTA file handling
- [ ] Support for batch prediction
- [ ] Web interface for easy access
- [ ] Pre-trained models on large datasets
- [ ] Visualization of results and feature importance
- [ ] Integration with protein databases (UniProt API)

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Resources

### Learn More About Protein Bioinformatics

- [UniProt](https://www.uniprot.org/) - Comprehensive protein database
- [BioPython Tutorial](https://biopython.org/wiki/Documentation) - Python tools for bioinformatics
- [scikit-learn Documentation](https://scikit-learn.org/) - Machine learning in Python
- [NCBI Protein Database](https://www.ncbi.nlm.nih.gov/protein/) - Sequence databases

### Related Tools

- [ProtParam](https://web.expasy.org/protparam/) - Protein parameter computation
- [BLAST](https://blast.ncbi.nlm.nih.gov/) - Sequence similarity search
- [InterPro](https://www.ebi.ac.uk/interpro/) - Protein family classification


---

**Note**: This is a demonstration project for educational purposes. For production use in research or clinical settings, use validated tools and larger, curated datasets.
