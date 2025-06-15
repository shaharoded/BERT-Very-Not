import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import Dict, List, Tuple, Optional
import argparse
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TEDataset(Dataset):
    """
    PyTorch Dataset for textual entailment examples.
    Compatible with the provided dataset creation format.
    """
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Label mapping for string labels
        self.label_map = {
            'entailment': 0, 
            'contradiction': 1, 
            'neutral': 2,
            0: 0, 1: 1, 2: 2,  # Support numeric labels
            -1: 2  # Handle invalid labels as neutral
        }
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract premise and hypothesis - compatible with your format
        premise = str(item.get('premise', ''))
        hypothesis = str(item.get('hypothesis', ''))
        
        # Handle label mapping
        label = item.get('label', 0)
        if label == -1:  # Invalid labels in some datasets
            label = 2  # neutral
        elif isinstance(label, str):
            label = self.label_map.get(label.lower(), 0)
        
        # Tokenize premise and hypothesis together
        encoding = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SpacyAugmentedTrainer:
    """Trainer for BERT model with SpaCy-augmented data (SNLI + MNLI + SQuAD)"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_labels: int = 3, gpu_id: int = 5):
        # Set GPU device - defaults to GPU 3
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        ).to(self.device)
        
        logger.info(f"Initialized SpaCy-augmented model on device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load dataset from pickle file - compatible with your format"""
        try:
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
                
            # Handle different data formats
            if isinstance(data, pd.DataFrame):
                data = data.to_dict('records')
            elif isinstance(data, dict) and 'data' in data:
                data = data['data']
                
            logger.info(f"Loaded dataset: {len(data)} samples")
            
            # Filter out invalid examples
            valid_data = []
            for item in data:
                if (item.get('premise') and item.get('hypothesis') and 
                    item.get('label') is not None):
                    valid_data.append(item)
            
            logger.info(f"Valid samples after filtering: {len(valid_data)}")
            
            # Analyze dataset composition
            self.analyze_dataset_composition(valid_data)
            
            return valid_data
            
        except Exception as e:
            logger.error(f"Error loading dataset from {dataset_path}: {e}")
            return []
    
    def analyze_dataset_composition(self, data: List[Dict]):
        """Analyze the composition of the dataset"""
        label_counts = {}
        augmented_counts = {'spacy_augmented': 0, 'squad_converted': 0, 'original': 0}
        
        for item in data:
            label = item.get('label', 0)
            if label == -1:
                label = 2
            
            label_counts[label] = label_counts.get(label, 0) + 1
            
            # Try to identify data source (this depends on how you tag your data)
            if item.get('augmented') or item.get('source') == 'spacy':
                augmented_counts['spacy_augmented'] += 1
            elif item.get('source') == 'squad' or 'squad' in str(item).lower():
                augmented_counts['squad_converted'] += 1
            else:
                augmented_counts['original'] += 1
        
        logger.info("Dataset composition:")
        logger.info(f"Label distribution: {label_counts}")
        logger.info(f"Source distribution: {augmented_counts}")
    
    def create_data_loader(self, data: List[Dict], batch_size: int = 16, 
                          shuffle: bool = True) -> DataLoader:
        """Create data loader"""
        dataset = TEDataset(data, self.tokenizer)
        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle
        )
        return data_loader
    
    def train_epoch(self, train_loader: DataLoader, optimizer, scheduler, 
                   epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} (SpaCy+SQuAD)')
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def evaluate(self, eval_loader: DataLoader, dataset_name: str = "") -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        predictions = []
        true_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f'Evaluating {dataset_name}'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': total_loss / len(eval_loader)
        }
        
        return metrics, predictions, true_labels
    
    def evaluate_negation_performance(self, eval_data: List[Dict]) -> Dict[str, float]:
        """Evaluate performance specifically on negated/augmented examples"""
        # Separate original and augmented examples
        original_examples = []
        augmented_examples = []
        
        for item in eval_data:
            if item.get('augmented') or item.get('label') == 2:
                augmented_examples.append(item)
            else:
                original_examples.append(item)
        
        results = {}
        
        if original_examples:
            orig_loader = self.create_data_loader(original_examples, shuffle=False)
            orig_metrics, _, _ = self.evaluate(orig_loader, "Original Examples")
            results['original_accuracy'] = orig_metrics['accuracy']
            results['original_f1'] = orig_metrics['f1']
        
        if augmented_examples:
            aug_loader = self.create_data_loader(augmented_examples, shuffle=False)
            aug_metrics, _, _ = self.evaluate(aug_loader, "Augmented Examples")
            results['augmented_accuracy'] = aug_metrics['accuracy']
            results['augmented_f1'] = aug_metrics['f1']
        
        if 'original_accuracy' in results and 'augmented_accuracy' in results:
            results['negation_gap'] = results['original_accuracy'] - results['augmented_accuracy']
        
        return results
    
    def train(self, train_data: List[Dict], val_data: List[Dict] = None,
             epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5,
             save_path: str = "spacy_augmented_model.pt") -> Dict[str, List[float]]:
        """Train the SpaCy-augmented model"""
        
        # Create data loaders
        train_loader = self.create_data_loader(train_data, batch_size, shuffle=True)
        val_loader = self.create_data_loader(val_data, batch_size, shuffle=False) if val_data else None
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        best_val_acc = 0
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, scheduler, epoch + 1)
            history['train_loss'].append(train_loss)
            logger.info(f"Training loss: {train_loss:.4f}")
            
            # Validation
            if val_loader:
                val_metrics, _, _ = self.evaluate(val_loader, "Validation")
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                
                logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}, "
                          f"Accuracy: {val_metrics['accuracy']:.4f}, "
                          f"F1: {val_metrics['f1']:.4f}")
                
                # Evaluate negation performance
                if val_data:
                    neg_results = self.evaluate_negation_performance(val_data)
                    logger.info(f"Negation evaluation: {neg_results}")
                
                # Save best model
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    self.save_model(save_path)
                    logger.info(f"Saved best model with accuracy: {best_val_acc:.4f}")
        
        return history
    
    def save_model(self, path: str):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_labels': self.num_labels
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a saved model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")

def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Training Loss')
    if history['val_loss']:
        ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('SpaCy+SQuAD Model: Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy plot
    if history['val_accuracy']:
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('SpaCy+SQuAD Model: Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train SpaCy Augmented BERT Model')
    parser.add_argument('--train_path', type=str, required=True,
                       help='Path to training dataset pickle file (snli_mnli_squad_spacy_train.pkl)')
    parser.add_argument('--val_path', type=str, required=True,
                       help='Path to validation dataset pickle file (snli_mnli_squad_spacy_validation.pkl)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--save_path', type=str, default='spacy_augmented_model.pt',
                       help='Path to save the trained model')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='BERT model name')
    parser.add_argument('--gpu', type=int, default=3,
                       help='GPU ID to use (default: 3)')
    
    args = parser.parse_args()
    
    # Initialize trainer with specified GPU
    trainer = SpacyAugmentedTrainer(model_name=args.model_name, gpu_id=args.gpu)
    
    # Load datasets
    logger.info("Loading SpaCy-augmented datasets...")
    train_data = trainer.load_dataset(args.train_path)
    val_data = trainer.load_dataset(args.val_path)
    
    if not train_data:
        logger.error("No training data loaded!")
        return
    
    # Train model
    logger.info("Starting SpaCy-augmented training...")
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_path=args.save_path
    )
    
    # Plot training history
    plot_training_history(history, f"spacy_augmented_training_history.png")
    
    # Save training history
    with open('spacy_augmented_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final evaluation on negation performance
    if val_data:
        logger.info("Final negation performance evaluation...")
        neg_results = trainer.evaluate_negation_performance(val_data)
        logger.info("Final negation results:")
        for key, value in neg_results.items():
            logger.info(f"{key}: {value:.4f}")
        
        # Save negation results
        with open('spacy_augmented_negation_results.json', 'w') as f:
            json.dump(neg_results, f, indent=2)
    
    logger.info("SpaCy-augmented training completed!")

if __name__ == "__main__":
    main()