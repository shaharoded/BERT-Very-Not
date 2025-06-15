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

class LLMAugmentedTrainer:
    """Trainer for BERT model with LLM-augmented negation data (SNLI + MNLI + LLM)"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_labels: int = 3, gpu_id: int = 3):
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
        
        logger.info(f"Initialized LLM-augmented model on device: {self.device}")
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
        """Analyze the composition of the dataset with focus on LLM augmentation"""
        label_counts = {}
        augmented_counts = {'llm_augmented': 0, 'original': 0}
        
        for item in data:
            label = item.get('label', 0)
            if label == -1:
                label = 2
            
            label_counts[label] = label_counts.get(label, 0) + 1
            
            # Check if this is an LLM-augmented example
            if item.get('augmented') == True:
                augmented_counts['llm_augmented'] += 1
            else:
                augmented_counts['original'] += 1
        
        logger.info("LLM-Augmented Dataset composition:")
        logger.info(f"Label distribution: {label_counts}")
        logger.info(f"Source distribution: {augmented_counts}")
        
        # Calculate augmentation ratio
        total = len(data)
        llm_ratio = augmented_counts['llm_augmented'] / total if total > 0 else 0
        logger.info(f"LLM augmentation ratio: {llm_ratio:.2%}")
    
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
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} (LLM-Augmented)')
        
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
        """Evaluate performance specifically on LLM-generated negated examples"""
        # Separate original and LLM-augmented examples
        original_examples = []
        llm_augmented_examples = []
        
        for item in eval_data:
            if item.get('augmented') == True:
                llm_augmented_examples.append(item)
            else:
                original_examples.append(item)
        
        results = {}
        
        if original_examples:
            orig_loader = self.create_data_loader(original_examples, shuffle=False)
            orig_metrics, _, _ = self.evaluate(orig_loader, "Original Examples")
            results['original_accuracy'] = orig_metrics['accuracy']
            results['original_f1'] = orig_metrics['f1']
            logger.info(f"Original examples: {len(original_examples)} samples")
        
        if llm_augmented_examples:
            llm_loader = self.create_data_loader(llm_augmented_examples, shuffle=False)
            llm_metrics, _, _ = self.evaluate(llm_loader, "LLM-Augmented Examples")
            results['llm_augmented_accuracy'] = llm_metrics['accuracy']
            results['llm_augmented_f1'] = llm_metrics['f1']
            logger.info(f"LLM-augmented examples: {len(llm_augmented_examples)} samples")
        
        if 'original_accuracy' in results and 'llm_augmented_accuracy' in results:
            results['llm_negation_gap'] = results['original_accuracy'] - results['llm_augmented_accuracy']
            logger.info(f"LLM Negation Gap: {results['llm_negation_gap']:.4f}")
        
        return results
    
    def detailed_negation_analysis(self, test_data: List[Dict]) -> Dict[str, any]:
        """Perform detailed analysis of negation handling capabilities"""
        results = {}
        
        # Separate examples by type
        original_entailment = [item for item in test_data if item.get('label') == 0 and not item.get('augmented')]
        llm_contradictions = [item for item in test_data if item.get('label') == 2 and item.get('augmented')]
        
        results['dataset_stats'] = {
            'total_examples': len(test_data),
            'original_entailment': len(original_entailment),
            'llm_contradictions': len(llm_contradictions)
        }
        
        # Evaluate each subset
        if original_entailment:
            orig_loader = self.create_data_loader(original_entailment, shuffle=False)
            orig_metrics, orig_preds, orig_true = self.evaluate(orig_loader, "Original Entailment")
            results['original_entailment_accuracy'] = orig_metrics['accuracy']
        
        if llm_contradictions:
            llm_loader = self.create_data_loader(llm_contradictions, shuffle=False)
            llm_metrics, llm_preds, llm_true = self.evaluate(llm_loader, "LLM Contradictions")
            results['llm_contradiction_accuracy'] = llm_metrics['accuracy']
            
            # Specific analysis for contradiction detection
            correct_contradictions = sum(1 for pred, true in zip(llm_preds, llm_true) if pred == 2 and true == 2)
            total_contradictions = len(llm_true)
            results['contradiction_detection_rate'] = correct_contradictions / total_contradictions if total_contradictions > 0 else 0
        
        return results
    
    def train(self, train_data: List[Dict], val_data: List[Dict] = None,
             epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5,
             save_path: str = "llm_augmented_model.pt") -> Dict[str, List[float]]:
        """Train the LLM-augmented model"""
        
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
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'negation_gaps': []}
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
                    if 'llm_negation_gap' in neg_results:
                        history['negation_gaps'].append(neg_results['llm_negation_gap'])
                    logger.info(f"LLM negation evaluation: {neg_results}")
                
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
    """Plot training history for LLM-augmented model"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Training Loss')
    if history['val_loss']:
        axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('LLM-Augmented Model: Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Accuracy plot
    if history['val_accuracy']:
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('LLM-Augmented Model: Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
    
    # Negation gap plot
    if history['negation_gaps']:
        axes[2].plot(history['negation_gaps'], label='LLM Negation Gap', color='red')
        axes[2].set_title('LLM Negation Gap Over Training')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Negation Gap')
        axes[2].legend()
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_negation_comparison(results: Dict[str, float], save_path: str = None):
    """Plot comparison between original and LLM-augmented performance"""
    if 'original_accuracy' not in results or 'llm_augmented_accuracy' not in results:
        logger.warning("Insufficient data for negation comparison plot")
        return
    
    metrics = ['Accuracy', 'F1 Score']
    original_scores = [results['original_accuracy'], results['original_f1']]
    llm_scores = [results['llm_augmented_accuracy'], results['llm_augmented_f1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, original_scores, width, label='Original Examples', alpha=0.8)
    ax.bar(x + width/2, llm_scores, width, label='LLM-Augmented Examples', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance: Original vs LLM-Augmented Examples')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, (orig, llm) in enumerate(zip(original_scores, llm_scores)):
        ax.text(i - width/2, orig + 0.01, f'{orig:.3f}', ha='center')
        ax.text(i + width/2, llm + 0.01, f'{llm:.3f}', ha='center')
    
    # Add negation gap annotation
    if 'llm_negation_gap' in results:
        gap = results['llm_negation_gap']
        ax.text(0.5, 0.9, f'Negation Gap: {gap:.3f}', transform=ax.transAxes, 
                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train LLM Augmented BERT Model')
    parser.add_argument('--train_path', type=str, required=True,
                       help='Path to training dataset pickle file (snli_mnli_llm_train.pkl)')
    parser.add_argument('--val_path', type=str, required=True,
                       help='Path to validation dataset pickle file (snli_mnli_llm_validation.pkl)')
    parser.add_argument('--test_path', type=str, default=None,
                       help='Path to test dataset pickle file (snli_mnli_test_llm_test.pkl)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--save_path', type=str, default='llm_augmented_model.pt',
                       help='Path to save the trained model')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='BERT model name')
    parser.add_argument('--gpu', type=int, default=3,
                       help='GPU ID to use (default: 3)')
    
    args = parser.parse_args()
    
    # Initialize trainer with specified GPU
    trainer = LLMAugmentedTrainer(model_name=args.model_name, gpu_id=args.gpu)
    
    # Load datasets
    logger.info("Loading LLM-augmented datasets...")
    train_data = trainer.load_dataset(args.train_path)
    val_data = trainer.load_dataset(args.val_path)
    
    if not train_data:
        logger.error("No training data loaded!")
        return
    
    # Train model
    logger.info("Starting LLM-augmented training...")
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_path=args.save_path
    )
    
    # Plot training history
    plot_training_history(history, f"llm_augmented_training_history.png")
    
    # Save training history
    with open('llm_augmented_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final evaluation on negation performance
    if val_data:
        logger.info("Final LLM negation performance evaluation...")
        neg_results = trainer.evaluate_negation_performance(val_data)
        logger.info("Final LLM negation results:")
        for key, value in neg_results.items():
            logger.info(f"{key}: {value:.4f}")
        
        # Plot negation comparison
        plot_negation_comparison(neg_results, "llm_negation_comparison.png")
        
        # Save negation results
        with open('llm_augmented_negation_results.json', 'w') as f:
            json.dump(neg_results, f, indent=2)
    
    # Test set evaluation if provided
    if args.test_path:
        logger.info("Evaluating on test set...")
        test_data = trainer.load_dataset(args.test_path)
        if test_data:
            test_loader = trainer.create_data_loader(test_data, shuffle=False)
            test_metrics, test_preds, test_true = trainer.evaluate(test_loader, "Test Set")
            
            logger.info("Test Set Results:")
            for key, value in test_metrics.items():
                logger.info(f"{key}: {value:.4f}")
            
            # Detailed negation analysis on test set
            detailed_results = trainer.detailed_negation_analysis(test_data)
            logger.info("Detailed negation analysis:")
            logger.info(json.dumps(detailed_results, indent=2))
            
            # Save test results
            with open('llm_test_results.json', 'w') as f:
                json.dump({
                    'test_metrics': test_metrics,
                    'detailed_analysis': detailed_results
                }, f, indent=2)
    
    logger.info("LLM-augmented training completed!")

if __name__ == "__main__":
    main()