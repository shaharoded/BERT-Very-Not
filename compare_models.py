import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
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

class ModelEvaluator:
    """Comprehensive evaluator for all three trained models"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_labels: int = 3, gpu_id: int = 3):
        # Set GPU device - defaults to GPU 3
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Placeholder for models
        self.models = {}
        
        logger.info(f"Initialized evaluator on device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_model(self, model_path: str, model_key: str):
        """Load a trained model"""
        try:
            model = BertForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=self.num_labels
            ).to(self.device)
            
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.models[model_key] = model
            logger.info(f"Loaded {model_key} model from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading {model_key} model from {model_path}: {e}")
    
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load dataset from pickle file"""
        try:
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
                
            # Handle different data formats
            if isinstance(data, pd.DataFrame):
                data = data.to_dict('records')
            elif isinstance(data, dict) and 'data' in data:
                data = data['data']
                
            # Filter out invalid examples
            valid_data = []
            for item in data:
                if (item.get('premise') and item.get('hypothesis') and 
                    item.get('label') is not None):
                    valid_data.append(item)
            
            logger.info(f"Loaded dataset: {len(valid_data)} valid samples")
            return valid_data
            
        except Exception as e:
            logger.error(f"Error loading dataset from {dataset_path}: {e}")
            return []
    
    def create_data_loader(self, data: List[Dict], batch_size: int = 16) -> DataLoader:
        """Create data loader"""
        dataset = TEDataset(data, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return data_loader
    
    def evaluate_model(self, model, eval_loader: DataLoader, model_name: str = "") -> Dict[str, any]:
        """Evaluate a single model"""
        model.eval()
        predictions = []
        true_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f'Evaluating {model_name}'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(
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
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': total_loss / len(eval_loader),
            'predictions': predictions,
            'true_labels': true_labels,
            'per_class_metrics': {
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1': f1_per_class.tolist(),
                'support': support.tolist()
            }
        }
        
        return metrics
    
    def evaluate_all_models(self, test_data: List[Dict], batch_size: int = 16) -> Dict[str, Dict]:
        """Evaluate all loaded models on the test data"""
        test_loader = self.create_data_loader(test_data, batch_size)
        results = {}
        
        for model_key, model in self.models.items():
            logger.info(f"Evaluating {model_key} model...")
            metrics = self.evaluate_model(model, test_loader, model_key)
            results[model_key] = metrics
            
            logger.info(f"{model_key} Results:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
        
        return results
    
    def evaluate_negation_handling(self, test_data: List[Dict]) -> Dict[str, Dict]:
        """Evaluate negation handling across all models"""
        # Separate original and augmented examples
        original_examples = []
        llm_augmented_examples = []
        spacy_augmented_examples = []
        
        for item in test_data:
            if item.get('augmented') == True:
                llm_augmented_examples.append(item)
            elif item.get('label') == 2 and not item.get('augmented'):
                spacy_augmented_examples.append(item)
            else:
                original_examples.append(item)
        
        logger.info(f"Negation analysis data split:")
        logger.info(f"  Original examples: {len(original_examples)}")
        logger.info(f"  LLM augmented: {len(llm_augmented_examples)}")
        logger.info(f"  SpaCy augmented: {len(spacy_augmented_examples)}")
        
        negation_results = {}
        
        for model_key, model in self.models.items():
            model_results = {}
            
            # Evaluate on original examples
            if original_examples:
                orig_loader = self.create_data_loader(original_examples)
                orig_metrics = self.evaluate_model(model, orig_loader, f"{model_key}_original")
                model_results['original_accuracy'] = orig_metrics['accuracy']
                model_results['original_f1'] = orig_metrics['f1']
            
            # Evaluate on LLM-augmented examples
            if llm_augmented_examples:
                llm_loader = self.create_data_loader(llm_augmented_examples)
                llm_metrics = self.evaluate_model(model, llm_loader, f"{model_key}_llm_aug")
                model_results['llm_augmented_accuracy'] = llm_metrics['accuracy']
                model_results['llm_augmented_f1'] = llm_metrics['f1']
            
            # Evaluate on SpaCy-augmented examples
            if spacy_augmented_examples:
                spacy_loader = self.create_data_loader(spacy_augmented_examples)
                spacy_metrics = self.evaluate_model(model, spacy_loader, f"{model_key}_spacy_aug")
                model_results['spacy_augmented_accuracy'] = spacy_metrics['accuracy']
                model_results['spacy_augmented_f1'] = spacy_metrics['f1']
            
            # Calculate negation gaps
            if 'original_accuracy' in model_results and 'llm_augmented_accuracy' in model_results:
                model_results['llm_negation_gap'] = model_results['original_accuracy'] - model_results['llm_augmented_accuracy']
            
            if 'original_accuracy' in model_results and 'spacy_augmented_accuracy' in model_results:
                model_results['spacy_negation_gap'] = model_results['original_accuracy'] - model_results['spacy_augmented_accuracy']
            
            negation_results[model_key] = model_results
        
        return negation_results
    
    def plot_model_comparison(self, results: Dict[str, Dict], save_path: str = None):
        """Plot comparison of all models"""
        models = list(results.keys())
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            bars = axes[i].bar(models, values, alpha=0.7)
            axes[i].set_title(f'{metric.capitalize()} Comparison')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_negation_analysis(self, negation_results: Dict[str, Dict], save_path: str = None):
        """Plot negation handling analysis"""
        models = list(negation_results.keys())
        
        # Prepare data for plotting
        original_accs = []
        llm_aug_accs = []
        spacy_aug_accs = []
        llm_gaps = []
        spacy_gaps = []
        
        for model in models:
            results = negation_results[model]
            original_accs.append(results.get('original_accuracy', 0))
            llm_aug_accs.append(results.get('llm_augmented_accuracy', 0))
            spacy_aug_accs.append(results.get('spacy_augmented_accuracy', 0))
            llm_gaps.append(results.get('llm_negation_gap', 0))
            spacy_gaps.append(results.get('spacy_negation_gap', 0))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy comparison
        x = np.arange(len(models))
        width = 0.25
        
        ax1.bar(x - width, original_accs, width, label='Original Examples', alpha=0.8)
        ax1.bar(x, llm_aug_accs, width, label='LLM Augmented', alpha=0.8)
        ax1.bar(x + width, spacy_aug_accs, width, label='SpaCy Augmented', alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy on Different Example Types')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Negation gaps
        ax2.bar(x - width/2, llm_gaps, width, label='LLM Negation Gap', alpha=0.8)
        ax2.bar(x + width/2, spacy_gaps, width, label='SpaCy Negation Gap', alpha=0.8)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Negation Gap')
        ax2.set_title('Negation Handling Gaps')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_confusion_matrices(self, results: Dict[str, Dict], save_path: str = None):
        """Plot confusion matrices for all models"""
        num_models = len(results)
        fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5))
        
        if num_models == 1:
            axes = [axes]
        
        label_names = ['Entailment', 'Contradiction', 'Neutral']
        
        for i, (model_name, result) in enumerate(results.items()):
            cm = confusion_matrix(result['true_labels'], result['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=label_names, yticklabels=label_names,
                       ax=axes[i])
            axes[i].set_title(f'{model_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted Label')
            axes[i].set_ylabel('True Label')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def generate_detailed_report(self, results: Dict[str, Dict], negation_results: Dict[str, Dict]) -> str:
        """Generate a detailed evaluation report"""
        report = "=" * 80 + "\n"
        report += "COMPREHENSIVE MODEL EVALUATION REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Overall performance summary
        report += "1. OVERALL PERFORMANCE SUMMARY\n"
        report += "-" * 40 + "\n"
        
        for model_name, result in results.items():
            report += f"\n{model_name.upper()} MODEL:\n"
            report += f"  Accuracy:  {result['accuracy']:.4f}\n"
            report += f"  F1 Score:  {result['f1']:.4f}\n"
            report += f"  Precision: {result['precision']:.4f}\n"
            report += f"  Recall:    {result['recall']:.4f}\n"
            report += f"  Loss:      {result['loss']:.4f}\n"
        
        # Per-class performance
        report += "\n\n2. PER-CLASS PERFORMANCE\n"
        report += "-" * 40 + "\n"
        
        class_names = ['Entailment', 'Contradiction', 'Neutral']
        for model_name, result in results.items():
            report += f"\n{model_name.upper()} MODEL:\n"
            for i, class_name in enumerate(class_names):
                if i < len(result['per_class_metrics']['f1']):
                    report += f"  {class_name}:\n"
                    report += f"    Precision: {result['per_class_metrics']['precision'][i]:.4f}\n"
                    report += f"    Recall:    {result['per_class_metrics']['recall'][i]:.4f}\n"
                    report += f"    F1 Score:  {result['per_class_metrics']['f1'][i]:.4f}\n"
                    report += f"    Support:   {result['per_class_metrics']['support'][i]}\n"
        
        # Negation handling analysis
        report += "\n\n3. NEGATION HANDLING ANALYSIS\n"
        report += "-" * 40 + "\n"
        
        for model_name, neg_result in negation_results.items():
            report += f"\n{model_name.upper()} MODEL:\n"
            if 'original_accuracy' in neg_result:
                report += f"  Original Examples Accuracy:     {neg_result['original_accuracy']:.4f}\n"
            if 'llm_augmented_accuracy' in neg_result:
                report += f"  LLM Augmented Accuracy:         {neg_result['llm_augmented_accuracy']:.4f}\n"
            if 'spacy_augmented_accuracy' in neg_result:
                report += f"  SpaCy Augmented Accuracy:       {neg_result['spacy_augmented_accuracy']:.4f}\n"
            if 'llm_negation_gap' in neg_result:
                report += f"  LLM Negation Gap:               {neg_result['llm_negation_gap']:.4f}\n"
            if 'spacy_negation_gap' in neg_result:
                report += f"  SpaCy Negation Gap:             {neg_result['spacy_negation_gap']:.4f}\n"
        
        # Model ranking
        report += "\n\n4. MODEL RANKING\n"
        report += "-" * 40 + "\n"
        
        # Sort by overall accuracy
        sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        report += "\nBy Overall Accuracy:\n"
        for i, (model_name, result) in enumerate(sorted_models, 1):
            report += f"  {i}. {model_name}: {result['accuracy']:.4f}\n"
        
        # Sort by negation handling (smallest gap is best)
        if negation_results:
            # Use LLM negation gap as primary metric
            sorted_negation = sorted(
                [(k, v) for k, v in negation_results.items() if 'llm_negation_gap' in v],
                key=lambda x: x[1]['llm_negation_gap']
            )
            if sorted_negation:
                report += "\nBy LLM Negation Handling (lower gap is better):\n"
                for i, (model_name, result) in enumerate(sorted_negation, 1):
                    report += f"  {i}. {model_name}: {result['llm_negation_gap']:.4f}\n"
        
        # Key insights
        report += "\n\n5. KEY INSIGHTS\n"
        report += "-" * 40 + "\n"
        
        best_overall = sorted_models[0][0]
        report += f"• Best overall performer: {best_overall}\n"
        
        if sorted_negation:
            best_negation = sorted_negation[0][0]
            report += f"• Best negation handler: {best_negation}\n"
        
        # Performance differences
        if len(results) > 1:
            best_acc = sorted_models[0][1]['accuracy']
            worst_acc = sorted_models[-1][1]['accuracy']
            report += f"• Performance spread: {(best_acc - worst_acc):.4f}\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Evaluate and Compare All Models')
    parser.add_argument('--baseline_model', type=str, required=True,
                       help='Path to baseline model (baseline_model.pt)')
    parser.add_argument('--spacy_model', type=str, required=True,
                       help='Path to SpaCy augmented model (spacy_augmented_model.pt)')
    parser.add_argument('--llm_model', type=str, required=True,
                       help='Path to LLM augmented model (llm_augmented_model.pt)')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test dataset (snli_mnli_test_llm_test.pkl)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--gpu', type=int, default=3,
                       help='GPU ID to use (default: 3)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator with specified GPU
    evaluator = ModelEvaluator(gpu_id=args.gpu)
    
    # Load all models
    evaluator.load_model(args.baseline_model, 'baseline')
    evaluator.load_model(args.spacy_model, 'spacy_augmented')
    evaluator.load_model(args.llm_model, 'llm_augmented')
    
    # Load test data
    logger.info("Loading test dataset...")
    test_data = evaluator.load_dataset(args.test_data)
    
    if not test_data:
        logger.error("No test data loaded!")
        return
    
    # Evaluate all models
    logger.info("Evaluating all models...")
    results = evaluator.evaluate_all_models(test_data, args.batch_size)
    
    # Negation handling analysis
    logger.info("Analyzing negation handling...")
    negation_results = evaluator.evaluate_negation_handling(test_data)
    
    # Generate plots
    logger.info("Generating visualizations...")
    evaluator.plot_model_comparison(results, 
                                  os.path.join(args.output_dir, 'model_comparison.png'))
    
    evaluator.plot_negation_analysis(negation_results,
                                    os.path.join(args.output_dir, 'negation_analysis.png'))
    
    evaluator.plot_confusion_matrices(results,
                                     os.path.join(args.output_dir, 'confusion_matrices.png'))
    
    # Generate detailed report
    logger.info("Generating detailed report...")
    report = evaluator.generate_detailed_report(results, negation_results)
    
    # Save results
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump({
            'overall_results': results,
            'negation_results': negation_results
        }, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    with open(os.path.join(args.output_dir, 'evaluation_report.txt'), 'w') as f:
        f.write(report)
    
    # Print report to console
    print(report)
    
    logger.info(f"Evaluation completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()