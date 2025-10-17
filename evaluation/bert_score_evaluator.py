"""
BERT Score Evaluation for Askly RAG system
Comprehensive evaluation framework for retrieval and generation quality
"""
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
import numpy as np
from datetime import datetime


class BERTScoreEvaluator:
    """
    Evaluate RAG system using BERT Score metrics
    """
    
    def __init__(self, model_name: str = "bert-base-multilingual-cased"):
        """
        Initialize evaluator
        
        Args:
            model_name: Model to use for BERT Score
        """
        self.model_name = model_name
        self._bert_score = None
    
    def _load_bert_score(self):
        """Lazy load bert_score to avoid import errors"""
        if self._bert_score is None:
            try:
                from bert_score import score as bert_score
                self._bert_score = bert_score
                print(f"[INFO] BERT Score loaded with model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "bert-score not installed. "
                    "Install with: pip install bert-score"
                )
    
    def evaluate_single(
        self,
        prediction: str,
        reference: str,
        return_hash: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate a single prediction-reference pair
        
        Args:
            prediction: Generated answer
            reference: Reference (ground truth) answer
            return_hash: Return model hash for reproducibility
        
        Returns:
            Dictionary with P, R, F1 scores
        """
        self._load_bert_score()
        
        P, R, F1 = self._bert_score(
            [prediction],
            [reference],
            model_type=self.model_name,
            lang="vi",
            verbose=False,
            return_hash=return_hash
        )
        
        result = {
            "precision": float(P[0]),
            "recall": float(R[0]),
            "f1": float(F1[0])
        }
        
        return result
    
    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str],
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Evaluate multiple predictions
        
        Args:
            predictions: List of generated answers
            references: List of reference answers
            batch_size: Batch size for processing
        
        Returns:
            Dictionary with scores and statistics
        """
        self._load_bert_score()
        
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")
        
        print(f"[INFO] Evaluating {len(predictions)} predictions...")
        
        P, R, F1 = self._bert_score(
            predictions,
            references,
            model_type=self.model_name,
            lang="vi",
            verbose=True,
            batch_size=batch_size
        )
        
        # Convert to numpy for statistics
        P_np = P.numpy()
        R_np = R.numpy()
        F1_np = F1.numpy()
        
        results = {
            "num_samples": len(predictions),
            "precision": {
                "mean": float(np.mean(P_np)),
                "std": float(np.std(P_np)),
                "min": float(np.min(P_np)),
                "max": float(np.max(P_np)),
                "values": P_np.tolist()
            },
            "recall": {
                "mean": float(np.mean(R_np)),
                "std": float(np.std(R_np)),
                "min": float(np.min(R_np)),
                "max": float(np.max(R_np)),
                "values": R_np.tolist()
            },
            "f1": {
                "mean": float(np.mean(F1_np)),
                "std": float(np.std(F1_np)),
                "min": float(np.min(F1_np)),
                "max": float(np.max(F1_np)),
                "values": F1_np.tolist()
            },
            "model": self.model_name,
            "timestamp": datetime.now().isoformat()
        }
        
        return results
    
    def evaluate_from_dataset(
        self,
        dataset_path: Path,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Evaluate from a dataset file
        
        Dataset format (JSON):
        [
            {
                "question": "...",
                "prediction": "...",
                "reference": "..."
            },
            ...
        ]
        
        Args:
            dataset_path: Path to dataset JSON file
            output_path: Optional path to save results
        
        Returns:
            Evaluation results
        """
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"[INFO] Loaded dataset with {len(dataset)} samples")
        
        # Extract predictions and references
        predictions = [item['prediction'] for item in dataset]
        references = [item['reference'] for item in dataset]
        
        # Evaluate
        results = self.evaluate_batch(predictions, references)
        
        # Add per-sample results
        results['samples'] = []
        for i, item in enumerate(dataset):
            sample_result = {
                "question": item.get('question', ''),
                "prediction": predictions[i],
                "reference": references[i],
                "precision": results['precision']['values'][i],
                "recall": results['recall']['values'][i],
                "f1": results['f1']['values'][i]
            }
            results['samples'].append(sample_result)
        
        # Save results if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"[INFO] Results saved to: {output_path}")
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary"""
        print("\n" + "="*80)
        print("BERT SCORE EVALUATION SUMMARY")
        print("="*80)
        print(f"Model: {results['model']}")
        print(f"Samples: {results['num_samples']}")
        print(f"\nPrecision: {results['precision']['mean']:.4f} ± {results['precision']['std']:.4f}")
        print(f"Recall:    {results['recall']['mean']:.4f} ± {results['recall']['std']:.4f}")
        print(f"F1 Score:  {results['f1']['mean']:.4f} ± {results['f1']['std']:.4f}")
        print("="*80 + "\n")


class RAGEvaluator:
    """
    Comprehensive RAG system evaluator
    Combines multiple metrics
    """
    
    def __init__(self, pipeline):
        """
        Initialize RAG evaluator
        
        Args:
            pipeline: RAGPipeline instance
        """
        self.pipeline = pipeline
        self.bert_evaluator = BERTScoreEvaluator()
    
    def evaluate_retrieval(
        self,
        test_queries: List[Dict[str, Any]],
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval quality
        
        Test queries format:
        [
            {
                "query": "...",
                "relevant_docs": ["doc1", "doc2", ...]  # Ground truth
            },
            ...
        ]
        
        Returns:
            Retrieval metrics (precision, recall, MRR)
        """
        precision_scores = []
        recall_scores = []
        mrr_scores = []
        
        for item in test_queries:
            query = item['query']
            relevant_docs = set(item['relevant_docs'])
            
            # Get retrieved documents
            results = self.pipeline.search(query, k, print_results=False)
            retrieved_docs = [r['sentence_chunk'] for r in results]
            
            # Calculate metrics
            retrieved_set = set(retrieved_docs[:k])
            relevant_retrieved = relevant_set & retrieved_set
            
            precision = len(relevant_retrieved) / k if k > 0 else 0
            recall = len(relevant_retrieved) / len(relevant_docs) if len(relevant_docs) > 0 else 0
            
            # MRR (Mean Reciprocal Rank)
            rank = None
            for i, doc in enumerate(retrieved_docs, 1):
                if doc in relevant_docs:
                    rank = i
                    break
            mrr = 1.0 / rank if rank else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            mrr_scores.append(mrr)
        
        return {
            "precision@k": {
                "mean": np.mean(precision_scores),
                "std": np.std(precision_scores)
            },
            "recall@k": {
                "mean": np.mean(recall_scores),
                "std": np.std(recall_scores)
            },
            "mrr": {
                "mean": np.mean(mrr_scores),
                "std": np.std(mrr_scores)
            },
            "k": k,
            "num_queries": len(test_queries)
        }
    
    def evaluate_generation(
        self,
        test_questions: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Evaluate generation quality
        
        Test questions format:
        [
            {
                "question": "...",
                "reference_answer": "..."
            },
            ...
        ]
        
        Returns:
            Generation metrics using BERT Score
        """
        predictions = []
        references = []
        
        print(f"[INFO] Generating answers for {len(test_questions)} questions...")
        
        for item in test_questions:
            question = item['question']
            reference = item['reference_answer']
            
            # Generate answer
            answer = self.pipeline.ask(question, return_context=False)
            
            predictions.append(answer)
            references.append(reference)
        
        # Evaluate with BERT Score
        results = self.bert_evaluator.evaluate_batch(predictions, references)
        
        return results
    
    def full_evaluation(
        self,
        retrieval_dataset: Optional[List[Dict]] = None,
        generation_dataset: Optional[List[Dict]] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Perform full evaluation
        
        Args:
            retrieval_dataset: Dataset for retrieval evaluation
            generation_dataset: Dataset for generation evaluation
            output_dir: Directory to save results
        
        Returns:
            Complete evaluation results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_info": self.pipeline.get_pipeline_info()
        }
        
        # Retrieval evaluation
        if retrieval_dataset:
            print("\n[1/2] Evaluating retrieval...")
            results['retrieval'] = self.evaluate_retrieval(retrieval_dataset)
        
        # Generation evaluation
        if generation_dataset:
            print("\n[2/2] Evaluating generation...")
            results['generation'] = self.evaluate_generation(generation_dataset)
        
        # Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\n[INFO] Results saved to: {output_file}")
        
        return results
