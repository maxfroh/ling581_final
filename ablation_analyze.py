import os
import re
import json
from collections import defaultdict
import numpy as np

class AblationAnalyzer:
    def __init__(self, base_dir='ablation study'):
        self.base_dir = base_dir
        self.results = defaultdict(dict)
        self.class_names = ['13-17', '23-27', '33-42']
        
    def parse_ablation_file(self, filepath):
        """Parse a single ablation result file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        data = {}
        
        # Extract original probabilities
        prob_match = re.search(r'Original Probabilities: \[([\d\.\s]+)\]', content)
        if prob_match:
            data['original_probs'] = [float(x) for x in prob_match.group(1).split()]
        
        # Extract true and predicted labels
        true_match = re.search(r'True label: ([\w-]+)', content)
        pred_match = re.search(r'Predicted: ([\w-]+)', content)
        if true_match:
            data['true_label'] = true_match.group(1)
        if pred_match:
            data['predicted_label'] = pred_match.group(1)
        
        # Extract text snippet
        text_match = re.search(r'Text: (.+?)\.\.\.', content, re.DOTALL)
        if text_match:
            data['text_snippet'] = text_match.group(1)[:200] + '...'
        
        return data, content
    
    def parse_top_n_results(self, content):
        """Parse top-N removal results"""
        results = []
        pattern = r"Removing top (\d+) features.*?Probability change: ([\d\.]+) -> ([\d\.]+) \(diff=([-\d\.]+)\)"
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            results.append({
                'n': int(match.group(1)),
                'prob_before': float(match.group(2)),
                'prob_after': float(match.group(3)),
                'diff': float(match.group(4))
            })
        return results
    
    def parse_progressive_results(self, content):
        """Parse progressive removal results"""
        results = []
        pattern = r"Step (\d+): Removed '(\w+)' \(importance: ([-\d\.]+)\).*?Probability: ([\d\.]+) -> ([\d\.]+) \(diff=([-\d\.]+)\)"
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            results.append({
                'step': int(match.group(1)),
                'word': match.group(2),
                'importance': float(match.group(3)),
                'prob_before': float(match.group(4)),
                'prob_after': float(match.group(5)),
                'diff': float(match.group(6))
            })
        return results
    
    def parse_random_results(self, content):
        """Parse random baseline results"""
        data = {}
        
        lime_change = re.search(r'LIME top-5 removal:.*?Probability change: ([-\d\.]+)', content, re.DOTALL)
        random_change = re.search(r'Random word removal:.*?Probability change: ([-\d\.]+)', content, re.DOTALL)
        impact_ratio = re.search(r'LIME is ([\d\.]+)x more impactful', content)
        
        if lime_change:
            data['lime_change'] = float(lime_change.group(1))
        if random_change:
            data['random_change'] = float(random_change.group(1))
        if impact_ratio:
            data['impact_ratio'] = float(impact_ratio.group(1))
        
        return data
    
    def parse_lime_text_file(self, filepath):
        """Parse a lime_text_idx file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        data = {}
        
        # Extract text (everything before "Probability")
        text_match = re.search(r'^(.+?)(?=Probability)', content, re.DOTALL)
        if text_match:
            data['text'] = text_match.group(1).strip()
        
        # Extract probabilities
        prob_13_17 = re.search(r'Probability \(13-17\) = ([\d\.]+)', content)
        prob_23_27 = re.search(r'Probability \(23-27\) = ([\d\.]+)', content)
        prob_33_42 = re.search(r'Probability \(33-42\) = ([\d\.]+)', content)
        
        if prob_13_17 and prob_23_27 and prob_33_42:
            data['probabilities'] = {
                '13-17': float(prob_13_17.group(1)),
                '23-27': float(prob_23_27.group(1)),
                '33-42': float(prob_33_42.group(1))
            }
        
        # Extract true and predicted labels
        true_match = re.search(r'True label: ([\w-]+)', content)
        pred_match = re.search(r'Predicted: ([\w-]+)', content)
        if true_match:
            data['true_label'] = true_match.group(1)
        if pred_match:
            data['predicted_label'] = pred_match.group(1)
        
        # Extract top features
        features_match = re.search(r'Top features:\s*\[(.*?)\]', content, re.DOTALL)
        if features_match:
            features_str = features_match.group(1)
            # Parse tuples like (np.str_('MTV'), 0.048987102456823216)
            feature_pattern = r"np\.str_\('([^']+)'\),\s*([-\d\.]+)"
            features = []
            for match in re.finditer(feature_pattern, features_str):
                features.append({
                    'word': match.group(1),
                    'importance': float(match.group(2))
                })
            data['top_features'] = features
        
        return data
    
    def load_all_results(self):
        """Load all ablation results from the directory structure"""
        if not os.path.exists(self.base_dir):
            print(f"Directory {self.base_dir} not found!")
            return
        
        for idx_folder in sorted(os.listdir(self.base_dir)):
            if not idx_folder.startswith('idx='):
                continue
            
            idx = int(idx_folder.split('=')[1])
            idx_path = os.path.join(self.base_dir, idx_folder)
            
            self.results[idx] = {'metadata': {}}
            
            # Load each ablation type
            for filename in os.listdir(idx_path):
                if filename.startswith('ablation_results'):
                    filepath = os.path.join(idx_path, filename)
                    
                    if 'top_n' in filename:
                        metadata, content = self.parse_ablation_file(filepath)
                        self.results[idx]['metadata'] = metadata
                        self.results[idx]['top_n'] = self.parse_top_n_results(content)
                    
                    elif 'progressive' in filename:
                        _, content = self.parse_ablation_file(filepath)
                        self.results[idx]['progressive'] = self.parse_progressive_results(content)
                    
                    elif 'random' in filename:
                        _, content = self.parse_ablation_file(filepath)
                        self.results[idx]['random'] = self.parse_random_results(content)
                
                elif filename.startswith('lime_text_idx'):
                    filepath = os.path.join(idx_path, filename)
                    self.results[idx]['lime_text'] = self.parse_lime_text_file(filepath)
        
        print(f"Loaded results for {len(self.results)} samples")
    
    def generate_summary_statistics(self):
        """Generate summary statistics across all samples"""
        stats = {
            'total_samples': len(self.results),
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'avg_top10_impact': [],
            'avg_lime_vs_random': [],
            'label_distribution': defaultdict(int),
            'predicted_distribution': defaultdict(int)
        }
        
        for idx, data in self.results.items():
            metadata = data.get('metadata', {})
            
            # Count predictions
            if metadata.get('true_label') == metadata.get('predicted_label'):
                stats['correct_predictions'] += 1
            else:
                stats['incorrect_predictions'] += 1
            
            # Label distributions
            if 'true_label' in metadata:
                stats['label_distribution'][metadata['true_label']] += 1
            if 'predicted_label' in metadata:
                stats['predicted_distribution'][metadata['predicted_label']] += 1
            
            # Top-10 impact
            if 'top_n' in data:
                for result in data['top_n']:
                    if result['n'] == 10:
                        stats['avg_top10_impact'].append(abs(result['diff']))
            
            # LIME vs random
            if 'random' in data and 'impact_ratio' in data['random']:
                stats['avg_lime_vs_random'].append(data['random']['impact_ratio'])
        
        # Calculate averages
        if stats['avg_top10_impact']:
            stats['avg_top10_impact'] = np.mean(stats['avg_top10_impact'])
        else:
            stats['avg_top10_impact'] = 0
            
        if stats['avg_lime_vs_random']:
            stats['avg_lime_vs_random'] = np.mean(stats['avg_lime_vs_random'])
        else:
            stats['avg_lime_vs_random'] = 0
        
        return stats
    
    def generate_report(self, output_file='compiled_ablation_analysis.txt'):
        """Generate a comprehensive analysis report"""
        self.load_all_results()
        stats = self.generate_summary_statistics()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE ABLATION STUDY ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            # Overall Statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Samples Analyzed: {stats['total_samples']}\n")
            f.write(f"Correct Predictions: {stats['correct_predictions']} ({stats['correct_predictions']/stats['total_samples']*100:.1f}%)\n")
            f.write(f"Incorrect Predictions: {stats['incorrect_predictions']} ({stats['incorrect_predictions']/stats['total_samples']*100:.1f}%)\n\n")
            
            f.write("True Label Distribution:\n")
            for label, count in sorted(stats['label_distribution'].items()):
                f.write(f"  {label}: {count} ({count/stats['total_samples']*100:.1f}%)\n")
            
            f.write("\nPredicted Label Distribution:\n")
            for label, count in sorted(stats['predicted_distribution'].items()):
                f.write(f"  {label}: {count} ({count/stats['total_samples']*100:.1f}%)\n")
            
            f.write(f"\nAverage Impact of Removing Top-10 LIME Features: {stats['avg_top10_impact']:.4f}\n")
            f.write(f"Average LIME vs Random Impact Ratio: {stats['avg_lime_vs_random']:.2f}x\n\n")
            
            # Individual Sample Analysis
            f.write("\n" + "="*80 + "\n")
            f.write("INDIVIDUAL SAMPLE ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            for idx in sorted(self.results.keys()):
                data = self.results[idx]
                metadata = data.get('metadata', {})
                
                f.write(f"\n{'='*80}\n")
                f.write(f"INDEX {idx}\n")
                f.write(f"{'='*80}\n\n")
                
                # Metadata
                f.write("METADATA:\n")
                f.write(f"  True Label: {metadata.get('true_label', 'N/A')}\n")
                f.write(f"  Predicted Label: {metadata.get('predicted_label', 'N/A')}\n")
                f.write(f"  Prediction Correct: {'✓' if metadata.get('true_label') == metadata.get('predicted_label') else '✗'}\n")
                if 'original_probs' in metadata:
                    f.write(f"  Original Probabilities:\n")
                    for i, prob in enumerate(metadata['original_probs']):
                        f.write(f"    {self.class_names[i]}: {prob:.4f}\n")
                
                # LIME Text Analysis
                if 'lime_text' in data:
                    lime_data = data['lime_text']
                    f.write("\nLIME EXPLANATION:\n")
                    if 'text' in lime_data:
                        f.write(f"  Text: {lime_data['text'][:200]}...\n")
                    if 'probabilities' in lime_data:
                        f.write(f"  Probabilities:\n")
                        for label, prob in lime_data['probabilities'].items():
                            f.write(f"    {label}: {prob:.4f}\n")
                    if 'top_features' in lime_data:
                        f.write(f"  Top Features:\n")
                        for feat in lime_data['top_features'][:5]:  # Show top 5
                            f.write(f"    {feat['word']}: {feat['importance']:+.4f}\n")
                
                # Top-N Results
                if 'top_n' in data:
                    f.write("\nTOP-N FEATURE REMOVAL:\n")
                    for result in data['top_n']:
                        f.write(f"  Top-{result['n']}: {result['prob_before']:.4f} → {result['prob_after']:.4f} (Δ={result['diff']:+.4f})\n")
                
                # Progressive Results
                if 'progressive' in data:
                    f.write("\nPROGRESSIVE FEATURE REMOVAL:\n")
                    for result in data['progressive']:
                        f.write(f"  Step {result['step']}: Removed '{result['word']}' (imp={result['importance']:+.4f}) → Δ={result['diff']:+.4f}\n")
                
                # Random Baseline
                if 'random' in data:
                    f.write("\nLIME VS RANDOM BASELINE:\n")
                    rand = data['random']
                    f.write(f"  LIME Impact: {rand.get('lime_change', 0):.4f}\n")
                    f.write(f"  Random Impact: {rand.get('random_change', 0):.4f}\n")
                    f.write(f"  LIME is {rand.get('impact_ratio', 0):.2f}x more impactful\n")
                
                f.write("\n")
        
        print(f"\nCompiled analysis saved to: {output_file}")
        return output_file
    
    def generate_json_export(self, output_file='compiled_ablation_data.json'):
        """Export all data as JSON for further analysis"""
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        export_data = {
            'summary_statistics': convert_types(self.generate_summary_statistics()),
            'individual_results': convert_types(dict(self.results))
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"JSON data exported to: {output_file}")
        return output_file
    
    def generate_top_features_list(self, output_file='top_feature_list.txt'):
        """Gather all top features from all files and sort by weight"""
        all_features = []
        
        # Collect all features from all samples
        for idx, data in self.results.items():
            if 'lime_text' in data and 'top_features' in data['lime_text']:
                for feat in data['lime_text']['top_features']:
                    all_features.append({
                        'idx': idx,
                        'word': feat['word'],
                        'importance': feat['importance']
                    })
        
        # Sort by importance (highest to lowest)
        all_features.sort(key=lambda x: x['importance'], reverse=True)
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ALL TOP FEATURES SORTED BY IMPORTANCE (HIGHEST TO LOWEST)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total features collected: {len(all_features)}\n")
            f.write(f"From {len(self.results)} samples\n\n")
            
            f.write(f"{'Rank':<8} {'Word':<30} {'Importance':<15} {'Index':<10}\n")
            f.write("-"*80 + "\n")
            
            for rank, feat in enumerate(all_features, 1):
                f.write(f"{rank:<8} {feat['word']:<30} {feat['importance']:+.6f}      idx={feat['idx']}\n")
        
        print(f"Top features list saved to: {output_file}")
        return output_file
    
    def export_lime_words_and_weights(self, output_file='lime_words_weights.txt'):
        """Export all words and their weights from LIME explanations"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("LIME EXPLANATION WORDS AND WEIGHTS\n")
            f.write("="*80 + "\n\n")
            
            for idx in sorted(self.results.keys()):
                data = self.results[idx]
                
                if 'lime_text' in data and 'top_features' in data['lime_text']:
                    f.write(f"\nINDEX {idx}\n")
                    f.write("-"*80 + "\n")
                    
                    lime_features = data['lime_text']['top_features']
                    
                    f.write(f"{'Word':<30} {'Weight':<15}\n")
                    f.write("-"*80 + "\n")
                    
                    for feat in lime_features:
                        f.write(f"{feat['word']:<30} {feat['importance']:+.6f}\n")
                    
                    f.write("\n")
        
        print(f"LIME words and weights exported to: {output_file}")
        return output_file


def main():
    analyzer = AblationAnalyzer(base_dir='ablation study')
    
    # Generate text report
    analyzer.generate_report('compiled_ablation_analysis.txt')
    
    # Generate JSON export
    analyzer.generate_json_export('compiled_ablation_data.json')
    
    # Generate top features list
    analyzer.generate_top_features_list('top_feature_list.txt')
    
    print("\nAnalysis complete! Generated files:")
    print("  - compiled_ablation_analysis.txt (human-readable report)")
    print("  - compiled_ablation_data.json (structured data for further analysis)")
    print("  - top_feature_list.txt (all top features sorted by importance)")


if __name__ == "__main__":
    main()