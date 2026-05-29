import os
import subprocess
import re
import pandas as pd
from datetime import datetime
import glob
from pathlib import Path

class ModelEvaluationRunner:
    def __init__(self, work_dirs_path, evaluation_script_path, output_excel_path=None):
        """
        Initialize the model evaluation runner.
        
        Args:
            work_dirs_path (str): Path to the work_dirs directory containing model subdirectories
            evaluation_script_path (str): Path to your existing .py evaluation script
            output_excel_path (str): Path for the output Excel file (optional)
        """
        self.work_dirs_path = work_dirs_path
        self.evaluation_script_path = evaluation_script_path
        self.output_excel_path = output_excel_path or "model_evaluation_results.xlsx"
        self.results = []
        
    def find_model_files(self):
        """
        Find all model (.pth) and config (.py) file pairs in subdirectories.
        
        Returns:
            list: List of tuples containing (model_path, config_path, subfolder_name)
        """
        model_pairs = []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.work_dirs_path):
            # Skip the root directory itself
            if root == self.work_dirs_path:
                continue
                
            model_file = None
            config_file = None
            
            # Look for the model and config files in this directory
            for file in files:
                if file.startswith("best_coco_segm") and file.endswith(".pth"):
                    model_file = os.path.join(root, file)
                elif file.startswith("custom_mask2former") and file.endswith(".py"):
                    config_file = os.path.join(root, file)
            
            # If we found both files, add them to our list
            if model_file and config_file:
                subfolder_name = os.path.basename(root)
                model_pairs.append((model_file, config_file, subfolder_name))
                print(f"Found model pair in {subfolder_name}:")
                print(f"  Model: {os.path.basename(model_file)}")
                print(f"  Config: {os.path.basename(config_file)}")
            elif model_file or config_file:
                # Found one but not the other
                subfolder_name = os.path.basename(root)
                print(f"WARNING: Incomplete pair in {subfolder_name}")
                if model_file:
                    print(f"  Found model: {os.path.basename(model_file)} but no config file")
                if config_file:
                    print(f"  Found config: {os.path.basename(config_file)} but no model file")
        
        return model_pairs
    
    def parse_evaluation_output(self, output_text):
        """
        Parse the evaluation output to extract AP metrics.
        
        Args:
            output_text (str): The stdout from the evaluation script
            
        Returns:
            dict: Dictionary containing parsed metrics
        """
        metrics = {}
        
        # Define patterns for the metrics we want
        patterns = {
            'bbox_ap_50_95': r'=====  BBOX  =====.*?Average Precision.*?IoU=0\.50:0\.95.*?area=\s*all.*?= ([\d\.-]+)',
            'bbox_ap_50': r'=====  BBOX  =====.*?Average Precision.*?IoU=0\.50\s+\|.*?area=\s*all.*?= ([\d\.-]+)',
            'bbox_ap_75': r'=====  BBOX  =====.*?Average Precision.*?IoU=0\.75.*?area=\s*all.*?= ([\d\.-]+)',
            'segm_ap_50_95': r'=====  SEGM  =====.*?Average Precision.*?IoU=0\.50:0\.95.*?area=\s*all.*?= ([\d\.-]+)',
            'segm_ap_50': r'=====  SEGM  =====.*?Average Precision.*?IoU=0\.50\s+\|.*?area=\s*all.*?= ([\d\.-]+)',
            'segm_ap_75': r'=====  SEGM  =====.*?Average Precision.*?IoU=0\.75.*?area=\s*all.*?= ([\d\.-]+)'
        }
        
        for metric_name, pattern in patterns.items():
            match = re.search(pattern, output_text, re.DOTALL)
            if match:
                try:
                    metrics[metric_name] = float(match.group(1))
                except ValueError:
                    metrics[metric_name] = None
            else:
                metrics[metric_name] = None
                
        return metrics
    
    def run_evaluation(self, model_path, config_path):
        """
        Run the evaluation script for a given model and config.
        
        Args:
            model_path (str): Path to the model file
            config_path (str): Path to the config file
            
        Returns:
            dict: Parsed evaluation metrics or None if failed
        """
        try:
            # Construct the command - you may need to adjust this based on your script's interface
            cmd = [
                'python', self.evaluation_script_path,
                '--checkpoint', model_path,
                '--config', config_path
            ]
            
            print(f"Running evaluation command: {' '.join(cmd)}")
            
            # Run the command and capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                # Parse the output
                metrics = self.parse_evaluation_output(result.stdout)
                return metrics
            else:
                print(f"Error running evaluation:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("Evaluation timed out after 1 hour")
            return None
        except Exception as e:
            print(f"Exception during evaluation: {str(e)}")
            return None
    
    def run_all_evaluations(self):
        """
        Run evaluations for all found model pairs.
        """
        model_pairs = self.find_model_files()
        
        if not model_pairs:
            print("No model pairs found!")
            return
        
        print(f"\nFound {len(model_pairs)} model pairs to evaluate")
        
        for i, (model_path, config_path, subfolder_name) in enumerate(model_pairs, 1):
            print(f"\n{'='*60}")
            print(f"Evaluating {i}/{len(model_pairs)}: {subfolder_name}")
            print(f"{'='*60}")
            
            metrics = self.run_evaluation(model_path, config_path)
            
            # Prepare result record
            result_record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'subfolder': subfolder_name,
                'model_file': os.path.basename(model_path),
                'config_file': os.path.basename(config_path),
                'model_path': model_path,
                'config_path': config_path,
                'status': 'success' if metrics else 'failed'
            }
            
            if metrics:
                result_record.update(metrics)
                print("Evaluation completed successfully!")
                print("Results:")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
            else:
                print("Evaluation failed!")
                # Add None values for all metrics
                for metric in ['bbox_ap_50_95', 'bbox_ap_50', 'bbox_ap_75', 
                              'segm_ap_50_95', 'segm_ap_50', 'segm_ap_75']:
                    result_record[metric] = None
            
            self.results.append(result_record)
            
            # Save intermediate results (in case of crashes)
            self.save_results()
    
    def save_results(self):
        """
        Save results to Excel file.
        """
        if not self.results:
            print("No results to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Reorder columns for better readability
        column_order = [
            'timestamp', 'subfolder', 'status',
            'bbox_ap_50_95', 'bbox_ap_50', 'bbox_ap_75',
            'segm_ap_50_95', 'segm_ap_50', 'segm_ap_75',
            'model_file', 'config_file', 'model_path', 'config_path'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in df.columns]
        df = df[available_columns]
        
        # Save to Excel
        try:
            df.to_excel(self.output_excel_path, index=False)
            print(f"\nResults saved to: {self.output_excel_path}")
        except Exception as e:
            print(f"Error saving Excel file: {str(e)}")
            # Fallback to CSV
            csv_path = self.output_excel_path.replace('.xlsx', '.csv')
            df.to_csv(csv_path, index=False)
            print(f"Saved as CSV instead: {csv_path}")
    
    def print_summary(self):
        """
        Print a summary of all results.
        """
        if not self.results:
            print("No results to summarize")
            return
        
        successful = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] == 'failed']
        
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total models evaluated: {len(self.results)}")
        print(f"Successful evaluations: {len(successful)}")
        print(f"Failed evaluations: {len(failed)}")
        
        if successful:
            print(f"\nTop performing models (by BBOX AP@0.5:0.95):")
            successful_df = pd.DataFrame(successful)
            if 'bbox_ap_50_95' in successful_df.columns:
                top_models = successful_df.nlargest(5, 'bbox_ap_50_95')
                for _, row in top_models.iterrows():
                    print(f"  {row['subfolder']}: {row['bbox_ap_50_95']:.3f}")

def main():
    # Configuration - UPDATE THESE PATHS
    WORK_DIRS_PATH = r"C:\Users\five\Desktop\Manny\AI_only\mmdetection\work_dirs"
    EVALUATION_SCRIPT_PATH = r"C:\Users\five\Desktop\Manny\AI_only\mmdetection\test_models_hand_annos.py"
    OUTPUT_EXCEL_PATH = r"C:\Users\five\Desktop\Manny\FIV906\model_evaluation_results.xlsx"
    
    # Create and run evaluator
    evaluator = ModelEvaluationRunner(
        work_dirs_path=WORK_DIRS_PATH,
        evaluation_script_path=EVALUATION_SCRIPT_PATH,
        output_excel_path=OUTPUT_EXCEL_PATH
    )
    
    print("Starting automated model evaluation...")
    print(f"Work directory: {WORK_DIRS_PATH}")
    print(f"Evaluation script: {EVALUATION_SCRIPT_PATH}")
    print(f"Output file: {OUTPUT_EXCEL_PATH}")
    
    # Run all evaluations
    evaluator.run_all_evaluations()
    
    # Print summary
    evaluator.print_summary()
    
    print(f"\nEvaluation complete! Results saved to {OUTPUT_EXCEL_PATH}")

if __name__ == "__main__":
    main()