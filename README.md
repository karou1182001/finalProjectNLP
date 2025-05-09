README


This project explores the mathematical reasoning capabilities of large language models (LLMs) on the GSM8K benchmark. We compare multiple GPT-2 variants (base, medium, large) and Phi-1.5 using zero-shot and few-shot chain-of-thought (CoT) prompting, and we analyze how performance changes when the dataset is modified with added noise or scaled numerical values.
For GPT_model_(large_and_medium) files and folders:

final_outputs/ → Final JSON result files for GPT-2 medium and large (few-shot, zero-shot)


models/ → Fine-tuned model folders and result JSONs


modified_data/ → JSONs of model outputs on modified datasets (noise, scaling)


dataset_extract.py → Script to extract and prepare dataset


base_case_runs.py → Script for running base case evaluations


gpt2_medium_large.py → Script for running GPT-2 medium and large models


re_runs_few_shot.py → Script for rerunning few-shot experiments


re_runs_zero_shot.py → Script for rerunning zero-shot experiments


requirements.txt → Python dependencies

For gpt2_model_base:

Finalgpt2.ipynb → Jupyter Notebook for final GPT-2 analysis and runs


all_predictions.json → Combined predictions file (all runs)


all_predictions_zeroCOT.json → Combined predictions for zero-shot chain-of-thought runs


perfect_subset.json → Subset of perfect (100% correct) predictions from original data


perfect_subset_modified.json → Subset of perfect predictions after dataset modifications


perfect_subset_zoroCOT.json → Subset of perfect predictions for zero-shot CoT runs

For Phi Model files and folder:
all_predictions_phi.json → Combined predictions for Phi model (all runs)


all_predictions_zeroCOT_phi.json → Combined predictions for Phi zero-shot CoT runs


perfect_subset_phi.json → Perfect (100% correct) subset, Phi model


perfect_subset_phiOriginal.json → Perfect subset from original Phi dataset


perfect_subset_zoroCOT_phi.json → Perfect subset, zero-shot CoT Phi results


perfect_subset_zoroCOT_phiOriginal.json → Perfect subset, original Phi zero-shot CoT


phi.ipynb → Jupyter Notebook for Phi base model experiments


phiNofinetune.ipynb → Phi runs without fine-tuning


phiRerunWithJsonFile1.ipynb → Rerun of Phi with JSON inputs (part 1)

Running Python Scripts
1. Make sure you have Python installed (recommended: Python 3.8 or later).
2.  Install required dependencies:
nginx
CopyEdit
pip install -r requirements.txt

3. Run any script using:
nginx
CopyEdit
python script_name.py

For example:
nginx
CopyEdit
python base_case_runs.py
python gpt2_medium_large.py

These scripts typically read input data (like JSON files), run evaluations or predictions, and save outputs (usually into a results/ or modified_data/ folder).

Running Jupyter Notebooks
Launch Jupyter Notebook:
nginx
CopyEdit
jupyter notebook

In your browser, navigate to the notebook file (e.g., Finalgpt2.ipynb or phi.ipynb) and click to open it.
 Run each cell in order:
Use Shift + Enter to execute a selected cell.


Make sure the environment has all necessary dependencies installed.


Output files or evaluation results will typically be saved in the same directory or in designated folders like final_outputs/ or modified_data/.

phiRerunWithJsonFile2.ipynb → Rerun of Phi with JSON inputs (part 2)

