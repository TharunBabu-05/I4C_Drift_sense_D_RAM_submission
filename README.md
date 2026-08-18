# Drift-Sense: Hybrid Siamese Localization (Final Model)

This repository contains the finalized production code for **Drift-Sense**, a highly robust Hybrid Siamese Neural Network designed for sub-pixel localization of highly-degraded, extreme-noise Scanning Electron Microscope (SEM) imagery.

## 🚀 Environment Setup

We recommend running this project inside a Python virtual environment to prevent dependency conflicts.

1. **Create a Virtual Environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate the Virtual Environment:**
   * On Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   * On Windows:
     ```cmd
     venv\Scripts\activate
     ```

3. **Install the Required Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Run the Bulk Evaluation

We have provided a bulk evaluation script that tests the Hybrid AI model against physical layout pairs and generates a formatted terminal report, visualizations, and a CSV manifest.

Run the following command:
```bash
python3 evaluate.py --data_dir all_60_pairs --checkpoint best_model_level1.pth
```
* **Output:** This will print the final Machine Learning metrics (Inference Speed, Mean Error, Accuracy) to your terminal. It will also create annotated bounding-box visualizations inside `all_60_pairs/visualizer/` and generate a `results_manifest.csv` file.

---

## 🔬 Run Individual Inference

If you want to run the pipeline on a single pair of images, use the Master Inference script.

By default, the script will execute the complete **Hybrid (OpenCV NCC + Siamese AI)** pipeline:
```bash
python master_inference_claude.py --reference img2.png --search img1.png
```

### Additional Inference Options:
* **Run Pure Classical Baseline:** If you want to force the pipeline to bypass the AI and execute only the classic 3-Layer OpenCV Pyramid (for ablation testing), add the `--ncc_only` flag:
  ```bash
  python master_inference_claude.py --reference img2.png --search img1.png --ncc_only
  ```
* **Enable Verbose Output:** Add `--verbose` to see the step-by-step execution times and fusion scores:
  ```bash
  python master_inference_claude.py --reference img2.png --search img1.png --verbose
  ```
