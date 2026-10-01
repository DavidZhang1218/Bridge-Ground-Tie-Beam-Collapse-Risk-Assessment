# Bridge Ground Tie Beam Assessment

This package processes a bridge drawing PDF with at least two pages. Page 1 is the profile arrangement drawing; every later page contains section views. The application detects bridge components and elevations, extracts section heights with Gemini, calculates excavation depth, and creates a structured result and PDF report.

## Package contents

```text
app.py
modules/
sample_image/
  *.pdf
  ICLprompt.png
weights/
  stage1.pt
  stage2.pt
requirements.txt
```

The application lists every PDF found directly in `sample_image/` and initially selects the first filename in alphabetical order. Use **Refresh samples** after adding a PDF while the application is running. An uploaded PDF takes precedence over the selected sample. The first page of each PDF is the profile drawing and all remaining pages are section views. The ICL reference image is preselected and can be replaced before a run. The two detection files are fixed and loaded from `weights/`; the interface does not expose detector settings.

## Installation

Python 3.11 is recommended. Install PyTorch and PaddlePaddle builds compatible with your operating system and, if applicable, your CUDA version. For a CPU environment, install the standard packages first. Then install the remaining packages:

```bash
python -m pip install torch paddlepaddle
python -m pip install -r requirements.txt
```

PaddleOCR downloads its recognition model on the first run if it is not cached. On Linux, the available system fonts determine the appearance of result images and reports. Tesseract is optional when PaddleOCR is available.

## Run

```bash
python app.py
```

The browser opens the local interface. Select the sample PDF or upload another PDF with a profile page followed by one or more section pages, confirm the ICL reference image, enter a Gemini API key, and click **Run Safety Assessment**. The key can instead be set with the `GEMINI_API_KEY` environment variable. Select a VLM available to your account. App labels, prompts, and reports are in English, and the Gradio footer is hidden. A native file chooser dialog follows the browser and operating system language settings.

The default compute device is CPU. Select `cuda:0` only when the installed PyTorch and PaddlePaddle builds support the GPU.

The application saves each run in a separate folder under `outputs/`. The PDF pages are rendered as JPG files at 300 dpi. The result folder contains detection artifacts, `assessment.json`, `final_visualization.png`, and `assessment_report.pdf` with result tables, the final visualization, and every section drawing. Existing runs are retained.

To convert a PDF without launching the interface:

```bash
python -m modules.pdf_input sample_image/test.pdf outputs/converted_test
```

This produces `test.jpg` for the profile and `test_sectionview.jpg` for the first section page. Additional section pages are saved as `test_sectionview_3.jpg`, `test_sectionview_4.jpg`, and so on. All section pages appear in the interface gallery and the report.

## Assessment parameters

The seven visible indicator Ri scores are editable and initially use the values from the project's demonstration interface: groundwater 55, surroundings 45, rainfall 80, geology 40, excavation method 25, monitoring 35, and technical applicability 20. They are not inferred from the drawing. Each Ri score must be between 0 and 100. The eight weights initially use equal values of 0.125 so a sample run can calculate P. These equal weights are release defaults, not coefficients supplied by the source interface or guideline, and remain editable in **Advanced P calculation**. The depth-band Ri defaults are the midpoints of their respective ranges: 12.5, 37.5, 62.5, and 87.5; each remains editable. The depth bands are below 3 m, 3 to below 5 m, 5 to 10 m, and above 10 m. Depth scores must fall in the corresponding guideline ranges of 0–25, 25–50, 50–75, and 75–100. Each weight must be between 0 and 1 and the eight weights must sum to 1. Select λ from 0.9, 0.95, 1.0, 1.05, or 1.1; the initial selection is 1.0. The application calculates `P = λ × Σ(Ri × γi)` and assigns the likelihood level using the guideline thresholds.

The four excavation-depth Ri values are manual choices made before running. The calculated d selects the corresponding depth band for each beam. If any scoring field is blank, drawing extraction and depth calculation still run, while P and its likelihood level remain unavailable. A missing section height or elevation also prevents assessment for that record. Candidates classified as excluded are omitted from the final table, visualization, and report; their count remains in the JSON result. The report and JSON identify whether equal or edited weights were used.

## Output definitions

- `H1`: elevation matched to the beam candidate, in metres.
- `H2`: ground elevation from the profile data band, in metres.
- `h`: section-view beam height, in centimetres.
- `d`: excavation depth, in metres, calculated as `h / 100 - H1 + H2`.

Candidates with `H1 − H2 ≤ 0` are classified as full excavation. Candidates with `0 < H1 − H2 < h / 100` are partial excavation. Candidates with `H1 − H2 ≥ h / 100` are excluded from risk assessment. Records without reliable source values are retained with an explicit unavailable status.

For `H2`, the program recognizes the ground elevation row, finds the nearest table divider that crosses that row, and reads the value immediately to its left. If the divider-based crop cannot be read, it retries at the beam position. Missing values remain unavailable instead of being estimated.

For `H1`, OCR alternatives are checked against the vertical order of nearby level boxes in the same drawing. A missing decimal point is corrected only when the surrounding values support one interpretation. The original reading and correction source are saved in the output JSON.

## Reproducibility checks

The two weight files are supplied with the package. Their SHA-256 checksums are:

```text
stage1.pt  e39f3d979d64b73cfc7c5d4640e252b8c68f0bec9d08e0dc7bd62fde6e2c4d66
stage2.pt  788aa9ab426ce6a9af2ab9567a8e4a58edaa6cadc50c62a2a96246b77dd43090
```

The application requires an active Gemini API key for section-height extraction. The ICL reference and all section pages are sent together in PDF page order. The application does not set a time limit for the Gemini request; the progress display updates its elapsed processing time every second. A connection or service interruption triggers one retry. If Gemini fails, the run still saves the detected drawing information and reports that section heights and related risk scores are unavailable. The result JSON records time spent on PDF rendering, CV/OCR, and Gemini. Credentials are used for the current run and are not included in saved results.

Run the local geometry, parsing, and scoring checks with:

```bash
python -m unittest discover -s tests -v
```
