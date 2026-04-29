# Improving Fakeddit for Africa: Domain Adaptation for Multimodal Misinformation Detection

This repository contains the final technical-group project for multimodal misinformation detection using image-text consistency features from CLIP and a lightweight logistic regression classifier. The project began with a Fakeddit-trained baseline and then adapted that same model using African-context data so that it performs better on African examples while remaining strong on the original benchmark.

## Team

Group name: `Technical Team 1`

Team members:

- Ishimwe Karekezi Guy Gael — Andrew ID: `iguygael`
- Lynne Chepkwony — Andrew ID: `lchepkwo`
- Emile Lucky Muhigira — Andrew ID: `emuhigir`

The repository includes:

- the main experimentation notebook
- the African-context dataset and image assets used for adaptation
- a deployed Streamlit application for interactive prediction
- paper-writing materials for proposal, midterm, and final report stages
- collection and annotation documentation

## Project summary

Many misleading posts do not fabricate an image completely. Instead, they reuse a real image and attach text that changes the implied location, event, actors, or meaning. This project treats multimodal misinformation detection as a semantic consistency problem:

- the image is encoded with CLIP
- the text is encoded with CLIP
- a feature vector is built from image-text similarity and embedding differences
- a logistic regression classifier predicts whether the pair is `misinformation` or `likely_consistent`

Our final project focus is not just benchmark performance. It is model improvement:

- start from a benchmark-trained Fakeddit model
- test how it behaves on African-context data
- add African training data to adapt that same model
- check whether the improved model helps on Africa without hurting Fakeddit

## Main contributions

- A lightweight multimodal misinformation detector built from CLIP (`ViT-B/32`) and logistic regression.
- An African-context dataset of locally stored image-text examples curated for adaptation experiments.
- A comparative evaluation across four setups:
  - original model on Fakeddit
  - original model on Africa before adaptation
  - adapted model on Africa holdout
  - adapted model on Fakeddit
- A Streamlit application that demonstrates the final model and provides simple explanation outputs.

## Key results

The current notebook results used in the final report are:

| Setup | Accuracy | Macro F1 | Misinfo Precision | Misinfo Recall | Misinfo F1 |
|---|---:|---:|---:|---:|---:|
| Fakeddit -> Fakeddit | 0.8473 | 0.8457 | 0.8179 | 0.8424 | 0.8300 |
| Fakeddit -> Africa | 0.6685 | 0.6336 | 0.7619 | 0.3951 | 0.5203 |
| Fakeddit + Africa -> Africa | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 |
| Fakeddit + Africa -> Fakeddit | 0.9078 | 0.9068 | 0.8854 | 0.9094 | 0.8972 |

What these results mean:

- the original benchmark model works well in-domain on Fakeddit
- the same model is weaker on African-context examples before adaptation
- adding African data improves African misinformation recall and misinformation F1
- the adapted model remains strong on Fakeddit and even improves on the benchmark in the current run

## Repository structure

```text
MBD_Multimodal_Misinformation/
|- app.py
|- model.pkl
|- README.md
|- requirements.txt
|- packages.txt
|- runtime.txt
|- .streamlit/
|  \- config.toml
|- notebooks/
|  \- MBD_Technical_Group.ipynb
|- data/
|  |- README.md
|  |- african_context/
|  |  \- african_context_raw.csv
|  |- african_validation_images/
|  |  \- img_*.jpg
|  |- fakeddit/
|  \- fakeddit_repo/
|- demo/
|  |- adapted_model.pkl
|  |- model.pkl
|  \- README.md
|- outputs/
|- models/
|- src/
```

## Important files

### Core code and experiments

- [app.py](app.py): Streamlit application
- [notebooks/MBD_Technical_Group.ipynb](notebooks/MBD_Technical_Group.ipynb): main notebook for dataset inspection, feature extraction, training, adaptation, and evaluation
- [model.pkl](model.pkl): baseline logistic regression model

### African-context data

- [data/african_context/african_context_raw.csv](data/african_context/african_context_raw.csv): current working African-context dataset
- [African validation images (Google Drive)](https://drive.google.com/drive/folders/1wejHsr8ai5TO6URrMH0vRylmMuqvLmLh?usp=sharing): public folder containing the validation image assets used by the African dataset

## Data description

### 1. Fakeddit benchmark data

The benchmark portion of the project is based on Fakeddit, a multimodal misinformation dataset built from Reddit posts. In this project, Fakeddit provides:

- the original source-domain training signal
- the benchmark test set for in-domain evaluation
- the baseline model that we later adapt

The notebook filters the raw multimodal table to a practical modeling subset with usable text and local images.

### 2. African-context dataset

The African-context dataset is stored locally and used to measure transfer failure and adaptation benefit. It follows a one-row-per-image format. The working CSV includes fields such as:

- `id`
- `source_name`
- `language`
- `topic`
- `claim_text`
- `image_path`
- `verdict_raw`

Current project summary:

- total rows: `178`
- misinformation: `81`
- likely consistent: `97`
- African train split: `142`
- African holdout split: `36`

### 3. Labels

The task is binary:

- `misinformation`: the text reframes or misrepresents the image
- `likely_consistent`: the text appears semantically compatible with the image

## Annotation and data quality

The African-context dataset was annotated by the project team with practical safeguards:

- preference for public-scene images over sensitive close-up portraits
- avoidance of unnecessary defamation against identifiable private individuals
- emphasis on visible scene grounding when writing paired text

Annotation workflow:

1. all three annotators labeled examples independently
2. annotators did not view one another's labels during the first pass
3. final labels were assigned by majority voting
4. ambiguous examples were revisited collaboratively

Important limitation:

- the holdout set is small, so the results should be treated as directional evidence rather than a final claim about broad African media performance

## Method overview

The modeling pipeline is intentionally lightweight.

### Feature construction

For each image-text pair:

1. encode image with CLIP
2. encode text with CLIP
3. normalize embeddings
4. build a 1537-dimensional feature vector:
   - `1` cosine similarity
   - `512` absolute difference features
   - `1024` concatenated image and text embedding features

### Classifier

The downstream classifier is a `LogisticRegression` model. This was chosen because it is:

- lightweight
- easy to retrain
- easier to interpret than a heavier end-to-end architecture
- strong enough in the project's benchmark experiments

## Adaptation workflow

The adaptation process is centered on improving the original Fakeddit model.

1. Train the original model on the Fakeddit training split.
2. Evaluate it on the Fakeddit test set.
3. Evaluate that same model on the African dataset before adaptation.
4. Split the African dataset into:
   - African-train
   - African-holdout
5. Retrain the same pipeline on:
   - Fakeddit train
   - plus African-train
6. Evaluate the adapted model on:
   - African holdout
   - Fakeddit test
7. Export the adapted classifier for demo use.

## Notebook guide

The main experimentation notebook is:

- [notebooks/MBD_Technical_Group.ipynb](notebooks/MBD_Technical_Group.ipynb)

The final project flow after the earlier benchmark stages is organized around the African adaptation story.

Key sections include:

- African dataset summary
- sample African images with text and labels
- African feature construction
- original model on Fakeddit
- original model on Africa before adaptation
- African train/holdout split
- tuning on Fakeddit + Africa-train
- adapted model on African holdout
- adapted model on Fakeddit
- final comparison summary
- adapted model export for the app

## Running the project locally

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open:

- `notebooks/MBD_Technical_Group.ipynb`

and rerun the cells from the African dataset section onward if you want the latest CSV and image updates to take effect.

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

## Streamlit app

The app is a lightweight demo for the trained model.

Inputs:

- one uploaded image
- one text string such as a caption, claim, or post

Outputs:

- predicted label
- misinformation probability
- risk level
- confidence message
- simple feature-group contribution summary
- leave-one-word-out word influence estimates

### Model-loading behavior

`app.py` prefers an `adapted_model.pkl` located beside the app if one exists. Otherwise, it falls back to `model.pkl`.

Current artifact roles:

- `model.pkl` at the project root is the root app's current fallback model
- `demo/adapted_model.pkl` is the final adapted model artifact produced by the notebook workflow

If you want the root Streamlit app to use the final adapted model directly, copy or export `demo/adapted_model.pkl` to the project root as `adapted_model.pkl`, beside `app.py`.

## Deployment notes

This repository is set up for Streamlit deployment.

Relevant files:

- `requirements.txt`
- `packages.txt`
- `runtime.txt`
- `.streamlit/config.toml`

To deploy on Streamlit Community Cloud:

- set the repository to this GitHub project
- set the branch to `main`
- set the main file path to `app.py`

## Reproducibility

To reproduce the final adaptation results as closely as possible:

1. ensure the African CSV and image paths are present locally
2. open the notebook
3. rerun the African-context cells from the dataset summary onward
4. retrain the adapted model
5. rerun the final comparison cells
6. optionally export the adapted model for the app

Core assumptions:

- CLIP backbone: `ViT-B/32`
- device: CPU in the app
- feature dimension: `1537`
- classifier family: logistic regression

## Limitations

- The system estimates misinformation risk rather than verifying factual truth.
- The African dataset is still relatively small and partly manually curated.
- The holdout set is small, so the results are best treated as promising directional evidence.
- Performance on underrepresented languages, countries, or unseen misinformation styles may differ.
- A good benchmark result does not guarantee broad real-world fairness or transfer.

## Ethical considerations

- The project involves politically and socially sensitive image-text pairs.
- Misclassification can create both false positives and false negatives.
- The tool should support human review, not replace it.
- The African dataset should be interpreted carefully because it reflects limited local curation rather than continent-wide coverage.

## Acknowledgment of current practical issue

One practical detail to keep in mind is that the app and the exported adapted model should live in consistent locations. At the moment, we still have both:

- a root-level baseline model used by `app.py`
- a `demo/` copy of the adapted model

This is acceptable for archival purposes, but for a cleaner final deployment we should standardize the model path so the app always loads the intended final adapted artifact.

## License

The code in this repository is released under the GNU AGPLv3 License. See [LICENSE](LICENSE).

Important note:

- the AGPLv3 license applies to the project code and documentation written by the team
- third-party datasets, images, and external materials may remain subject to their own original licenses or usage terms
- anyone reusing the data assets should verify those source-specific conditions separately

## Citation

If this repository or its code is reused, the project should be credited to Technical Team 1. Citation metadata is provided in [CITATION.cff](CITATION.cff).




