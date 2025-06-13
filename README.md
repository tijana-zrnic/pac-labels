# Probably Approximately Correct (PAC) Labels  
By [Emmanuel Candes](https://candes.su.domains/), [Andrew Ilyas](https://www.andrewilyas.com/), [Tijana Zrnic](https://tijana-zrnic.github.io/)

> **Official repository for the paper "[Probably Approximately Correct (PAC) Labels](https://arxiv.org/abs/2506.10908)."**  

PAC Labeling is a framework for collecting **statistically sound** pseudo-labels while reducing costly human annotation. Given a target error budget $$\varepsilon$$ and confidence $$1-\alpha$$, the algorithm decides which examples must be sent to an oracle (e.g. a human annotator) and which can be **auto-labeled** by a model—guaranteeing the overall labeling error is bounded by $$\varepsilon$$ with probability $$1-\alpha$$.

---

[[Paper](https://arxiv.org/abs/2506.10908)] [[Quick start](#quick-start)] [[Usage](#usage-in-a-nutshell)]

---

## Citation
```bibtex
@inproceedings{paclabels2025,
  title        = {Probably Approximately Correct (PAC) Labels},
  author       = {Emmanuel J. Candès and Andrew Ilyas and Tijana Zrnic},
  year         = {2025},
  booktitle    = {Arxiv preprint arXiV:2506.10908},
}
```

## Quick start
1. Clone the repo
   ```bash
   git clone https://github.com/your-username/pac-labels.git
   cd pac-labels
   ```
2. (Optional) create a fresh environment and install the few Python dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt  # or: pip install numpy scipy torch tqdm
   ```
3. Run the demonstration notebooks:
   * **`pac_labels_single_model.ipynb`** – PAC-labeling for a single classifier.
   * **`pac_router.ipynb`** – Using PAC Labels to route between multiple models/expert-oracle.

If you prefer Colab, open the notebook links above directly in the browser – they will automatically fetch the latest version of the repo.

## Usage in a nutshell
Below is the core API for PAC labeling a dataset given model predictions and per-example uncertainties:
```python
from pac_utils import pac_labeling, zero_one_loss

# Y: ground-truth labels for the subset you already queried
# Yhat: model predictions for every example
# loss: a loss function, e.g. zero_one_loss
# epsilon, alpha: PAC parameters
# uncertainty: scalar uncertainty per example (the lower, the more confident)
# pi: prior probabilities of requesting a label (can be uniform)
# num_draws: Monte-Carlo samples for the PAC bound
# asymptotic: whether to use the asymptotic (CLT-based) or non-asymptotic (betting-based) CIs
Y_tilde, labeled_mask, frac_unique = pac_labeling(
    Y, Yhat, loss=zero_one_loss,
    epsilon=0.05,  # target error
    alpha=0.05,    # confidence level
    uncertainty=uncertainty,
    pi=0.2*np.ones_like(Yhat),
    num_draws=500,
    asymptotic=False
)
# Y_tilde is the final labels
```
For a deeper dive take a look at the notebooks and at `pac_utils.py`.

## Repository structure
```
pac-labels/
├── datasets/                  # Example datasets (synthetic & real)
├── pac_utils.py               # Core PAC labeling implementation
├── router_utils.py            # Utilities for model routing with PAC guarantees
├── plotting_utils.py          # Helpers for paper figures
├── *.ipynb                    # Reproducibility notebooks
└── README.md
```

## License
This project is released under the MIT License – see the [LICENSE](LICENSE) file for details.


