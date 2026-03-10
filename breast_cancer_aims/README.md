# README

This notebook contains the code to reproduce the analysis and figures presented in the research paper:

**Kelly, C.J., Wilson, M., Warren, L.M. _et al._ Diagnostic accuracy, fairness and clinical implementation of AI for breast cancer screening: results of multicenter retrospective and prospective technical feasibility studies. _Nat Cancer_ (2026). https://doi.org/10.1038/s43018-026-01127-0**

## Overview

This notebook demonstrates how to generate the key performance metrics and visualizations reported in the study, which evaluated Google's mammography AI system (v1.2) across retrospective and prospective cohorts in the UK National Health Service (NHS).

The code includes plotting functions for:

* **Main Performance (Figure 2):** Generating Receiver Operating Characteristic (ROC) curves and calculating sensitivity and specificity at the case, breast, and lesion levels.
* **Subgroup Analysis (Figure 3):** assessing fairness and bias by breaking down performance across key subgroups, including age, ethnicity, breast density, and Index of Multiple Deprivation (IMD).
* **Prospective Monitoring (Figure 4):** Visualizing Cancer Detection Rate (CDR) vs. Recall Rate (RR) and monitoring performance stability over time during the prospective deployment.

## Disclaimer

The content of this research code repository (i) is not intended to be a medical device; and (ii) is not intended for clinical use of any kind, including but not limited to diagnosis or prognosis.

## License

See https://github.com/Google-Health/google-health/blob/master/LICENSE for details.
