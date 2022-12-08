# -PLSAE

This repository is an implementation of the paper "Efective and Efcient Hybrid Android Malware
Classifcation Using Pseudo‑Label Stacked Auto‑Encoder" (https://link.springer.com/article/10.1007/s10922-021-09634-4).

In this paper, we propose a simple, yet practical and efcient framework for Android malware category classifcation. The new characteristics of this approach in contrast to the previous one, i.e., PLDNN are: (1) it is a
hybrid approach that integrates both static and dynamic analysis of malware and (2) it applies a semi-supervised Pseudo-Label Stacked Auto-Encoder (PLSAE) for malware category classifcation. The hybrid approach utilizes the strengths
of both static and dynamic analysis. Unlike PLDNN, PLSAE consists of stacking multiple  Auto-Encoders  (AEs) and benefts from the unsupervised pre-training that leads the network towards global minima and supports better generalization.

We analyze both statically and dynamically extracted features of malware samples using PLSAE. We adopt CopperDroid, a Virtual Machine Introspection
(VMI)-based analysis system [12], to extract the static and dynamic features. The
results of the static and dynamic analysis are available for researchers together
with malware samples (https://www.unb.ca/cic/datasets/maldroid-2020.html).

This project is implemented using Python and Tensorflow libraries.
