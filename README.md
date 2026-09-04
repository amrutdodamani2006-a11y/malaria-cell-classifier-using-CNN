# \# 🔬 Malaria Cell Classifier



A deep learning model that detects malaria-infected blood cells from microscopy images, built with transfer learning (ResNet34) and deployed as a live interactive web app.



\*\*🔗 Live Demo:\*\* \[malaria-classifier-amrut.streamlit.app](https://malaria-classifier-amrut.streamlit.app)



#### \## Problem Statement



Malaria diagnosis traditionally relies on manual microscopic examination of blood smears by trained professionals — a process that's time-consuming and prone to human error, especially in resource-limited settings. This project explores whether a CNN can assist in screening blood cell images as \*\*Parasitized\*\* or \*\*Uninfected\*\*, potentially supporting faster preliminary screening.



\## Dataset



\- \*\*Source:\*\* \[NIH Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) (Kaggle)

\- \*\*Size:\*\* 27,560 images (13,780 per class), balanced dataset

\- \*\*Classes:\*\* Parasitized, Uninfected



\## Approach



\- \*\*Architecture:\*\* ResNet34 (transfer learning, pretrained on ImageNet), fine-tuned using fastai

\- \*\*Preprocessing:\*\* Resize to 224×224, normalization, data augmentation (flip, rotate, zoom)

\- \*\*Training:\*\* 70/15/15 train/validation/test split, fine-tuned for 4 epochs



\## Results



| Metric | Score |

|---|---|

| Validation Accuracy | 97.77% |

| Recall (Parasitized) | 97.2% |

| Precision (Parasitized) | 98.3% |



Recall was prioritized as the key metric, since minimizing missed infections matters more than occasional false alarms in a screening context.



\*\*Confusion Matrix:\*\*



| | Predicted: Parasitized | Predicted: Uninfected |

|---|---|---|

| \*\*Actual: Parasitized\*\* | 5,298 | 155 |

| \*\*Actual: Uninfected\*\* | 91 | 5,479 |



\## Explainability — Grad-CAM



To verify the model learns meaningful patterns (not artifacts), Grad-CAM heatmaps were generated to visualize which regions influence predictions:



!\[Grad-CAM Example](gradcam\_example.png)



The heatmap confirms the model focuses on the actual parasite region, not irrelevant image features.



\## Tech Stack



`Python` `PyTorch` `fastai` `torchvision` `scikit-learn` `Streamlit` `Grad-CAM` `Git/GitHub`



\## How to Run Locally



```bash

git clone https://github.com/amrutdodamani2006-a11y/malaria-cell-classifier-using-CNN.git

cd malaria-cell-classifier-using-CNN

pip install -r requirements.txt

streamlit run app.py

```



\## Future Improvements



\- Expand to multi-class classification (different parasite stages)

\- Add model versioning and experiment tracking (e.g., Weights \& Biases)

\- Improve UI with batch image upload support



\## Author



\*\*Amrut Dodamani\*\*

\[GitHub](https://github.com/amrutdodamani2006-a11y)

