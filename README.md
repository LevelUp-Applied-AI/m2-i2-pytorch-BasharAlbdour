[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YUvA8hIt)
# Integration 2 — PyTorch: Housing Price Prediction

**Module 2 — Programming for AI & Data Science**

See the [Module 2 Integration Task Guide](https://levelup-applied-ai.github.io/aispire-14005-pages/modules/module-2/learner/integration-guide) for full instructions.

---

## Quick Reference

**File to complete:** `train.py`

**Install PyTorch before running:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Branch:** `integration-2/pytorch`

**Submit:** PR URL → TalentLMS Unit 8 text field
---
## Model Overview

# What the Model Predicts
The model predicts housing prices in Jordanian Dinars (JOD).
-Target variable:
  ```price_jod — the actual property price```

-Input features (5):
    ```area_sqm``` — property size in square meters
    ```bedrooms``` — number of bedrooms
    ```floor``` — floor number
    ```age_years``` — age of the property
    ```distance_to_center_km``` — distance from the city center

# Training Configuration
    Number of epochs: 100
    Learning rate: 0.01
    Optimizer: Adam
    Loss function: Mean Squared Error (MSELoss)

# Training Outcome

The loss consistently decreased during training, indicating that the model was learning.

    Initial loss: ~1,950,638,080
    Final loss: ~1,945,380,736

# Behavioral Observation

The loss decreased gradually and slowly across epochs, rather than dropping sharply early in training. This suggests that learning was stable but not very efficient, likely due to the large scale of the target variable (```price_jod```) compared to the standardized input features.
