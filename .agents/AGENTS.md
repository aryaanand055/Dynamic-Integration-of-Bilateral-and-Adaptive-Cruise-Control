# Documentation Logging

Always document and append new model training phases, parameter changes, and their results to the l_training_log.md artifact after completing any retraining tasks.
To ensure the user can debug later, these logs MUST be highly detailed. Always include:

- Exact hyperparameter changes or reward function weight updates
- Relevant code logic or formula changes (e.g., physics/geometry fixes)
- The exact TensorBoard run ID (e.g., TD3_21) and the saved model path
- Quantitative post-training evaluation metrics (e.g., Mean Gap, Mean Speed, Mean Jerk, Collisions)
- All emergent behaviors or side effects noticed during evaluation
