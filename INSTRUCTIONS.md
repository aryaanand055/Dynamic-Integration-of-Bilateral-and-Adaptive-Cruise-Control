# Instructions for Running on Google Colab

Follow these steps to train the traffic agent using Google Colab.

## 1. Upload Files

Upload the following files to the Google Colab environment (click the folder icon on the left sidebar):

* `car.py`
* `city.py`
* `road.py`
* `rl_city.py`
* `accel_traffic_env.py`
* `train_td3_accel.py`

## 2. Install Dependencies

In a new code cell, run the following command to install the required libraries:

```python
!pip install gymnasium stable-baselines3 shimmy numpy
```

## 3. Run Training

Execute the training script by running the following command in a new code cell:

```python
!python train_td3_accel.py
```

This will train the TD3 direct acceleration agent for the configured number of timesteps.

## 4. Download Saved Model

After training completes, the model will be saved as `td3_accel_agent.zip`.

To download it:

1. Refresh the file browser on the left sidebar.
2. Right-click on `td3_accel_agent.zip`.
3. Select **Download**.
