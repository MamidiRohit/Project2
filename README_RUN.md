## How to Run the Gradient Boosted Trees Model

This guide provides step-by-step instructions on how to set up and run the Gradient Boosted Trees (GBT) model for classification and regression tasks using Python. This example uses the Iris dataset for classification and an example Concrete Data dataset for regression.

### Prerequisites

Ensure you have the following installed:

- Python 3
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- pickle (for model serialization)

### Setup and Running

#### Step 1: Clone the Repository

First, clone the repository containing the GBT code and datasets:

```bash
git clone <https://github.com/Hj006/Project2-ML.git>
```

#### Step 2: Install Required Libraries

```bash
pip install numpy pandas scikit-learn matplotlib
```

#### Step 3: Running the Model

```bash
python gbt_script.py
```

## Model Training and Evaluation

The script provides multiple options for training and evaluating models:

### Train and Save the Iris Model

- **Description**: This option trains the GBT model on the Iris dataset and saves the trained model to disk.
- **How to Run**: To run this option, choose `1` when prompted.

### Load and Plot Iris Model

- **Description**: This option loads a previously saved Iris model and plots predictions against true values.
- **How to Run**: To run this option, choose `2` when prompted.

### Train Concrete Data Model

- **Description**: This option trains the GBT model on a concrete dataset (default `Concrete_Data.xls`) and saves the model.
- **How to Run**: To run this option, choose `3` when prompted.

### Train Custom Dataset

- **Description**: This option allows you to specify a custom dataset for model training.
- **How to Run**: To run this option, choose `4` when prompted and provide the filename.

## Exiting the Program

- **How to Exit**: To exit the program, type `q` at the main menu prompt.
