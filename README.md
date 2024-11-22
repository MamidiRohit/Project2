## Team Members:
1. Badri Adusumalli A20530163
2. Bhuvana Chandrika Natharga A20553587
3. Santhosh Kumar Kathiresan A20546185
4. Sriram Ravichandran A20583347




## How to Run the Code

Follow the steps below to set up and run the code on any system. These instructions will guide you through downloading the repository, installing dependencies, and executing the tests.

### Step 1: Download the Repository

1. First, download the repository from GitHub to your local machine. You can do this by either:
   - Cloning the repository using `git clone` command (recommended):
     ```bash
     git clone https://github.com/your-username/your-repo-name.git
     ```
     Replace `your-username/your-repo-name` with the actual URL of your GitHub repository.

   - Alternatively, you can download the ZIP file from GitHub and extract it to your desired location.
  
### Step 2: Open Git Bash and Navigate to the Project Folder

1. Open **Git Bash** (or any command line terminal that supports Git) on your computer.
2. Navigate to the directory where the project is located. For example:
   ```bash
   cd ~/videos/project2
   ```
   In this example, we are assuming that the project is located in the `videos/project2` directory. Replace this path with the actual path where you have downloaded the repository.

### Step 3: Install the Required Dependencies

1. To run the project, you need to install the necessary dependencies listed in the `requirements.txt` file.
2. Use the following command to install all the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   - This command tells `pip` to install all the packages specified in the `requirements.txt` file. Make sure you have **Python** and **pip** installed on your system. If not, you will need to install them first.

### Step 4: Install the Project in "Editable" Mode Using `setup.py`

1. To allow the project to be used in any location, install it in **editable mode**. This will let Python recognize the `elasticnet` module regardless of your current working directory.
2. Run the following command:
   ```bash
   pip install -e .
   ```
   - The `-e` flag stands for "editable," which allows changes to the source code to be reflected immediately without having to reinstall the package.
   - The `.` specifies the current directory, where the `setup.py` file is located.

### Step 5: Run the Tests to Verify the Installation

1. Now that the dependencies are installed and the project is set up, you can run the tests to ensure everything is working correctly.
2. Execute the following command to run the test file:
   ```bash
   pytest -s GradientBoosting/tests/test_GradientBoosting.py
   ```
   - The `-s` flag ensures that any `print` statements in the test file are displayed in the terminal.
   - `pytest` will run the test cases defined in `test_GradientBoosting.py` to verify the functionality of your GradientBoosting implementation.

### **Step 6: Interactive Input During Testing**

After running the test command, the script will prompt you to provide necessary inputs for testing the Gradient Boosting model:

1. **Dataset File Path**:  
   - You will see the following prompt in the terminal:  
     ```
     Please enter the path to your dataset file:
     ```

2. **Target Column Name**:  
   - After entering the dataset file path, the script will display:  
     ```
     Enter the target column name:
     ```
   - Enter the name of the target column (e.g., `y`) that you wish to use as the dependent variable for training the model.


### **Overview**

This project implements a Gradient Boosting model for regression tasks. Gradient Boosting is an ensemble learning method that builds a sequence of weak learners, typically decision trees, where each new learner focuses on correcting the residual errors of the previous ones. It is a powerful and flexible technique for regression problems, known for its ability to handle complex datasets and achieve high predictive accuracy.

### **Key Features**

- **Iterative Residual Correction**: The model improves predictions iteratively by minimizing the residual errors from previous models.
- **Decision Tree Base Learners**: Utilizes decision trees as weak learners, which are combined to form a strong predictive model.
- **Learning Rate Control**: Incorporates a learning rate to manage the contribution of each tree and prevent overfitting.
- **Hyperparameter Optimization**: Supports grid search to tune key hyperparameters such as the number of estimators, learning rate, and maximum tree depth for optimal performance.
- **Robustness and Flexibility**: Handles complex data structures, making it well-suited for various regression tasks, even with non-linear relationships.  


