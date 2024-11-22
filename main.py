from lib import (
    LinearRegression,
    k_fold_cross_validation,
    bootstrapping,
    generate_data,
)

def get_user_input(prompt, input_type=int, valid_choices=None):
    """
    A helper function to get user input with type checking and validation.
    """
    while True:
        try:
            user_input = input_type(input(prompt))
            if valid_choices and user_input not in valid_choices:
                print(f"Please enter one of the following options: {valid_choices}")
            else:
                return user_input
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__}.")
        

def main():
    print("Welcome to Model Selection!")
    print("To generate synthetic data give me parameter values:")
    n_samples = get_user_input("Enter number of samples: ")
    n_features = get_user_input("Enter number of features: ")
    X, y = generate_data(n_samples, n_features)

    print("\nChoose a model validation method:")
    print("1. k-Fold Cross-Validation")
    print("2. Bootstrapping")
    
    method = get_user_input("Enter your choice (1 or 2): ", input_type=str, valid_choices=["1", "2"])

    model = LinearRegression()

    if method == "1":
        k = get_user_input("Enter number of folds (k): ")
        shuffle = input("Shuffle data? (yes/no): ").strip().lower() == "yes"
        print("\nPerforming k-Fold Cross-Validation...")
        metrics, averages = k_fold_cross_validation(model, X, y, k, shuffle)
        print("\nk-Fold Cross-Validation Results:")
    elif method == "2":
        s = get_user_input("Enter size of training dataset (s): ")
        epochs = get_user_input("Enter number of epochs: ")
        print("\nPerforming Bootstrapping...")
        metrics, averages = bootstrapping(model, X, y, s, epochs)
        print("\nBootstrapping Results:")

    print("\nAverage Metrics:")
    for key, value in averages.items():
        print(f"{key.upper()}: {value}")

if __name__ == "__main__":
    main()
