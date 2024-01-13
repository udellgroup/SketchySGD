import openml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def main():
    seed = 123
    ids = [41166, 40996, 40923]
    destination = './data'

    for id in ids:
        print(f"Processing dataset {id}")
        data = openml.datasets.get_dataset(id)
        X, y, _, _ = data.get_data(
            dataset_format='dataframe',
            target=data.default_target_attribute
        )

        X = X.values.astype(float) # Avoid any weird errors with int
        y = y.values

        # Find columns that are not all zeros
        non_zero_columns = np.any(X != 0, axis=0)

        # Use boolean indexing to keep only those columns
        X = X[:, non_zero_columns]

        # Get 60/20/20 split
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.6, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, train_size=0.5, random_state=seed)

        # Normalize columns
        scaler = StandardScaler()
        X_train_nrmlzd = scaler.fit_transform(X_train)
        X_val_nrmlzd = scaler.transform(X_val)
        X_test_nrmlzd = scaler.transform(X_test)

        destination_id = os.path.join(destination, str(id))
        if not os.path.exists(destination_id):
            os.makedirs(destination_id)

        np.save(os.path.join(destination_id, 'X_train.npy'), X_train_nrmlzd)
        np.save(os.path.join(destination_id, 'X_val.npy'), X_val_nrmlzd)
        np.save(os.path.join(destination_id, 'X_test.npy'), X_test_nrmlzd)
        np.save(os.path.join(destination_id, 'y_train.npy'), y_train)
        np.save(os.path.join(destination_id, 'y_val.npy'), y_val)
        np.save(os.path.join(destination_id, 'y_test.npy'), y_test)

if __name__ == "__main__":
    main()