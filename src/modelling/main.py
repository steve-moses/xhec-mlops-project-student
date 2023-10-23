# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse


def main(
    trainset_path: str,
    artifacts_filepath: Optional[str] = None, 
    version: Optional[str] = None ) -> dict:
    """Train a model using the data at the given path and save the model (pickle)."""
    # 

    # Read data and Preprocess data
    x_train, y_train, dv = process_data(trainset_path)
    
    # (Optional) Pickle encoder if need be
    save_pickle(os.path.join(artifacts_filepath,"dv__v",version,".pkl"),dv)

    # Train model
    model = train_model(x_train, y_train)
    
    # Pickle model --> The model should be saved in pkl format the `src/web_service/local_objects` folder
    save_pickle(os.path.join(artifacts_filepath,"model__v",version,".pkl"),model)
    
    #not sure what the function should return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)