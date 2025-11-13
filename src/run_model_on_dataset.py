from model import Model
import pandas as pd

RESULTS_PATH = ""
BATCH_SIZE = 8

def run_model_on_dataset():
    """
    This function loops over all datasets, their categories and subcategories
    """

    # TODO: Somehow loop over dataset so the questions can be processed at least in batches in 8
    datasets = []
    results_df = pd.DataFrame()

    model = Model()

    for dataset in datasets:
        for categorie in dataset:
            for subcategorie in categorie:
                # TODO: Put categories, subcategories and questins into lists
                dataframe_to_append = model.run_batch_and_compute_confidence(
                    dataset_name="", 
                    categories = [], 
                    subcategories = [],
                    questions = []
                    )
                results_df.concat([results_df, dataframe_to_append])
    
    # TODO: save results_dfto RESULTS_PATH


if __name__ == "__main__":
    run_model_on_dataset()