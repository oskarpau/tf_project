from model import Model
import pandas as pd

RESULTS_PATH = "initial_results.csv"

def run_model_on_dataset():
    """
    This function loops over all datasets, their categories and subcategories
    """

    # TODO: Somehow loop over the datasets
    datasets = [[1]]
    results_df = pd.DataFrame()

    model = Model()

    for dataset in datasets:
        for row in dataset:
            dataframe_to_append = model.run_batch_and_compute_confidence(
                dataset_name="dummy", 
                categories = ["dummy_categorie"], 
                subcategories = ["dummy_subcategorie"],
                questions = ["What is equal to 2 + 2?"],
                right_answers = [["4"]],
                question_types = ["multi_str"]
                )
            results_df = pd.concat([results_df, dataframe_to_append])
    
    results_df.to_csv(RESULTS_PATH, index=False, sep=";")


if __name__ == "__main__":
    run_model_on_dataset()