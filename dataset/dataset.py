import yaml

def load_dataset(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)

    # Create the clean dictionary your code already knows how to use
    dataset = {
        "question": [],
        "user_input": [],
        "ground_truth": [],
        "reference": [],
        "contexts": [],
        "retrieved_contexts": [],
        "difficulty": [],
        "answer_type": [],
        "test_set": [],
    }

    # Extract only what Ragas needs
    for item in yaml_data:
        dataset["question"].append(item["question"])
        dataset["user_input"].append(item["question"])
        # Note: Ragas specifically looks for the key 'ground_truth', not 'answer'
        dataset["ground_truth"].append(item["ground_truth"])
        dataset["reference"].append(item["ground_truth"])
        
        dataset["difficulty"].append(item["difficulty"])
        dataset["answer_type"].append(item["answer_type"])
        dataset["test_set"].append(item["test_set"])

    return dataset


