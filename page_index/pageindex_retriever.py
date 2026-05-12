import os
import json
from pageindex import PageIndexClient
import pageindex.utils as utils
from utils.model_factories import create_default_model


def retrieve_dataset(doc_id, dataset):
    if "PAGE_INDEX_API_KEY" not in os.environ:
        raise "missing PAGE_INDEX_API_KEY"
   
    pi_client = PageIndexClient(api_key=os.environ.get("PAGE_INDEX_API_KEY"))
    llm = create_default_model()

    if not pi_client.is_retrieval_ready(doc_id):
        raise "Document was not processed"
    
    tree = pi_client.get_tree(doc_id, node_summary=True)['result']
    for query in dataset["question"]:
        dataset["contexts"].append([ retrieve(tree, llm, query) ])
        dataset["retrieved_contexts"].append(dataset["contexts"])
    return dataset

def retrieve(tree, llm, query):
    tree_without_text = utils.remove_fields(tree.copy(), fields=['text'])

    search_prompt = f"""
    You are given a question and a tree structure of a document.
    Each node contains a node id, node title, and a corresponding summary.
    Your task is to find all nodes that are likely to contain the answer to the question.

    Question: {query}

    Document tree structure:
    {json.dumps(tree_without_text, indent=2)}

    Please reply in the following JSON format:
    {{
        "thinking": "<Your thinking process on which nodes are relevant to the question>",
        "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
    }}
    Directly return the final JSON structure. Do not output anything else.
    """

    tree_search_result = llm.invoke(search_prompt).text
    node_list = json.loads(tree_search_result)["node_list"]
    node_map = utils.create_node_mapping(tree)
    return "\n\n".join(node_map[node_id]["text"] for node_id in node_list)
