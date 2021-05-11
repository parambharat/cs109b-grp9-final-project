import os

import datasets
import faiss
import numpy as np
from tqdm import tqdm_notebook
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from sklearn.preprocessing import normalize


def load_text(examples):
    """Create text features from head matter and opinion text for a batch of examples"""

    batch = [
        item[0] + "\n" + item[1]
        for item in zip(examples["head_matter"], examples["opinion_text"])
    ]
    return {"text": batch}


def load_embeddings(examples, tokenizer, model, embedding_type):
    """Tokenize and load embeddings from given huggingface pretrained model"""

    tokenized = tokenizer(
        examples["text"],
        return_tensors="tf",
        padding=True,
        truncation=True,
        max_length=512,
    )
    if embedding_type == "pooled":
        embeddings = {"embeddings": normalize(model(**tokenized)[1].numpy())}
    else:
        embeddings = {"embeddings": normalize(model(**tokenized)[0][:, 0, :].numpy())}
    return embeddings


def load_embeddings_dataset(
    dataset_dir,
    data_fname,
    embedding_type,
    embedding_model="allenai/specter",
    num_proc=15,
    batch_size=256,
    faiss_device=0,
    keep_in_memory=False,
):
    """Load embeddings dataset and create faiss index on embeddings column"""

    if os.path.isfile(f"{dataset_dir}/state.json"):
        print(f"Found existing embeddings at {dataset_dir}. loading from disk ...")
        dataset = datasets.Dataset.load_from_disk(
            dataset_dir, keep_in_memory=keep_in_memory
        )
        if os.path.isfile(f"{dataset_dir}/embeddings.faiss"):
            print(f"Found existing fiass index at {dataset_dir} loading from disk ...")
            dataset.load_faiss_index("embeddings", f"{dataset_dir}/embeddings.faiss")
        else:
            print("No fiass index found. creating and saving new index to disk ...")
            dataset.add_faiss_index(
                column="embeddings",
                device=faiss_device,
                metric_type=faiss.METRIC_INNER_PRODUCT,
            )
            dataset.save_faiss_index("embeddings", f"{dataset_dir}/embeddings.faiss")
    else:
        print("No existing embeddings found. Creating and saving to disk ...")
        tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        model = TFAutoModel.from_pretrained(embedding_model, from_pt=True)

        print("Loading dataset and text column ...")
        dataset = datasets.load_dataset(
            "json", data_files=data_fname, split=datasets.splits.Split("train")
        )
        exclude_columns = [
            "jurisdiction_id",
            "court_id",
            "decision_date",
            "head_matter",
            "opinion_text",
            "citation_ids",
        ]
        dataset = dataset.map(
            load_text, batched=True, num_proc=num_proc, remove_columns=exclude_columns
        )
        print("Loading embeddings ...")
        embedder = partial(
            load_embeddings,
            tokenizer=tokenizer,
            model=model,
            embedding_type=embedding_type.split("_")[-1],
        )
        dataset = dataset.map(embedder, batched=True, batch_size=batch_size)
        print(f"Saving Dataset to disk at {dataset_dir}")
        dataset.save_to_disk(dataset_dir)
        print("Creating new fiass index and saving to disk ...")
        dataset.add_faiss_index(
            column="embeddings",
            device=faiss_device,
            metric_type=faiss.METRIC_INNER_PRODUCT,
        )
        dataset.save_faiss_index("embeddings", f"{dataset_dir}/embeddings.faiss")
    return dataset


def retrieve_top_k_preds(database, df, k=50, key=None):
    query_embeddings = database[df.index.tolist()]["embeddings"]
    scores, samples = database.get_nearest_examples_batch(
        "embeddings", query_embeddings, k=k
    )
    if key:
        preds = [sample[key] for sample in samples]
    else:
        preds = samples
    return preds


def load_sorted_preds(embedding_dataset_dir, model_checkpoint, sample_cases):
    model = tf.keras.models.load_model(model_checkpoint)
    database = load_embeddings_dataset(
        embedding_dataset_dir,
        data_fname=None,
        embedding_type=None,
        embedding_model=None,
        keep_in_memory=True,
    )

    database.set_format("numpy", columns=["embeddings", "id"], dtype=np.float32)

    preds = retrieve_top_k_preds(database, sample_cases, key=None)
    queries = database[sample_cases.index]["embeddings"]

    unsorted_preds = []
    sorted_preds = []
    for query, item in tqdm_notebook(zip(queries, preds), total=len(sample_cases)):
        result_embeddings = item["embeddings"]
        pred = item["id"]
        num_results = len(result_embeddings)
        query_embeddings = np.array([query] * num_results)
        preds = model.predict(
            (query_embeddings, result_embeddings), batch_size=num_results
        )
        preds = preds.flatten()
        unsorted_preds.append(np.array(pred))
        sorted_preds.append(np.array(pred)[np.argsort(preds)])
    del model, database
    return sorted_preds, unsorted_preds


def load_model_paths(subset_dir, models_dir):
    embedding_types = [
        "legalbert_cls",
        "legalbert_pooled",
        "specter_cls",
        "specter_pooled",
    ]
    embedding_locs = [
        {
            "embedding_type": f"{embedding_type}",
            "dataset_dir": f"{subset_dir}/{embedding_type}_embeddings_dataset",
            "model_checkpoint": f"{models_dir}/{embedding_type}_clf_model",
        }
        for embedding_type in embedding_types
    ]

    available_embedding_locs = []
    for item in embedding_locs:
        if os.path.isfile(f'{item["dataset_dir"]}/state.json') and os.path.isdir(
            item["model_checkpoint"]
        ):
            available_embedding_locs.append(item)
    return available_embedding_locs
