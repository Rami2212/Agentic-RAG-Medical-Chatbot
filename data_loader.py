import pandas as pd
import chromadb
from config import (
    QNA_DATA_PATH,
    DEVICE_DATA_PATH,
    SAMPLE_SIZE,
    RANDOM_STATE,
    CHROMA_DB_PATH,
    QNA_COLLECTION_NAME,
    DEVICE_COLLECTION_NAME,
)

collection1 = None
collection2 = None


def load_and_prepare_data():
    """Load and prepare datasets."""
    df_qa = pd.read_csv(QNA_DATA_PATH)
    df_qa = df_qa.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

    df_qa["combined"] = (
        "Question: " + df_qa["Question"].astype(str) + ". " +
        "Answer: " + df_qa["Answer"].astype(str) + ". " +
        "Type: " + df_qa["qtype"].astype(str) + ". "
    )

    df_md = pd.read_csv(DEVICE_DATA_PATH)
    df_md = df_md.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

    df_md["combined"] = (
        "Device Name: " + df_md["Device_Name"].astype(str) + ". " +
        "Model: " + df_md["Model_Number"].astype(str) + ". " +
        "Manufacturer: " + df_md["Manufacturer"].astype(str) + ". " +
        "Indications: " + df_md["Indications_for_Use"].astype(str) + ". " +
        "Contraindications: " + df_md["Contraindications"].fillna("None").astype(str)
    )

    return df_qa, df_md


def initialize_chromadb():
    """Initialize ChromaDB collections."""
    global collection1, collection2

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    collection1 = client.get_or_create_collection(name=QNA_COLLECTION_NAME)

    collection2 = client.get_or_create_collection(name=DEVICE_COLLECTION_NAME)

    return collection1, collection2


def populate_collections(df_qa, df_md):
    """Populate ChromaDB collections with data."""
    global collection1, collection2

    if collection1 and collection2:
        try:
            collection1.delete(where={})
        except:
            pass
        try:
            collection2.delete(where={})
        except:
            pass

    qa_metadatas = []
    for _, row in df_qa.iterrows():
        meta = {k: str(v) for k, v in row.to_dict().items()}
        qa_metadatas.append(meta)

    collection1.add(
        documents=df_qa["combined"].tolist(),
        metadatas=qa_metadatas,
        ids=df_qa.index.astype(str).tolist()
    )

    md_metadatas = []
    for _, row in df_md.iterrows():
        meta = {k: str(v) for k, v in row.to_dict().items()}
        md_metadatas.append(meta)

    collection2.add(
        documents=df_md["combined"].tolist(),
        metadatas=md_metadatas,
        ids=df_md.index.astype(str).tolist()
    )


def get_collections():
    """Get the initialized collections."""
    return collection1, collection2


def init_data(force_reload=False):
    """Initialize data loading and ChromaDB."""
    global collection1, collection2

    if collection1 is None or collection2 is None or force_reload:
        df_qa, df_md = load_and_prepare_data()
        initialize_chromadb()
        populate_collections(df_qa, df_md)

    return get_collections()