import numpy as np # type: ignore
import pytest # type: ignore
from project_root.dataset.representation_dataset import RepresentationDataset
from project_root.dataset.wrapped_representation_dataset import WrappedRepresentationDataset

# ---- Dummy Dataset for Testing ----
class DummyProteinDataset(RepresentationDataset):
    def __init__(self, n=5, d_embed=8, d_attn=5):
        self.embeddings = [np.random.rand(d_embed) for _ in range(n)]
        self.attention_weights = [np.random.rand(d_attn, d_attn) for _ in range(n)]
        self.ids = np.array([[f"ID_{i}"] for i in range(n)])
        self.labels = np.array([[i % 2] for i in range(n)])
        self.lengths = np.array([[np.random.randint(50, 100)] for _ in range(n)])
        self.additional_attributes = {
            'tags': np.array([[f"tag_{i}", f"tag_{i+1}"] for i in range(n)]),
        }

    def get_embeddings(self):
        return self.embeddings

    def get_attention_weights(self):
        return self.attention_weights

    def get_ids(self):
        return self.ids

    def get_labels(self):
        return self.labels

    def get_attribute(self, name):
        return self.additional_attributes[name]

    def get_lengths(self):
        return self.lengths

# ---- Test Setup ----
@pytest.fixture
def wrapped_dataset():
    dummy = DummyProteinDataset()
    return WrappedRepresentationDataset(dummy, reduce_method='pca', pca_method='threshold', threshold=0.9)

# ---- Tests ----
def test_select_embeddings_only(wrapped_dataset):
    data = wrapped_dataset.select_data(embedding=True)
    assert isinstance(data, np.ndarray)
    assert data.shape[1] > 1

def test_select_attention_only(wrapped_dataset):
    data = wrapped_dataset.select_data(attention_weights=True)
    assert isinstance(data, np.ndarray)
    assert data.shape[1] > 1

def test_select_both(wrapped_dataset):
    data = wrapped_dataset.select_data(embedding=True, attention_weights=True)
    assert isinstance(data, np.ndarray)
    assert data.shape[1] > 1

def test_select_with_ids(wrapped_dataset):
    data = wrapped_dataset.select_data(embedding=True, id_column=True)
    assert isinstance(data, np.ndarray)
    assert data.shape[1] > 1

def test_select_with_labels(wrapped_dataset):
    data = wrapped_dataset.select_data(attention_weights=True, target_column=True)
    assert isinstance(data, np.ndarray)
    assert data.shape[1] > 1

def test_select_with_additional_column(wrapped_dataset):
    data = wrapped_dataset.select_data(embedding=True, additional_columns=['tags'])
    assert isinstance(data, np.ndarray)
    assert data.shape[1] > 1

def test_select_with_length_column(wrapped_dataset):
    data = wrapped_dataset.select_data(embedding=True, length_column=True)
    assert isinstance(data, np.ndarray)
    assert data.shape[1] > 1

def test_select_with_one_hot_columns(wrapped_dataset):
    wrapped_dataset.one_hot_encode_attribute('tags')
    data = wrapped_dataset.select_data(embedding=True, one_hot_columns=['tags'])
    assert isinstance(data, np.ndarray)
    assert data.shape[1] > 1

@pytest.mark.parametrize("params", [
    {"embedding": True},
    {"attention_weights": True},
    {"embedding": True, "attention_weights": True, "target_column": True, "id_column": True, "additional_columns": ['tags']},
    {"embedding": True, "length_column": True},
    {"embedding": True, "one_hot_columns": ['tags']}
])
def test_select_combinations(wrapped_dataset, params):
    data = wrapped_dataset.select_data(**params)
    assert isinstance(data, np.ndarray)
    assert data.shape[0] > 0
    assert data.shape[1] > 0
