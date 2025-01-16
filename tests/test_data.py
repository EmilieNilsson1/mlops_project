from torch.utils.data import Dataset

from src.image_classifier.data import Datahandler


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = Datahandler('data/processed', 'data/raw')
    assert isinstance(dataset, Dataset)
