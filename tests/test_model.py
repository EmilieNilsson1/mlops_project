import torch
from torch.utils.data import DataLoader, TensorDataset
from image_classifier.model import ImageClassifier


# Helper function to create random data for testing
def create_random_data(batch_size, input_size, num_classes):
    # Create random images (batch_size, 3, input_size, input_size) and labels (batch_size)
    images = torch.randn(batch_size, 3, input_size, input_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size)


# Test 1: Ensure model initializes correctly
def test_model_initialization():
    num_classes = 10
    model = ImageClassifier(num_classes=num_classes)

    # Check if the final layer has the correct number of outputs
    assert model.model.fc.out_features == num_classes


# Test 2: Test forward pass
def test_forward_pass():
    num_classes = 10
    model = ImageClassifier(num_classes=num_classes)

    # Create random data with a batch size of 2 and input size of 224
    data_loader = create_random_data(batch_size=2, input_size=224, num_classes=num_classes)
    images, labels = next(iter(data_loader))  # Get a batch

    # Perform a forward pass
    output = model(images)

    # Check that the output has the correct shape (batch_size, num_classes)
    assert output.shape == (2, num_classes)


# Test 3: Test training step
def test_training_step():
    num_classes = 10
    model = ImageClassifier(num_classes=num_classes)

    # Create random data with a batch size of 2 and input size of 224
    data_loader = create_random_data(batch_size=2, input_size=224, num_classes=num_classes)
    images, labels = next(iter(data_loader))  # Get a batch

    # Mock the batch_idx argument and call the training_step method
    loss = model.training_step((images, labels), batch_idx=0)

    # Ensure loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndimension() == 0  # Loss should be scalar


# Test 4: Test validation step
def test_validation_step():
    num_classes = 10
    model = ImageClassifier(num_classes=num_classes)

    # Create random data with a batch size of 2 and input size of 224
    data_loader = create_random_data(batch_size=2, input_size=224, num_classes=num_classes)
    images, labels = next(iter(data_loader))  # Get a batch

    # Mock the batch_idx argument and call the validation_step method
    loss = model.validation_step((images, labels), batch_idx=0)

    # Ensure loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndimension() == 0  # Loss should be scalar


# Test 5: Test optimizer configuration
def test_configure_optimizers():
    model = ImageClassifier(num_classes=10)

    # Check if the optimizer is an instance of Adam
    optimizers = model.configure_optimizers()
    assert isinstance(optimizers, torch.optim.Adam)
