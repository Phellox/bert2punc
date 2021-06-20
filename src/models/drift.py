from torch.utils.data import TensorDataset
from torch import nn
import torchdrift
from src.data.load_data import load_dataset, create_dataloader
from src.models.predict_model import EvalModel

def drift_data(data: TensorDataset, segment_size: int = 32):
    data.tensors[0][:,int(segment_size/2)] += 1
    return data

class DriftModel(object):
    def __init__(self, dataset: str = 'test_data'):
        self.test_data = load_dataset(dataset)
        self.test_loader = create_dataloader(self.test_data)
        self.model = EvalModel()
        self.drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
        torchdrift.utils.fit(self.test_loader, self.model, self.drift_detector, num_batches=32)

    def sequential_model(self):
        self.drift_detection_model = nn.Sequential(
            self.model,
            self.drift_detector)

if __name__ == "__main__":
    drift_model = DriftModel()

    drifted_data = create_dataloader(drift_data(load_dataset('test_data')))

    outputs = drift_model.model(drifted_data)
    score = drift_model.drift_detector(outputs)
    p_val = drift_model.drift_detector.compute_p_value(outputs)
    print('Score: {:0.2f}, p_val: {:0.2f}'.format(score, p_val))