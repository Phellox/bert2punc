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

    def check_drift(self, data):
        out = self.model(data)
        self.test_score = self.drift_detector(out)
        self.test_p_values = self.drift_detector.compute_p_value(out)

if __name__ == "__main__":
    drift_model = DriftModel()

    drifted_data = create_dataloader(drift_data(load_dataset('test_data')))
    drift_model.check_drift(drifted_data)

    print('Score: {:0.2f}, p_val: {:0.2f}'.format(drift_model.test_score, drift_model.test_p_values))