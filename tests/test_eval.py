from datetime import datetime
from arizona_spotting.models import Wav2KWS
from arizona_spotting.datasets import Wav2KWSDataset
from arizona_spotting.learners import Wav2KWSLearner

def test_evaluate():
    
    test_dataset = Wav2KWSDataset(
        mode='test',
        root='./data/gsc_v2.0'
    )

    model = Wav2KWS(
        num_classes=2,
        encoder_hidden_dim=768,
        out_channels=112,
        pretrained_model='wav2vec-base-en'
    )

    learner = Wav2KWSLearner(model=model)
    learner.load_model(model_path='./models/wav2kws_model.pt')
    _, acc = learner.evaluate(
        test_dataset=test_dataset,
        batch_size=48,
        num_workers=4,
        view_classification_report=True
    )

    print(f"\nAccuracy: {acc} \n ")

test_evaluate() 