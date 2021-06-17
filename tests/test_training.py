from datetime import datetime
from denver_spotting.models import Wav2KWS
from denver_spotting.datasets import Wav2KWSDataset
from denver_spotting.learners import Wav2KWSLearner

def test_train():

    train_dataset = Wav2KWSDataset(
        mode='train',
        root='./data/gsc_v2.0/'
    )
    test_dataset = Wav2KWSDataset(
        mode='test',
        root='./data/gsc_v2.0/'
    )

    model = Wav2KWS(
        num_classes=2,
        encoder_hidden_dim=768,
        out_channels=112,
        pretrained_model='wav2vec-base-en'
    )
    
    learner = Wav2KWSLearner(model=model)
    learner.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=48,
        encoder_learning_rate=1e-5,
        decoder_learning_rate=5e-4,
        weight_decay=1e-5,
        n_epochs=100,
        num_workers=4,
        shuffle=True,
        view_model=True,
        save_path='./models',
        model_name='wav2kws_model'
    )

test_train()
