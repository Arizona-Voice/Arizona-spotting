from datetime import datetime
from denver_spotting.models import Wav2KWS
from denver_spotting.learners import Wav2KWSLearner

def test_inference():

    model = Wav2KWS(
        num_classes=2,
        encoder_hidden_dim=768,
        out_channels=112,
        pretrained_model='wav2vec-base-en'
    )

    learner = Wav2KWSLearner(model=model)
    learner.load_model(model_path='./models/wav2kws_model.pt')

    now = datetime.now()

    output = learner.inference(input='data/gsc_v2.0/test/non_active/0c540988_nohash_2.wav')
    
    print(output)

    print(f"\nInference time: {(datetime.now() - now) * 1000} ms")
    
test_inference()