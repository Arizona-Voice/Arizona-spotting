## Table of contents

1. [Instroduction](#introduction)
2. [How to use `Arizona-spotting`](#how_to_use)
   - [Installation](#installation)
   - [Data structrue](#data_structure)
   - [Example usage](#usage)
4. [Reference](#reference)

# <a name='introduction'></a> Arizona-spotting

Arizona-spotting is a library provide Transformer architectures for Keyword spotting problem. Keywords spotting (in other words, Voice Trigger or Wake-up Words Detection) is a very important research problem, is used to detect specific-words from a stream of audio, typically in a low-power always-on setting such as smart speakers and mobile phones or detect profanity-words in live-streaming.

We provide two main SoTA architectures:

1. Keywords-Transformer [1]

2. Wav2KWS [2]

# <a name='how_to_use'></a> How to use Arizona-spotting

## Installation <a name='installation'></a>

```js
>>> git@github.com:phanxuanphucnd/Arizona-spotting.git

>>> cd Arizona-spotting

>>> python setup.py bdist_wheel

>>> pip install dist/arizona_spotting-0.0.1-py3-none-any.whl 

>>> pip install dist/fairseq-1.0.0a0+9b5b09b-cp36-cp36m-linux_x86_64.whl
```

## <a name='data_structure'></a> Data Structure

```
data
    gsc_v2.1
        train
            active
                right_1.wav
                right_2.wav
                ...
            
            non_active
                on_1.wav
                on_2.wav
                ...

            ...

        valid

        test

```

## <a name='usage'></a> Example usage

### Training

```py
from arizona_spotting.models import Wav2KWS
from arizona_spotting.datasets import Wav2KWSDataset
from arizona_spotting.learners import Wav2KWSLearner

train_dataset = Wav2KWSDataset(
    mode='train',
    root='./data/gsc_v2.1/'
)
test_dataset = Wav2KWSDataset(
    mode='test',
    root='./data/gsc_v2.1/'
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
    max_steps=10,
    n_epochs=100,
    num_workers=4,
    shuffle=True,
    view_model=True,
    save_path='./models',
    model_name='wav2kws_model'
)
```

### Evaluation

```py
from arizona_spotting.models import Wav2KWS
from arizona_spotting.datasets import Wav2KWSDataset
from arizona_spotting.learners import Wav2KWSLearner
    
test_dataset = Wav2KWSDataset(
    mode='test',
    root='./data/gsc_v2.1'
)

model = Wav2KWS(
    num_classes=2,
    encoder_hidden_dim=768,
    out_channels=112,
    pretrained_model='wav2vec-base-en'
)

learner = Wav2KWSLearner(model=model)
learner.load_model(model_path='./models/wav2kws_model_0.97_v2.1.pt')
_, acc = learner.evaluate(
    test_dataset=test_dataset,
    batch_size=48,
    num_workers=4,
    view_classification_report=True
)

print(f"\nAccuracy: {acc} \n ")
```

### Inference

```py
from datetime import datetime
from arizona_spotting.models import Wav2KWS
from arizona_spotting.datasets import Wav2KWSDataset
from arizona_spotting.learners import Wav2KWSLearner


model = Wav2KWS(
    num_classes=2,
    encoder_hidden_dim=768,
    out_channels=112,
    pretrained_model='wav2vec-base-en'
)

learner = Wav2KWSLearner(model=model)
learner.load_model(model_path='./models/wav2kws_model.pt')

now = datetime.now()

output = learner.inference(input='data/gsc_v2.1/test/active/5f01c798_nohash_1.wav')

print(output)

print(f"\nInference time: {(datetime.now() - now) * 1000} ms")
```


# <a name='reference'></a> Reference

[1] Axel Berg, Mark O'Connor and Miguel Tairum Cruz: “[Keyword Transformer: A Self-Attention Model for Keyword Spotting](https://arxiv.org/pdf/2104.00769v2.pdf)”, in arXiv:2104.00769, 2021.

[2] D.J Seo, H.S Oh and Y.C Jung: “[Wav2KWS: Transfer Learning from Speech Representations for Keyword Spotting](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9427206)”, in IEEE 2021.

[3] Alexei Baevski, Henry Zhou, Abdelrahman Mohamed and Michael Auli: “[wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/pdf/2006.11477.pdf)”, in arXiv:2006.11477, 2020.

# License

      MIT License

      Copyright (c) 2021 Phuc Phan

      Permission is hereby granted, free of charge, to any person obtaining a copy
      of this software and associated documentation files (the "Software"), to deal
      in the Software without restriction, including without limitation the rights
      to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
      copies of the Software, and to permit persons to whom the Software is
      furnished to do so, subject to the following conditions:

      The above copyright notice and this permission notice shall be included in all
      copies or substantial portions of the Software.

      THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
      IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
      FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
      AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
      LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
      OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
      SOFTWARE.

  
# Author

Arizona-spotting was developed by Phuc Phan © Copyright 2021.

For any questions or comments, please contact the following email: phanxuanphucnd@gmail.com
