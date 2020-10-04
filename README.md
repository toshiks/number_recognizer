### Numbers
Solution of a test task from the "Myna Labs" for a research 
project "Generative adversarial Denoiser".

### Requirements

* Python 3.8+
* [Conda](https://docs.conda.io/en/latest/)
* Nvidia GPU (With around 1GB memory)

### Installation 

Create new conda environment by command: 

```bash
./build_env.sh
```

### Description of solution

This aproach contains Deep Learning model looks like [DeepSpeech](https://arxiv.org/abs/1412.5567).
For training that need some data preparation: 
 * raw audio files was converted to MelSpectrogram by 
[torchaudio](https://pytorch.org/audio/transforms.html#melspectrogram) 

 * labels was converted from numbers to words by [num2words](https://github.com/divan/num2words). 
 
The error function was CTCLoss. So, current training pipeline: 
reading csv file and parsing from that path to audio and number, 
preparing data, passing to model, calculating loss.

Also, for validation was used metrics such as CER, WER for word sequences. 
Of course, model outputs need in decoding, so the [CTCDecoder](https://github.com/parlance/ctcdecode)
does this.

When word sequence is received, it's converted to number. 

### How to train

In ```conf``` folder rename ```template_train_conf.yaml``` to ```train_conf.yaml```.
Fill ```paths``` and ```train_config``` fields. After that run in terminal: 

```bash
conda activate myna
python train.py
```

You can run tensorboard and seeing losses and metrics there.

### How to test

In ```conf``` folder rename ```template_test_conf.yaml``` to ```test_conf.yaml```.
Fill ```dataset_path``` and ```model_config``` fields. After that run in terminal: 

```bash
conda activate myna
python test.py
```

The result will be in the folder with ```test.csv```.