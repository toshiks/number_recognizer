import os

import pandas as pd
import torch
import torchaudio

from app.dataset import get_alphabet, get_blank_id, text_to_number, get_index_to_letter_map
from app.dataset.preprocessing import LogMelSpectrogram
from app.model import CTCDecoder, RecognitionModule


class RecognizeNumbers:
    def __init__(self, path_to_graph, num_mels):
        alphabet = list(get_alphabet())
        blank_id = get_blank_id()

        self._index_to_letter_map = get_index_to_letter_map()
        self._device = 'cuda:0'
        self._model = self._load_model(path_to_graph, num_mels, len(alphabet)).to(self._device)
        self._decoder = CTCDecoder(blank_id=blank_id, alphabet=alphabet)
        self._transform = LogMelSpectrogram(n_mels=num_mels)

    @staticmethod
    def _load_model(path_to_graph, num_mels, num_classes):
        model = RecognitionModule(num_mels, out_classes=num_classes)
        state_dict = torch.load(path_to_graph)['state_dict']
        model.load_state_dict({i.replace('model.', ''): j for i, j in state_dict.items()})

        return model

    def _read_raw_audio(self, path_to_audio):
        waveform, _ = torchaudio.load(path_to_audio)
        return self._transform(waveform).to(self._device)

    def _predict_number(self, spectrogram):
        hidden = self._model.get_init_hidden(1, self._device)
        out, _ = self._model(spectrogram, hidden)
        out = torch.nn.functional.softmax(out, dim=2)
        out = out.transpose(0, 1)

        prediction_indices, _ = self._decoder(out)
        prediction_indices = prediction_indices[0].detach().cpu().numpy()
        prediction_text = ''.join([self._index_to_letter_map[x] for x in prediction_indices])
        return text_to_number(prediction_text)

    def _predict(self, path_to_audio):
        spectrogram = self._read_raw_audio(path_to_audio)
        return self._predict_number(spectrogram)

    def prediction(self, path_to_csv):
        predictions = []

        table = pd.read_csv(path_to_csv)
        file_paths = table['path'].to_list()
        directory = os.path.dirname(path_to_csv)

        for path in file_paths:
            full_path = os.path.join(directory, path)
            predictions.append(self._predict(full_path))

        pd.DataFrame({
            'path': file_paths,
            'number': predictions
        }).to_csv(os.path.join(directory, 'recognition_output.csv'), index=False)
