import torch.nn as nn

from typing import List

from ctcdecode import CTCBeamDecoder


class CTCDecoder:
    def __init__(self, blank_id: int, alphabet: List[str], count_prediction=10):
        self.decoder = CTCBeamDecoder(alphabet, beam_width=count_prediction, blank_id=blank_id)

    def __call__(self, output):
        result, _, _, sec_len = self.decoder.decode(output)
        len_best_result = sec_len[:, 0]
        best_results = []
        for i, res in enumerate(result):
            best_results.append(res[0, :len_best_result[i]])

        tensor_res = nn.utils.rnn.pad_sequence(sequences=best_results, batch_first=True)
        return tensor_res, len_best_result
