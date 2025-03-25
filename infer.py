import sys
import torch
import scipy.io.wavfile as wavfile
from train import load_model
from params import create_hparams
from text import text_to_sequence


if __name__ == "__main__":
    path = sys.argv[1]
    hparams = create_hparams()

    symbols = list('_-!\'(),.:;? ')
    with open(hparams["table_file"]) as file:
        symbols += [line.strip().split("\t")[1] for line in file]

    model = load_model(hparams)
    model.load_state_dict(torch.load(path, map_location='cpu')['state_dict'])
    model.eval().cuda()

    hifigan, vocoder_train_setup, denoiser = torch.hub.load(
        'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan')
    hifigan.eval().cuda()
    denoiser.eval().cuda()

    repeat = True
    with torch.no_grad():
        while repeat:
            text = input("Prompt:")
            sequence = torch.IntTensor([text_to_sequence(text.strip(), symbols)])
            mel = model.inference(sequence.cuda())[1]
            audio = hifigan(mel).float()
            audio = denoiser(audio.squeeze(1), 0.005)[0, 0].cpu().numpy()
            wavfile.write("result.wav", hparams["sampling_rate"], audio)
        