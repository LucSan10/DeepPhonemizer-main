from dp.phonemizer import Phonemizer

if __name__ == '__main__':

    phonemizer = Phonemizer.from_checkpoint(
    './checkpoints/autoreg/batch-8/model-64/fft-64/best_model.pt')

    text = 'Então, que desculpa melhor para ficarmos juntos?'
    result = phonemizer.phonemise_list([text], lang='pt_br')

    print(result.phonemes)
    for text, pred in result.predictions.items():
        tokens, probs = pred.phoneme_tokens, pred.token_probs
        for o, p in zip(tokens, probs):
            print(f'{o} {p}')
        tokens = ''.join(tokens)
        print(f'{text} | {tokens} | {pred.confidence}')

# "'na~w~ 'tSi.du 'pa.ra 'pa.ra 'pa~w~ 'pa.tSi 'pa.ra 'se.ra 'se~w~"