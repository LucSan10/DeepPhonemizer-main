from dp.phonemizer import Phonemizer

if __name__ == '__main__':

    phonemizer = Phonemizer.from_checkpoint(
    './checkpoints/autoreg/batch-8/model-512/fft-1024/layers-4/lr-5e-05/test-4/cv_7/best_model.pt')

    fullText = 'mão.'
    texts = fullText.split(". ")
    for text in texts:
        result = phonemizer.phonemise_list([text], lang='pt_br')

        print(result.phonemes)
        for text, pred in result.predictions.items():
            tokens, probs = pred.phoneme_tokens, pred.token_probs
            for o, p in zip(tokens, probs):
                print(f'{o} {p}')
            tokens = ''.join(tokens)
            print(f'{text} | {tokens} | {pred.confidence}')

# u~ 'dZi.a vo'se a'pre~.dZi ki ajs veR.da'dej.rajs a.mi'za.dZis ko~.tSi~'nu.a~w a kreseX, 'mez.mu a 'lo~.gajs dZis'ta~.siajs, i u ki i~'pOX.ta 'na~w Eu ki vo.se te~j.na 'vi.da, majs 'ke~m te~j.na 'vi.da, a'pre~.dZi ki 'na~w 'te~.mus ki mu'daR dZi a'mi.gus, si ko~.pre.e~'dER.mus ki uz a'mi.gus 'mu.da~w
# u~ 'dZi.a vo'se a'pre~.dZi ki ajs veR.da'dej.rajs a.mi'za.dZis ko~.tSi~'nu.a~w~ a kreseX, 'mez.mu a 'lo~.gajs dZis'ta~.siajs, i u ki i~'pOX.ta 'na~w~ Eu ki vo.se te~j~.na 'vi.da, majs 'ke~m te~j~.na 'vi.da, a'pre~.dZi ki 'na~w~ 'te~.mus ki mu'daR dZi a'mi.gus, si ko~.pre.e~'dER.mus ki uz a'mi.gus 'mu.da~w~