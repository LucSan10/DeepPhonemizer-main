import shutil
from dp.preprocess import preprocess
from dp.train import train
from dp.utils.io import read_config, save_config
import pandas as pd
import glob
from os import path

if __name__ == '__main__':

    allFiles = glob.glob("dp/notebooks/lexicons/*")
    allDF = (pd.read_csv(f, encoding='utf-8', sep='\t', names=['grapheme', 'phoneme']) for f in allFiles)

    df = pd.concat(allDF, ignore_index=True)
    df.insert(0, 'lang', 'pt_br')

    df['grapheme'] = df['grapheme'].map(str)
    df['phoneme'] = df['phoneme'].map(str)

    graphemes = ''.join(sorted(list(set(df['grapheme'].sum()))))

    phonemes = (
        df['phoneme'].str.split("\\")
            .explode().drop_duplicates()
            .sort_values().reset_index(drop=True)
            .values.tolist()
    )

    phonemes.append("~")
    
    df['phoneme'] = df['phoneme'].str.replace('\\', '')
    df['gsize'] = df['grapheme'].apply(lambda x : len(x))
    df['psize'] = df['phoneme'].apply(lambda x : len(x))
    df = df[df['gsize'] <= 225]
    
    dfsize = df.shape[0]

    subset = df[['lang', 'grapheme', 'phoneme']]
    train_data = [tuple(x) for x in subset.to_numpy()]

    for modelType in ['autoreg', 'forward']:
        config_file = f"./dp/configs/{modelType}_config.yaml"
        config = read_config(config_file)
        
        config['preprocessing']['languages'] = ['pt_br']
        config['preprocessing']['text_symbols'] = graphemes
        config['preprocessing']['phoneme_symbols'] = phonemes
        config['preprocessing']['lowercase'] = False

        epochs = 50
        for batch in [2**exp for exp in range(5, 2, -1)]:
            for d_model in [2**exp for exp in range(9, 2, -1)]:
                for d_fft in [2**exp for exp in range(10, 3, -1)]:

                    #batch=8
                    #d_model=512
                    #d_fft=1024
                    checkpoint_dir = f"checkpoints/{modelType}/batch-{batch}/model-{d_model}/fft-{d_fft}"
                    if (path.isdir(f"./{checkpoint_dir}")): continue

                    print(f'batch: {batch}')
                    print(f'model dim: {d_model}')
                    print(f'fft dim: {d_fft}')

                    n_val = dfsize//5

                    steps = (dfsize-n_val)//batch
                    total_steps = steps*epochs

                    config['preprocessing']['n_val'] = n_val

                    config['model']['d_model'] = d_model
                    config['model']['d_fft'] = d_fft

                    config['training']['epochs'] = epochs
                    config['training']['batch_size'] = batch
                    config['training']['batch_size_val'] = batch

                    config['training']['warmup_steps'] = total_steps//10
                    config['training']['generate_steps'] = total_steps//20
                    config['training']['validate_steps'] = total_steps//20
                    config['training']['checkpoint_steps'] = total_steps//2

                    config['paths']['checkpoint_dir'] = checkpoint_dir

                    save_config(config, 'config.yaml')

                    for k, v in config.items():
                        print(f'{k} {v}')

                    preprocess(config_file='config.yaml',
                            train_data=train_data,
                            deduplicate_train_data=False)

                    try:
                        train(config_file='config.yaml')
                    except Exception:
                        shutil.rmtree(checkpoint_dir, ignore_errors=True)