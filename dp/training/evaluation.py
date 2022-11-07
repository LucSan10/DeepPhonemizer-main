from typing import List, Tuple, Dict, Any

from dp.training.metrics import *


def evaluate_samples(lang_samples: Dict[str, List[Tuple[List[str], List[str], List[str]]]]) -> Dict[str, Any]:
    """Calculates word and phoneme error rates per language and their mean across languages

    Args:
      lang_samples (Dict): Data to evaluate. Contains languages as keys and list of result samples as values.
                           Prediction samples is given as a List of Tuples, where each Tuple is a tokenized representation of
                           (text, result, target).

    Returns:
      Dict: Evaluation result carrying word and phoneme error rates per language.

    """

    evaluation_result = dict()
    lang_trans_edd, lang_trans_count, lang_trans_err = dict(), dict(), dict()
    lang_phon_edd, lang_phon_count, lang_phon_err = dict(), dict(), dict()
    lang_syll_edd, lang_syll_count, lang_syll_err = dict(), dict(), dict()
    languages = sorted(lang_samples.keys())
    for lang in languages:
        for word, generated, target in lang_samples[lang]:
            word = ''.join(word)

            trans_edd, trans_count = transcription_edit_distance(generated, target)
            trans_err = transcription_error(generated, target)
            
            phon_edd, phon_count = phoneme_edit_distance(generated, target)
            phon_err = phoneme_error(generated, target)
            
            syll_edd, syll_count = syllable_edit_distance(generated, target)
            syll_err = syllable_error(generated, target)

            trans_edd_dict = lang_trans_edd.setdefault(lang, dict())
            trans_count_dict = lang_trans_count.setdefault(lang, dict())
            trans_err_dict = lang_trans_err.setdefault(lang, dict())
            
            phon_edd_dict = lang_phon_edd.setdefault(lang, dict())
            phon_count_dict = lang_phon_count.setdefault(lang, dict())
            phon_err_dict = lang_phon_err.setdefault(lang, dict())
            
            syll_edd_dict = lang_syll_edd.setdefault(lang, dict())
            syll_count_dict = lang_syll_count.setdefault(lang, dict())
            syll_err_dict = lang_syll_err.setdefault(lang, dict())
            
            best_trans_err, best_trans_count = trans_edd_dict.get(word, None), trans_count_dict.get(word, None)
            if best_trans_err is None or trans_edd / trans_count < best_trans_err / best_trans_count:
                trans_edd_dict[word] = trans_edd
                trans_count_dict[word] = trans_count
                trans_err_dict[word] = trans_err
                
                phon_edd_dict[word] = phon_edd
                phon_count_dict[word] = phon_count
                phon_err_dict[word] = phon_err
                
                syll_edd_dict[word] = syll_edd
                syll_count_dict[word] = syll_count
                syll_err_dict[word] = syll_err

    trans_edits, trans_sizes, trans_errors, trans_counts = [], [], [], []
    phon_edits, phon_sizes, phon_errors, phon_counts = [], [], [], []
    syll_edits, syll_sizes, syll_errors, syll_counts = [], [], [], []
    for lang in languages:
        trans_edd = sum(lang_trans_err[lang].values())
        trans_edits.append(trans_edd)

        trans_size = sum(lang_trans_count[lang].values())
        trans_sizes.append(trans_size)
        
        trans_err = sum(lang_trans_err[lang].values())
        trans_errors.append(trans_err)
        
        trans_count = len(lang_trans_err[lang])
        trans_counts.append(trans_count)
        
        phon_edd = sum(lang_phon_err[lang].values())
        phon_edits.append(phon_edd)

        phon_size = sum(lang_phon_count[lang].values())
        phon_sizes.append(phon_size)
        
        phon_err = sum(lang_phon_err[lang].values())
        phon_errors.append(phon_err)
        
        phon_count = len(lang_phon_err[lang])
        phon_counts.append(phon_count)
        
        syll_edd = sum(lang_syll_err[lang].values())
        syll_edits.append(syll_edd)

        syll_size = sum(lang_syll_count[lang].values())
        syll_sizes.append(syll_size)
        
        syll_err = sum(lang_syll_err[lang].values())
        syll_errors.append(syll_err)
        
        syll_count = len(lang_syll_err[lang])
        syll_counts.append(syll_count)
        
        trans_edits_rate = trans_edd / trans_size
        trans_error_rate = trans_err / trans_count
        phon_edits_rate = phon_edd / phon_size
        phon_error_rate = phon_err / phon_count
        syll_edits_rate = syll_edd / syll_size
        syll_error_rate = syll_err / syll_count
        
        evaluation_result.setdefault(lang, {}).update({'trans_edits_rate': trans_edits_rate})
        evaluation_result.setdefault(lang, {}).update({'trans_error_rate': trans_error_rate})
        evaluation_result.setdefault(lang, {}).update({'phon_edits_rate': phon_edits_rate})
        evaluation_result.setdefault(lang, {}).update({'phon_error_rate': phon_error_rate})
        evaluation_result.setdefault(lang, {}).update({'syll_edits_rate': syll_edits_rate})
        evaluation_result.setdefault(lang, {}).update({'syll_error_rate': syll_error_rate})
    
    mean_trans_edits_rate = sum(trans_edits) / sum(trans_sizes)
    mean_trans_error_rate = sum(trans_errors) / sum(trans_counts)
    mean_phon_edits_rate = sum(phon_edits) / sum(phon_sizes)
    mean_phon_error_rate = sum(phon_errors) / sum(phon_counts)
    mean_syll_edits_rate = sum(syll_edits) / sum(syll_sizes)
    mean_syll_error_rate = sum(syll_errors) / sum(syll_counts)
    
    evaluation_result['mean_trans_edits_rate'] = mean_trans_edits_rate
    evaluation_result['mean_trans_error_rate'] = mean_trans_error_rate
    evaluation_result['mean_phon_edits_rate'] = mean_phon_edits_rate
    evaluation_result['mean_phon_error_rate'] = mean_phon_error_rate
    evaluation_result['mean_syll_edits_rate'] = mean_syll_edits_rate
    evaluation_result['mean_syll_error_rate'] = mean_syll_error_rate

    return evaluation_result
