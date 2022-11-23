from typing import List, Tuple, Dict, Any

from dp.training.metrics import *

def collect_evaluation_results(eval_result, languages, lang_edits, lang_counts, lang_errors, type: str):
    edits, sizes, errors, counts = [], [], [], []
    for lang in languages:
        edit = sum(lang_edits[lang].values())
        edits.append(edit)

        size = sum(lang_counts[lang].values())
        sizes.append(size)
        
        error = sum(lang_errors[lang].values())
        errors.append(error)
        
        count = len(lang_errors[lang])
        counts.append(count)

        edits_rate = edit / size
        error_rate = error / count

        eval_result.setdefault(lang, {}).update({f'{type}_edits_rate': edits_rate})
        eval_result.setdefault(lang, {}).update({f'{type}_error_rate': error_rate})

    eval_result[f'mean_{type}_edits_rate'] = sum(edits) / sum(sizes)
    eval_result[f'mean_{type}_error_rate'] = sum(errors) / sum(counts)
        

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
    lang_stress_edd, lang_stress_count, lang_stress_err = dict(), dict(), dict()
    lang_pause_edd, lang_pause_count, lang_pause_err = dict(), dict(), dict()

    languages = sorted(lang_samples.keys())
    for lang in languages:
        for word, generated, target in lang_samples[lang]:
            word = ''.join(word)
            generated = ''.join(str(s) for s in generated)
            target = ''.join(str(s) for s in target)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
            trans_edd_dict = lang_trans_edd.setdefault(lang, dict())
            trans_count_dict = lang_trans_count.setdefault(lang, dict())
            trans_err_dict = lang_trans_err.setdefault(lang, dict())
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
            phon_edd_dict = lang_phon_edd.setdefault(lang, dict())
            phon_count_dict = lang_phon_count.setdefault(lang, dict())
            phon_err_dict = lang_phon_err.setdefault(lang, dict())
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
            syll_edd_dict = lang_syll_edd.setdefault(lang, dict())
            syll_count_dict = lang_syll_count.setdefault(lang, dict())
            syll_err_dict = lang_syll_err.setdefault(lang, dict())
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
            stress_edd_dict = lang_stress_edd.setdefault(lang, dict())
            stress_count_dict = lang_stress_count.setdefault(lang, dict())
            stress_err_dict = lang_stress_err.setdefault(lang, dict())
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
            pause_edd_dict = lang_pause_edd.setdefault(lang, dict())
            pause_count_dict = lang_pause_count.setdefault(lang, dict())
            pause_err_dict = lang_pause_err.setdefault(lang, dict())
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
            trans_edd, trans_count = transcription_edit_distance(generated, target)
            trans_err = transcription_error(generated, target)

            best_trans_err, best_trans_count = trans_edd_dict.get(word, None), trans_count_dict.get(word, None)
            if best_trans_err is None or trans_edd / trans_count < best_trans_err / best_trans_count:
                trans_edd_dict[word] = trans_edd
                trans_count_dict[word] = trans_count
                trans_err_dict[word] = trans_err
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
                phon_edd, phon_count = phoneme_edit_distance(generated, target)
                phon_err = phoneme_error(generated, target)

                phon_edd_dict[word] = phon_edd
                phon_count_dict[word] = phon_count
                phon_err_dict[word] = phon_err
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
                syll_edd, syll_count = syllable_edit_distance(generated, target)
                syll_err = syllable_error(generated, target)
                
                syll_edd_dict[word] = syll_edd
                syll_count_dict[word] = syll_count
                syll_err_dict[word] = syll_err
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
                stress_edd, stress_count = syllable_with_stress_edit_distance(generated, target)
                stress_err = syllable_with_stress_error(generated, target)

                stress_edd_dict[word] = stress_edd
                stress_count_dict[word] = stress_count
                stress_err_dict[word] = stress_err
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
                stress_edd, stress_count = syllable_with_stress_edit_distance(generated, target)
                stress_err = syllable_with_stress_error(generated, target)

                stress_edd_dict[word] = stress_edd
                stress_count_dict[word] = stress_count
                stress_err_dict[word] = stress_err
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
                pause_edd, pause_count = pause_edit_distance(generated, target)
                pause_err = pause_error(generated, target)

                pause_edd_dict[word] = pause_edd
                pause_count_dict[word] = pause_count
                pause_err_dict[word] = pause_err

    collect_evaluation_results(evaluation_result, languages, lang_trans_edd, lang_trans_count, lang_trans_err, 'trans')
    collect_evaluation_results(evaluation_result, languages, lang_phon_edd, lang_phon_count, lang_phon_err, 'phon')
    collect_evaluation_results(evaluation_result, languages, lang_syll_edd, lang_syll_count, lang_syll_err, 'syll')
    collect_evaluation_results(evaluation_result, languages, lang_stress_edd, lang_stress_count, lang_stress_err, 'stress')
    collect_evaluation_results(evaluation_result, languages, lang_pause_edd, lang_pause_count, lang_pause_err, 'pause')
    
    return evaluation_result
