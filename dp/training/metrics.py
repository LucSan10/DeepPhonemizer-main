import re
import numpy
from typing import List, Union, Tuple


def transcription_error(predicted: str, target: str) -> float:
    return equal_strings(predicted=predicted, target=target)

def transcription_edit_distance(predicted: str, target: str) -> Tuple[int, int]:
    return edit_distance(predicted=predicted, target=target)

def phoneme_error(predicted: str, target: str) -> Tuple[int, int]:
    pred_only_phoneme = remove_syllable_markers(predicted)
    target_only_phoneme = remove_syllable_markers(target)
    return equal_strings(predicted=pred_only_phoneme, target=target_only_phoneme)

def phoneme_edit_distance(predicted: str, target: str) -> Tuple[int, int]:
    pred_only_phoneme = remove_syllable_markers(predicted)
    target_only_phoneme = remove_syllable_markers(target)
    return edit_distance(predicted=pred_only_phoneme, target=target_only_phoneme)

def syllable_error(predicted: str, target: str) -> float:
    pred_syllables = remove_phonemes(standardize_syllable_markers(predicted))
    target_syllables = remove_phonemes(standardize_syllable_markers(target))
    return equal_strings(predicted=pred_syllables, target=target_syllables)

def syllable_edit_distance(predicted: str, target: str) -> Tuple[int, int]:
    pred_only_syllable = remove_phonemes(standardize_syllable_markers(predicted))
    target_only_syllable = remove_phonemes(standardize_syllable_markers(target))
    return edit_distance(predicted=pred_only_syllable, target=target_only_syllable)

def syllable_with_stress_error(predicted: str, target: str) -> float:
    pred_syllable_with_stress = remove_phonemes(add_syllable_marker(predicted))
    target_syllable_with_stress = remove_phonemes(add_syllable_marker(target))
    return equal_strings(predicted=pred_syllable_with_stress, target=target_syllable_with_stress)

def syllable_with_stress_edit_distance(predicted: str, target: str) -> Tuple[int, int]:
    pred_syllable_with_stress = remove_phonemes(add_syllable_marker(predicted))
    target_syllable_with_stress = remove_phonemes(add_syllable_marker(target))
    return edit_distance(predicted=pred_syllable_with_stress, target=target_syllable_with_stress)

def pause_error(predicted: str, target: str) -> float:
    pred_pause = keep_pauses(predicted)
    target_pause = keep_pauses(target)
    return equal_strings(predicted=pred_pause, target=target_pause)

def pause_edit_distance(predicted: str, target: str) -> Tuple[int, int]:
    pred_pause = keep_pauses(predicted)
    target_pause = keep_pauses(target)
    return edit_distance(predicted=pred_pause, target=target_pause)

def equal_strings(predicted: str, target: str) -> float:
    """Calculates the word error rate of a single word result.

    Args:
      predicted: Predicted word.
      target: Target word.
      predicted: List[Union[str: 
      int]]: 
      target: List[Union[str: 

    Returns:
      Word error

    """

    return int(predicted != target)

def edit_distance_transposition(predicted: str, target: str) -> Tuple[int, int]:
    """Calculates the phoneme error rate of a single result based on the Levenshtein distance.

    Args:
      predicted: Predicted word.
      target: Target word.
      predicted: List[Union[str: 
      int]]: 
      target: List[Union[str: 

    Returns:
      Phoneme error.

    """

    d = numpy.zeros((len(target) + 1, len(predicted) + 1), dtype=numpy.uint8)
    for i in range(len(target) + 1):
        d[i][0] = i
    for j in range(len(predicted) + 1):
        d[0][j] = j

    for i in range(1, len(target) + 1):
        for j in range(1, len(predicted) + 1):
            if target[i - 1] == predicted[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

            if i > 1 and j > 1 and target[i-1] == predicted[j-2] and target[i-2] == predicted[j-1]:
               d[i, j] = min(d[i, j], d[i-2, j-2] + 1)

    return d[len(target)][len(predicted)], len(target)

def edit_distance(predicted: str, target: str) -> Tuple[int, int]:
    """Calculates the phoneme error rate of a single result based on the Levenshtein distance.

    Args:
      predicted: Predicted word.
      target: Target word.
      predicted: List[Union[str: 
      int]]: 
      target: List[Union[str: 

    Returns:
      Phoneme error.

    """

    d = numpy.zeros((len(target) + 1, len(predicted) + 1), dtype=numpy.uint8)
    for i in range(len(target) + 1):
        d[i][0] = i
    for j in range(len(predicted) + 1):
        d[0][j] = j

    for i in range(1, len(target) + 1):
        for j in range(1, len(predicted) + 1):
            if target[i - 1] == predicted[j - 1]:
               d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(target)][len(predicted)], len(target)

def standardize_syllable_markers(string: str) -> Union[str,int]:
    return add_syllable_marker(stress_marker_to_syllable_marker(string))

def stress_marker_to_syllable_marker(string: str) -> Union[str,int]:
    return re.sub(r"'", r".", string)

def add_syllable_marker(string: str) -> Union[str,int]:
    return re.sub(r"(^|\s)([^\s\.,'])", r"\1.\2", string)

def remove_phonemes(string: str) -> Union[str,int]:
    return re.sub(r"[^\.'\s,]", '', string)

def remove_syllable_markers(string: str) -> Union[str,int]:
    return re.sub(r"[\.']", '', string)

def remove_pauses(string: str) -> Union[str,int]:
    return re.sub(r"[\s,]", '', string)

def keep_pauses(string: str) -> Union[str,int]:
    return remove_phonemes(remove_syllable_markers(string))