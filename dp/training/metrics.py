import re
import numpy
from typing import List, Union, Tuple


def transcription_error(predicted: List[Union[str, int]], target: List[Union[str, int]]) -> float:
  return equal_strings(predicted=predicted, target=target)

def transcription_edit_distance(predicted: List[Union[str, int]], target: List[Union[str, int]]) -> Tuple[int, int]:
  return edit_distance(predicted=predicted, target=target)

def phoneme_error(predicted: List[Union[str, int]], target: List[Union[str, int]]) -> Tuple[int, int]:
  pred_only_phoneme = remove_syllable_markers(predicted)
  target_only_phoneme = remove_syllable_markers(target)
  return equal_strings(predicted=pred_only_phoneme, target=target_only_phoneme)

def phoneme_edit_distance(predicted: List[Union[str, int]], target: List[Union[str, int]]) -> Tuple[int, int]:
  pred_only_phoneme = remove_syllable_markers(predicted)
  target_only_phoneme = remove_syllable_markers(target)
  return edit_distance(predicted=pred_only_phoneme, target=target_only_phoneme)

def syllable_error(predicted: List[Union[str, int]], target: List[Union[str, int]]) -> float:
  pred_syllables = remove_phonemes(add_syllable_marker(predicted))
  target_syllables = remove_phonemes(add_syllable_marker(target))
  return equal_strings(predicted=pred_syllables, target=target_syllables)

def syllable_edit_distance(predicted: List[Union[str, int]], target: List[Union[str, int]]) -> Tuple[int, int]:
  pred_only_syllable = remove_phonemes(add_syllable_marker(predicted))
  target_only_syllable = remove_phonemes(add_syllable_marker(target))
  return edit_distance(predicted=pred_only_syllable, target=target_only_syllable)

def syllable_size_error(predicted: List[Union[str, int]], target: List[Union[str, int]]) -> float:
  pred_syllable_size = len(remove_phonemes(add_syllable_marker(predicted)))
  target_syllable_size = len(remove_phonemes(add_syllable_marker(target)))
  return equal_strings(predicted=pred_syllable_size, target=target_syllable_size)

def equal_strings(predicted: List[Union[str, int]], target: List[Union[str, int]]) -> float:
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

def edit_distance(predicted: List[Union[str, int]], target: List[Union[str, int]]) -> Tuple[int, int]:
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

    d = numpy.zeros((len(target) + 1) * (len(predicted) + 1),
                    dtype=numpy.uint8)
    d = d.reshape((len(target) + 1, len(predicted) + 1))
    for i in range(len(target) + 1):
        for j in range(len(predicted) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

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

def add_syllable_marker(string: List[Union[str, int]]) -> Union[str,int]:
  return re.sub(r"(^|\s)([^\s\.,'])", r"\1.\2", string)

def remove_phonemes(string: List[Union[str, int]]) -> Union[str,int]:
  return re.sub(r"[^\.'\s]", '', string)

def remove_syllable_markers(string: List[Union[str, int]]) -> Union[str,int]:
  return re.sub(r"[\.']", '', string)