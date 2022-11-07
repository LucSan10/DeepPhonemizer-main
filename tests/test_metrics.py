import unittest

from dp.training.metrics import *


class TestWordError(unittest.TestCase):

    def test_call(self):
        predicted = ['a', 'b', 'c', 'd']
        target = ['a', 'k', 'c', 'a']
        result = transcription_error(predicted, target)
        self.assertEqual(1, result)

        predicted = ['r', 'r', 'r', 'r']
        target = ['a', 'k', 'c', 'a']
        result = transcription_error(predicted, target)
        self.assertEqual(1., result)

        predicted = ['a']
        target = ['a']
        result = transcription_error(predicted, target)
        self.assertEqual(0, result)


class TestPhonemeErrorRate(unittest.TestCase):

    def test_call(self):
        predicted = ['a', 'b', 'c', 'd']
        target = ['a', 'k', 'c', 'a']
        e, c = transcription_edit_distance(predicted, target)
        self.assertEqual(2, e)
        self.assertEqual(4, c)

        predicted = ['r', 'r', 'r', 'r']
        target = ['a', 'k', 'c', 'a']
        e, c = transcription_edit_distance(predicted, target)
        self.assertEqual(4, e)
        self.assertEqual(4, c)

        predicted = ['a']
        target = ['a']
        e, c = transcription_edit_distance(predicted, target)
        self.assertEqual(0, e)
        self.assertEqual(1, c)

        predicted = ['a']
        target = ['b']
        e, c = transcription_edit_distance(predicted, target)
        self.assertEqual(1, e)
        self.assertEqual(1, c)

        predicted = ['a', 'b']
        target = ['b']
        e, c = transcription_edit_distance(predicted, target)
        self.assertEqual(1, e)
        self.assertEqual(1, c)

        predicted = ['a', 'b', 'c']
        target = ['b']
        e, c = transcription_edit_distance(predicted, target)
        self.assertEqual(2, e)
        self.assertEqual(1, c)

        predicted = ['a']
        target = ['a', 'b']
        e, c = transcription_edit_distance(predicted, target)
        self.assertEqual(1, e)
        self.assertEqual(2, c)
