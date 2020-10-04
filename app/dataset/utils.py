from typing import Dict

from .word_to_number_dicts import COMPONENT_NUMBERS, THOUSANDS


def get_alphabet() -> str:
    """
    :return: alphabet with last blank symbol
    """
    return ' абвгдеёжзийклмнопрстуфхцчшщъыьэюя_'


def get_blank_id() -> int:
    return get_alphabet().index('_')


def get_letter_to_index_map() -> Dict[str, int]:
    alphabet = get_alphabet()
    return {letter: index for index, letter in enumerate(list(alphabet))}


def get_index_to_letter_map() -> Dict[int, str]:
    alphabet = get_alphabet()
    return {index: letter for index, letter in enumerate(list(alphabet))}


def text_to_number(text: str) -> int:
    words = text.split(' ')
    number = 0
    for i in words:
        if i in COMPONENT_NUMBERS:
            number += COMPONENT_NUMBERS[i]
            continue

        if i in THOUSANDS:
            if number == 0:
                number = 1000
            else:
                number *= 1000

    return number
