from typing import Dict


def get_alphabet() -> str:
    """
    :return: alphabet with last blank symbol
    """
    return ' абвгдеёжзийклмнопрстуфхцчшщъыьэюя_'


def get_letter_to_index_map() -> Dict[str, int]:
    alphabet = get_alphabet()
    return {letter: index for index, letter in enumerate(list(alphabet))}


def get_index_to_letter_map() -> Dict[int, str]:
    alphabet = get_alphabet()
    return {index: letter for index, letter in enumerate(list(alphabet))}
