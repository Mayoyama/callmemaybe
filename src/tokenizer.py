def bytes_to_unicode() -> dict[int, str]:
    """Build the GPT-2 byte-to-unicode mapping table.

    Maps all 256 byte values to unique printable Unicode characters,
    following the same convention used by GPT-2-derived tokenisers.
    Printable ASCII and Latin-1 characters map to themselves; the
    remaining bytes map to characters starting at U+0100.

    Returns:
        A dict mapping each byte value (0-255) to a single Unicode
        character.
    """
    unicode_dict = {}
    print_ascii = [chr(asc) for asc in range(33, 127)]
    latin_supps1 = [chr(asc) for asc in range(161, 173)]
    latin_supps2 = [chr(asc) for asc in range(174, 256)]
    standard_chars = [print_ascii, latin_supps1, latin_supps2]
    non_print = [chr(asc) for asc in range(0, 33)]
    extended = [chr(asc) for asc in range(127, 161)]
    extended.append(chr(173))

    for charset in standard_chars:
        for char in charset:
            unicode_dict[ord(char)] = char

    i = 256
    for char in non_print:
        unicode_dict[ord(char)] = chr(i)
        i += 1

    for char in extended:
        unicode_dict[ord(char)] = chr(i)
        i += 1

    return unicode_dict


def raw_to_BPE(string: str) -> str:
    """Convert a raw string to its BPE-space representation.

    Encodes the input to UTF-8 bytes, then maps each byte through
    the bytes_to_unicode table, producing a string of BPE characters
    suitable for vocabulary lookup.

    Args:
        string: The input text to convert.

    Returns:
        The BPE-space representation of the input string.
    """
    unicode_dict = bytes_to_unicode()
    utf_string = string.encode('utf-8')
    BPE_string = ""

    for byte in utf_string:
        BPE_string += unicode_dict[byte]
    return BPE_string


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    """Encode a text string into a list of token IDs.

    Converts the text to BPE space via raw_to_BPE, then performs a
    greedy longest-match scan over the BPE string, looking up each
    substring in the vocabulary dictionary.

    Args:
        text: The input text to tokenise.
        vocab: A dict mapping BPE token strings to integer IDs.

    Returns:
        A list of integer token IDs for the input text.

    Raises:
        ValueError: If a BPE character has no match in the vocabulary.
    """
    bpe_text = raw_to_BPE(text)
    tokens: list[int] = []
    i = 0
    while i < len(bpe_text):
        for j in range(len(bpe_text), i, -1):
            needle = bpe_text[i:j]
            if needle in vocab:
                tokens.append(vocab[needle])
                i += len(needle)
                break
        else:
            raise ValueError(f"Encoding failed: No token found for "
                             f"character {bpe_text[i]!r}")
    return tokens
