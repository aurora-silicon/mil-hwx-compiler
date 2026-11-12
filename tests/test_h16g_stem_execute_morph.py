#!/usr/bin/env python3
"""Unit checks for value-only and compiler-framed stem morph composition."""
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from h16g_stem_execute_morph import GROUPS, VALUE_GROUPS, build_case  # noqa: E402


def endpoint(marker, header_values, register_values):
    data = bytearray(256)
    data[240] = marker
    task = {
        "offset": 16,
        "header": [0, header_values[0], 0, 0, 0, 0, 0, header_values[1], 0],
        "registers": dict(register_values),
        "register_offsets": {},
    }
    cursor = 80
    for address, value in register_values.items():
        task["register_offsets"][address] = cursor
        struct.pack_into("<I", data, cursor, value)
        cursor += 4
    struct.pack_into("<I", data, 20, header_values[0])
    struct.pack_into("<I", data, 44, header_values[1])
    return bytes(data), {"text_off": 0}, [task]


class StemExecuteMorphTests(unittest.TestCase):
    def test_groups_cover_each_named_difference_once(self):
        self.assertEqual([group[0] for group in GROUPS], [
            "header-hints", "schedule", "result-span-record",
            "result-layout", "source-layout"])
        addresses = [address for group in VALUE_GROUPS for address in group[3]]
        self.assertEqual(len(addresses), len(set(addresses)))

    def test_value_patch_does_not_change_structural_endpoint(self):
        addresses = sorted({address for group in VALUE_GROUPS for address in group[3]})
        main = endpoint(0x4D, (12, 0x1FFFFFF),
                        {address: 0x10000000 + address for address in addresses})
        rb3 = endpoint(0x52, (15, 0x1FFFFFFF),
                       {address: 0x20000000 + address for address in addresses})
        values = {group[0]: "rb3" for group in VALUE_GROUPS}
        payload = build_case(main, rb3, "main", values)
        self.assertEqual(payload[240], 0x4D)
        self.assertEqual(struct.unpack_from("<I", payload, 20)[0], 15)
        for address in addresses:
            offset = main[2][0]["register_offsets"][address]
            self.assertEqual(struct.unpack_from("<I", payload, offset)[0],
                             0x20000000 + address)


if __name__ == "__main__":
    unittest.main()
