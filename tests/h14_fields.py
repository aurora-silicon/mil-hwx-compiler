"""Read the public H14 field vocabulary for Python structural tests.

Tests use the compiler header as the single source of register addresses and
bit ranges.  They intentionally do not carry a second table of fixed words.
"""

from pathlib import Path
import re


class H14Fields:
    """Minimal parser for ``Field`` and named register declarations."""

    _FIELD = re.compile(
        r"inline constexpr Field (\w+)\{(\d+),\s*(\d+)\};")
    _REGISTER = re.compile(
        r"inline constexpr std::uint32_t (\w+) = (0x[0-9a-fA-F]+)u;")

    def __init__(self, root):
        header = (Path(root) / "plugins/H14/H14Fields.h").read_text()
        self.fields = {
            name: (int(lsb), int(width))
            for name, lsb, width in self._FIELD.findall(header)
        }
        register_block = header.split("namespace registers {", 1)[1].split(
            "} // namespace registers", 1)[0]
        self.registers = {
            name: int(value, 16)
            for name, value in self._REGISTER.findall(register_block)
        }

    def field(self, name, value=1):
        lsb, width = self.fields[name]
        return (int(value) & ((1 << width) - 1)) << lsb

    def get(self, name, encoded):
        lsb, width = self.fields[name]
        return (int(encoded) >> lsb) & ((1 << width) - 1)

    def register(self, name):
        return self.registers[name]
