"""Read the MobileNetV1 weights out of a TFLite flatbuffer (float or uint8
quantized), without TensorFlow: a minimal flatbuffer table reader for the
fields this needs (tensors, buffers, quantization, operators).

The quantized mobilenet_v1_1.0_224_quant.tflite (Apache-2.0, shipped with
Mesa's teflon tests) dequantizes to float weights with batch norm folded.
"""
import struct
import numpy as np

class Table:
    def __init__(self, buf, pos):
        self.buf, self.pos = buf, pos
        vt = pos - struct.unpack_from('<i', buf, pos)[0]
        self.vt_len = struct.unpack_from('<H', buf, vt)[0]
        self.vt = vt

    def _field(self, index):
        o = 4 + 2 * index
        if o >= self.vt_len:
            return 0
        return struct.unpack_from('<H', self.buf, self.vt + o)[0]

    def scalar(self, index, fmt, default=0):
        o = self._field(index)
        return struct.unpack_from('<' + fmt, self.buf, self.pos + o)[0] if o else default

    def table(self, index):
        o = self._field(index)
        if not o:
            return None
        p = self.pos + o
        return Table(self.buf, p + struct.unpack_from('<I', self.buf, p)[0])

    def _vector(self, index):
        o = self._field(index)
        if not o:
            return None, 0
        p = self.pos + o
        p += struct.unpack_from('<I', self.buf, p)[0]
        return p + 4, struct.unpack_from('<I', self.buf, p)[0]

    def vector(self, index, fmt):
        p, n = self._vector(index)
        if p is None:
            return []
        return list(struct.unpack_from(f'<{n}{fmt}', self.buf, p))

    def bytes(self, index):
        p, n = self._vector(index)
        return b'' if p is None else self.buf[p:p + n]

    def tables(self, index):
        p, n = self._vector(index)
        if p is None:
            return []
        return [Table(self.buf, p + 4 * i + struct.unpack_from('<I', self.buf, p + 4 * i)[0]) for i in range(n)]

    def string(self, index):
        return self.bytes(index).decode()

BUILTIN = {0: 'ADD', 1: 'AVERAGE_POOL_2D', 3: 'CONV_2D', 4: 'DEPTHWISE_CONV_2D', 22: 'RESHAPE', 25: 'SOFTMAX'}

def read_model(path):
    buf = open(path, 'rb').read()
    model = Table(buf, struct.unpack_from('<I', buf, 0)[0])
    codes = []
    for oc in model.tables(1):
        code = max(oc.scalar(0, 'b'), oc.scalar(3, 'i'))
        codes.append(BUILTIN.get(code, str(code)))
    buffers = model.tables(4)
    sg = model.tables(2)[0]
    tensors = []
    for t in sg.tables(0):
        q = t.table(4)
        tensors.append(dict(shape=t.vector(0, 'i'), type=t.scalar(1, 'b'), buffer=t.scalar(2, 'I'),
                            name=t.string(3),
                            scale=q.vector(2, 'f') if q else [], zero=q.vector(3, 'q') if q else []))
    ops = [dict(op=codes[o.scalar(0, 'I')], inputs=o.vector(1, 'i'), outputs=o.vector(2, 'i')) for o in sg.tables(3)]
    return tensors, ops, buffers

def tensor_value(t, buffers):
    data = buffers[t['buffer']].bytes(0)
    dtype = {0: np.float32, 2: np.int32, 3: np.uint8, 9: np.int8}[t['type']]
    a = np.frombuffer(data, dtype=dtype).reshape(t['shape']).astype(np.float64)
    if t['type'] != 0 and t['scale']:
        a = (a - np.float64(t['zero'][0] if t['zero'] else 0)) * np.float64(t['scale'][0])
    return a

def mobilenet_v1_weights(path):
    """-> list of 28 dict(w, b) in the TF layer order of mobilenet.tf_layers():
    weights as [Cout, Cin, kh, kw] (depthwise [C, 1, kh, kw]) float32."""
    tensors, ops, buffers = read_model(path)
    out = []
    for op in ops:
        if op['op'] not in ('CONV_2D', 'DEPTHWISE_CONV_2D'):
            continue
        w = tensor_value(tensors[op['inputs'][1]], buffers)
        b = tensor_value(tensors[op['inputs'][2]], buffers)
        if op['op'] == 'CONV_2D':          # OHWI -> OIHW
            w = w.transpose(0, 3, 1, 2)
        else:                              # 1HWC -> C1HW
            w = w.transpose(3, 0, 1, 2)
        out.append(dict(w=w.astype(np.float32), b=b.astype(np.float32)))
    if len(out) != 28:
        raise ValueError(f'expected 28 conv layers, found {len(out)}')
    return out

def input_quantization(path):
    tensors, ops, _ = read_model(path)
    t = tensors[ops[0]['inputs'][0]]
    return t['scale'][0], t['zero'][0]
