"""Source-owned fork/join fixtures for native channel concat."""
import struct

from mint_conv_probes import finite_index_payload
from mint_oracles import program, tensor_type


def graph_case(shared=False, consumer=False, payloads=None, channels=16, spatial=16,two_inputs=False):
    c,h=channels,spatial
    shapes=[(c,c,1,1),(c,),(c,c,5,5),(c,)]
    if consumer: shapes += [(2*c,2*c,1,1),(2*c,)]
    if payloads is None:
        import math
        payloads=[finite_index_payload(math.prod(shape),3991+i) for i,shape in enumerate(shapes)]
    assert len(payloads)==len(shapes)
    blob=bytearray(struct.pack("<II",len(payloads),2)+bytes(56));offsets=[]
    for payload in payloads:
        meta=len(blob);offsets.append(meta)
        blob += struct.pack("<IIQQ",0xdeadbeef,1,len(payload),meta+64)+bytes(40)+payload
        blob += bytes((-len(blob))%64)
    body=[
        'string pt = const()[name = string("pt"), val = string("same")];',
        'tensor<int32, [2]> st = const()[name = string("st"), val = tensor<int32, [2]>([1,1])];',
        'tensor<int32, [4]> pd = const()[name = string("pd"), val = tensor<int32, [4]>([0,0,0,0])];',
        'tensor<int32, [2]> dl = const()[name = string("dl"), val = tensor<int32, [2]>([1,1])];',
        'int32 gp = const()[name = string("gp"), val = int32(1)];',
    ]
    for i,(shape,offset) in enumerate(zip(shapes,offsets)):
        body.append(f'{tensor_type(shape)} p{i} = const()[name = string("p{i}"), val = {tensor_type(shape)}(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({offset})))];')
    kind=tensor_type((1,c,h,h));wide=tensor_type((1,2*c,h,h))
    body += [
        f'{kind} a = conv(x = x, weight = p0, bias = p1, strides = st, dilations = dl, pad = pd, pad_type = pt, groups = gp)[name = string("a")];',
        f'{kind} b = conv(x = {"a" if shared else "u" if two_inputs else "x"}, weight = p2, bias = p3, strides = st, dilations = dl, pad = pd, pad_type = pt, groups = gp)[name = string("b")];',
        f'{wide} {"z" if consumer else "y"} = concat(values = (a,b), axis = int32(1), interleave = bool(false))[name = string("concat")];',
    ]
    if consumer:
        body.append(f'{wide} y = conv(x = z, weight = p4, bias = p5, strides = st, dilations = dl, pad = pd, pad_type = pt, groups = gp)[name = string("y")];')
    return program(f"{kind} x"+(f", {kind} u" if two_inputs else ""),body,"y"),bytes(blob)
