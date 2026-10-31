"""Write MIL + blob files for conv / conv+relu graphs (single layer or a chain)."""
import struct, os, numpy as np
def blob_write(path, arrays):
    """arrays: list of fp16 numpy arrays; returns list of metadata offsets."""
    out = bytearray(struct.pack('<II', 1, 2) + bytes(56))
    offs = []
    for a in arrays:
        data = np.ascontiguousarray(a, dtype=np.float16).tobytes()
        meta = len(out); data_off = meta + 64
        out += struct.pack('<IIQQ', 0xDEADBEEF, 1, len(data), data_off) + bytes(40)
        out += data
        out += bytes((-len(out)) % 64)
        offs.append(meta)
    open(path, 'wb').write(out)
    return offs
def conv_mil(layers, inshape, outdir, seed=0, relu_all=True, weights=None):
    """layers: list of dict(cout,k,s,g,bias,pad,relu). Returns (mil_path, weights list)."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(outdir, 'weights'), exist_ok=True)
    C, H, W = inshape
    consts = []; body = []; arrays = []; params = []
    cur = 'x'; c, h, w = C, H, W
    for i, L in enumerate(layers):
        co, k, s, g = L['cout'], L['k'], L.get('s', 1), L.get('g', 1)
        pad = L.get('pad', 'same')
        if pad == 'same': p = (k - 1) // 2; ho = (h + 2*p - k)//s + 1; wo = (w + 2*p - k)//s + 1
        else: p = 0; ho = (h - k)//s + 1; wo = (w - k)//s + 1
        cig = c // g
        if weights is not None: wt = weights[i]['w']; bs = weights[i].get('b')
        else:
            wt = (rng.standard_normal((co, cig, k, k)) / np.sqrt(cig*k*k)).astype(np.float16)
            bs = (rng.standard_normal(co) * 0.1).astype(np.float16) if L.get('bias') else None
        params.append(dict(w=wt, b=bs))
        wi = len(arrays); arrays.append(wt)
        bi = None
        if bs is not None: bi = len(arrays); arrays.append(bs)
        pt = 'custom' if pad == 'same' and k > 1 else 'valid'
        padv = [p, p, p, p] if pt == 'custom' else [0, 0, 0, 0]
        body.append(f'    string pt{i} = const()[name = string("pt{i}"), val = string("{pt}")];')
        body.append(f'    tensor<int32, [2]> st{i} = const()[name = string("st{i}"), val = tensor<int32, [2]>([{s}, {s}])];')
        body.append(f'    tensor<int32, [4]> pd{i} = const()[name = string("pd{i}"), val = tensor<int32, [4]>([{padv[0]}, {padv[1]}, {padv[2]}, {padv[3]}])];')
        body.append(f'    tensor<int32, [2]> dl{i} = const()[name = string("dl{i}"), val = tensor<int32, [2]>([1, 1])];')
        body.append(f'    int32 gp{i} = const()[name = string("gp{i}"), val = int32({g})];')
        body.append(f'    tensor<fp16, [{co}, {cig}, {k}, {k}]> W{i} = const()[name = string("W{i}"), val = tensor<fp16, [{co}, {cig}, {k}, {k}]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(@W{i}@)))];')
        biasarg = ''
        if bi is not None:
            body.append(f'    tensor<fp16, [{co}]> B{i} = const()[name = string("B{i}"), val = tensor<fp16, [{co}]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(@B{i}@)))];')
            biasarg = f'bias = B{i}, '
        out = f'c{i}'
        body.append(f'    tensor<fp16, [1, {co}, {ho}, {wo}]> {out} = conv({biasarg}dilations = dl{i}, groups = gp{i}, pad = pd{i}, pad_type = pt{i}, strides = st{i}, weight = W{i}, x = {cur})[name = string("conv{i}")];')
        cur = out
        if L.get('relu'):
            body.append(f'    tensor<fp16, [1, {co}, {ho}, {wo}]> r{i} = relu(x = {cur})[name = string("relu{i}")];')
            cur = f'r{i}'
        c, h, w = co, ho, wo
    offs = blob_write(os.path.join(outdir, 'weights', 'weight.bin'), arrays)
    text = '\n'.join(body)
    j = 0
    for i, P in enumerate(params):
        text = text.replace(f'@W{i}@', str(offs[j])); j += 1
        if P['b'] is not None: text = text.replace(f'@B{i}@', str(offs[j])); j += 1
    mil = f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {C}, {H}, {W}]> x) {{
{text}
  }} -> ({cur});
}}
'''
    path = os.path.join(outdir, 'model.mil'); open(path, 'w').write(mil)
    return path, params, (c, h, w)
