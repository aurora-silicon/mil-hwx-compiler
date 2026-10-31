#!/usr/bin/env python3
"""Compile, inspect and execute spatial dispatches in software. No device IO."""
import copy
import csv
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'tools'))
import h13_reference as ref
from spatial_package import Package, HEADER_BYTES

COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'build/mil-hwxc').resolve()
ACCEPTANCE_ONLY = '--acceptance-only' in sys.argv
NUMERICS = []


def typ(shape, dtype='fp16'):
    return f'tensor<{dtype}, [{",".join(map(str,shape))}]>'


def mil(parameters,body,outputs='y'):
    return ('program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n'
            f'func main<ios18>({parameters}) {{\n{body}\n}} -> ({outputs});\n}}\n')


class Model:
    def __init__(self,root):
        self.root = root
        self.lines = []
        self.payload = bytearray(128)

    def weight(self,name,shape,seed=1,values=None):
        n = math.prod(shape)
        if values is None:
            rng = np.random.default_rng(seed)
            values = rng.normal(0,.05,n).astype('<f2')
        data = np.asarray(values,dtype='<f2').reshape(-1).tobytes()
        assert len(data) == n*2
        self.payload.extend(bytes(-len(self.payload)%64))
        offset = len(self.payload)
        self.payload.extend(struct.pack('<IIQQ',0xDEADBEEF,1,len(data),offset+64)+bytes(40)+data)
        t = typ(shape)
        self.lines.append(f'{t} {name} = const()[name=string("{name}"),val={t}'
                          f'(BLOBFILE(path=string("@model_path/weights.bin"),offset=uint64({offset})))];')
        (self.root/'weights.bin').write_bytes(self.payload)
        return name

    def conv(self,x,xs,y,ys,k,stride,groups,bias=True,padding='same'):
        ci,co = xs[1],ys[1]
        self.weight(y+'_w',[co,ci//groups,k,k],seed=len(self.lines)+1)
        if bias:
            self.weight(y+'_b',[co],seed=len(self.lines)+1)
        args = f'x={x},weight={y}_w,' + (f'bias={y}_b,' if bias else '')
        # Constants use the exact MIL spelling in the benchmark probes.
        for name,shape,dtype,value in (
            ('s',[2],'int32',[stride,stride]),('d',[2],'int32',[1,1]),
            ('p',[4],'int32',[0,0,0,0])):
            t = typ(shape,dtype)
            self.lines.append(f'{t} {y}_{name}=const()[name=string("{y}_{name}"),val={t}({value})];')
        self.lines += [f'int32 {y}_g=const()[name=string("{y}_g"),val=int32({groups})];',
                       f'string {y}_pt=const()[name=string("{y}_pt"),val=string("{padding}")];']
        args += f'strides={y}_s,dilations={y}_d,pad={y}_p,pad_type={y}_pt,groups={y}_g'
        self.lines.append(f'{typ(ys)} {y}=conv({args})[name=string("{y}")];')

    def source(self,inputs,output='y'):
        return mil(', '.join(f'{typ(s)} {n}' for n,s in inputs.items()),'\n'.join(self.lines),output)


def compile_package(root,source,target='H13',label='case',expected=None):
    path,out = root/(label+'.mil'),root/label
    path.write_text(source)
    result = subprocess.run([str(COMPILER),'--target',target,'--schedule','spatial',
        '--mil',str(path),'--model-root',str(root),'--output',str(out)],
        capture_output=True,text=True,timeout=180)
    if expected:
        assert result.returncode == 65 and expected in result.stderr, result.stdout+result.stderr
        assert not out.exists(), 'failed compilation wrote package'
        return None
    assert result.returncode == 0, result.stdout+result.stderr
    return Package(out)


def reference(source,root,inputs):
    return ref.evaluate(source,root,inputs)


class SpatialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='spatial-compiler-')
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def compare(self,source,inputs,atol=0,rtol=0,label='case'):
        expected = reference(source,self.root,inputs)
        for target in ('H13','H14'):
            p = compile_package(self.root,source,target,label+target)
            result = p.simulate(inputs)
            self.assertEqual(set(result),set(expected))
            for name,raw in result.items():
                actual,want = np.frombuffer(raw,dtype='<f2'),np.frombuffer(expected[name],dtype='<f2')
                NUMERICS.append({'case':label,'target':target,'output':name,
                    'atol':atol,'rtol':rtol,'maxAbsoluteError':float(np.max(np.abs(
                        actual.astype(np.float32)-want.astype(np.float32))))})
                np.testing.assert_allclose(actual,want,atol=atol,rtol=rtol,
                    err_msg=f'{target} {label} {name}')
        return p

    def test_halo_regular_depthwise_stride_and_batch(self):
        for case,(batch,ci,co,h,w,k,stride,groups,padding) in enumerate((
            (1,3,2,17,19,3,1,1,'same'),
            (1,3,2,65,67,3,1,1,'same'),
            (1,3,2,65,67,3,2,1,'same'),
            (8,3,4,9,11,3,2,1,'same'),
            (1,3,3,31,33,3,1,3,'same'),
            (1,3,3,30,32,3,2,3,'same'),
            (1,65,3,5,7,1,1,1,'valid'),
            (1,3,2,17,19,3,2,1,'valid'),
        )):
            with self.subTest(case=case):
                xs = [batch,ci,h,w]
                ys = [batch,co,(h+stride-1)//stride if padding=='same' else (h-k)//stride+1,
                      (w+stride-1)//stride if padding=='same' else (w-k)//stride+1]
                model = Model(self.root)
                model.conv('x',xs,'y',ys,k,stride,groups)
                if padding == 'valid':
                    model.lines[-2] = model.lines[-2].replace('"same"','"valid"')
                source = model.source({'x':xs})
                data = ((np.arange(math.prod(xs)) * 7 % 23 - 11) * .125).astype('<f2').tobytes()
                self.compare(source,{'x':data},atol=.004,rtol=.008,label=f'conv{case}')

    def test_concat_axes_and_fanout(self):
        source = mil('tensor<fp16,[2,1,2]> a,tensor<fp16,[2,2,2]> b',
            'tensor<fp16,[2,3,2]> y=concat(values=(a,b),axis=int32(-2));\n'
            'tensor<fp16,[2,1,2]> z=relu(x=a);','y,z')
        self.compare(source,{'a':np.array([-1,2,3,-4],dtype='<f2').tobytes(),
                             'b':np.arange(8,dtype='<f2').tobytes()})

    def test_full_frame_emission(self):
        for h,w in ((1080,1920),(720,1280)):
            model = Model(self.root)
            model.conv('x',[1,3,h,w],'y',[1,4,h,w],3,1,1)
            for target in ('H13','H14'):
                p = compile_package(self.root,model.source({'x':[1,3,h,w]}),target,f'frame{h}{target}')
                self.assertGreater(p.dispatch_count,1)
                self.assertLess(len(p.payloads),10)
                self.assertEqual(p.manifest['tensors']['y']['elements'],4*h*w)

    def test_resize_modes_and_partial_windows(self):
        for mode in ('DEFAULT','STRICT_ALIGN_CORNERS','UNALIGN_CORNERS','nearest'):
            for shape,out in (((1,1,2,3),(1,1,5,7)),((2,2,1,130),(2,2,1,390))):
                if mode == 'nearest':
                    out = (*shape[:-2],shape[-2]*2,shape[-1]*3)
                name = 'resize_nearest_neighbor' if mode == 'nearest' else 'resize_bilinear'
                mode_arg = '' if mode == 'nearest' else f',sampling_mode=string("{mode}")'
                source = mil(f'{typ(shape)} x',f'{typ(out)} y={name}(x=x,'
                    f'target_size_height=int32({out[-2]}),target_size_width=int32({out[-1]}){mode_arg});')
                data = ((np.arange(math.prod(shape))%23-11)*.125).astype('<f2').tobytes()
                self.compare(source,{'x':data},atol=.002,rtol=.004,label=f'{mode}{shape[-1]}')

    def test_upsample_bilinear_half_pixel_and_nearest(self):
        for name,args,out in (
            ('upsample_bilinear','scale_factor_height=int32(2),scale_factor_width=int32(2),align_corners=bool(false)',[1,1,4,4]),
            ('upsample_nearest_neighbor','scale_factor_height=int32(2),scale_factor_width=int32(3)',[1,1,4,6]),
        ):
            source = mil('tensor<fp16,[1,1,2,2]> x',f'{typ(out)} y={name}(x=x,{args});')
            self.compare(source,{'x':np.array([0,1,2,3],dtype='<f2').tobytes()},label=name)

    def test_constant_runtime_surface(self):
        source = mil('',
            'tensor<fp16,[1,1,2,2]> x=const()[name=string("x"),val=tensor<fp16,[1,1,2,2]>([1,2,3,4])];\n'
            'tensor<fp16,[1,1,4,4]> y=resize_nearest_neighbor(x=x,target_size_height=int32(4),target_size_width=int32(4));')
        self.compare(source,{},label='constant')

    def test_connected_conv_resize_concat_conv(self):
        model = Model(self.root)
        model.conv('x',[1,3,7,9],'feature',[1,2,7,9],3,1,1)
        model.lines += [
            'tensor<fp16,[1,2,14,18]> up=upsample_bilinear(x=feature,scale_factor_height=int32(2),scale_factor_width=int32(2),align_corners=bool(false));',
            'tensor<fp16,[1,3,14,18]> joined=concat(values=(up,skip),axis=int32(1));',
        ]
        model.conv('joined',[1,3,14,18],'y',[1,2,14,18],3,1,1)
        inputs = {'x':((np.arange(189)%19-9)*.125).astype('<f2').tobytes(),
                  'skip':((np.arange(252)%13-6)*.0625).astype('<f2').tobytes()}
        self.compare(model.source({'x':[1,3,7,9],'skip':[1,1,14,18]}),inputs,
                     atol=.004,rtol=.01,label='connected')

    def test_mean_softmax_and_biased_linear_batch8(self):
        source = mil('tensor<fp16,[8,1024,7,7]> x',
            'tensor<fp16,[8,1024]> y=reduce_mean(x=x,axes=tensor<int32,[2]>([2,3]),keep_dims=bool(false));')
        values = ((np.arange(8*1024*49)%19-9)*.0625).astype('<f2').tobytes()
        self.compare(source,{'x':values},atol=.0002,rtol=.004,label='mean')
        source = mil('tensor<fp16,[8,1000]> x',
            'tensor<fp16,[8,1000]> y=softmax(x=x,axis=int32(-1));')
        values = np.linspace(-4,4,8000).astype('<f2').tobytes()
        self.compare(source,{'x':values},atol=.000004,rtol=.004,label='softmax')
        model = Model(self.root)
        model.weight('w',[1000,1024]); model.weight('b',[1000])
        model.lines.append('tensor<fp16,[8,1000]> y=linear(x=x,weight=w,bias=b);')
        values = ((np.arange(8192)%23-11)*.125).astype('<f2').tobytes()
        self.compare(model.source({'x':[8,1024]}),{'x':values},atol=.004,rtol=.01,label='linear')

    def test_clip_and_missing_initialized_region(self):
        source = mil('tensor<fp16,[1,3,13,15]> x',
            'tensor<fp16,[1,3,13,15]> y=clip(x=x,alpha=fp32(0),beta=fp32(6));')
        inputs = {'x':np.linspace(-8,8,585).astype('<f2').tobytes()}
        package = self.compare(source,inputs)
        calls = []
        def failed_dispatch(*args):
            calls.append(args[0]['file'])
            raise RuntimeError('dispatch failed')
        with self.assertRaisesRegex(RuntimeError,'dispatch failed'):
            package.execute(inputs,failed_dispatch)
        self.assertEqual(len(calls),1)
        package.manifest['dispatchPlan'].pop()
        with self.assertRaisesRegex(ValueError,'incomplete-output'):
            package.simulate(inputs)

    def test_named_refusals_and_malformed_contracts(self):
        bad = mil('tensor<fp16,[1,1,2,2]> x',
            'tensor<fp16,[1,1,3,3]> y=resize_nearest_neighbor(x=x,target_size_height=int32(3),target_size_width=int32(3));')
        compile_package(self.root,bad,label='bad',expected='spatial.nearest-fractional-scale')
        source = mil('tensor<fp16,[1,4,1,1]> x','tensor<fp16,[1,4,1,1]> y=relu(x=x);')
        p = compile_package(self.root,source)
        manifest = copy.deepcopy(p.manifest)
        manifest['dispatchPlan'][0]['body'][0]['inputs'][0]['count'] = 257
        (p.root/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'binding-size'):
            Package(p.root)
        (p.root/'manifest.json').write_text(json.dumps(p.manifest))
        file = p.root/p.manifest['kernels'][0]['file']
        original = file.read_bytes()
        mutation = bytearray(original); mutation[HEADER_BYTES-1] = 1; file.write_bytes(mutation)
        with self.assertRaisesRegex(ValueError,'reserved-bytes'):
            Package(p.root)
        mutation = bytearray(original)
        struct.pack_into('<I',mutation,52,1)
        file.write_bytes(mutation)
        with self.assertRaisesRegex(ValueError,'scratch-allocation'):
            Package(p.root)
        file.write_bytes(original)
        # A valid static slice can still read an unwritten scratch range;
        # execute checks actual production before invoking the callback.
        scratch = next(n for n,t in p.manifest['tensors'].items() if t['role']=='constant')
        p.manifest['tensors'][scratch]['role'] = 'scratch'
        with self.assertRaisesRegex(ValueError,'read-before-write'):
            p.simulate({'x':np.zeros(4,dtype='<f2').tobytes()})


def acceptance():
    """Every CSV conv/head point plus all four complete model/target pairs."""
    records = list(csv.DictReader((ROOT/'tests/data/mobilenet-compiler-gaps.csv').open()))
    results = []
    with tempfile.TemporaryDirectory(prefix='spatial-mobilenet-') as directory:
        root = Path(directory)
        for batch in (1,8):
            shapes = {}
            model = Model(root)
            x, xs = 'x',[batch,3,224,224]
            config = [(64,1),(128,2),(128,1),(256,2),(256,1),(512,2)]+[(512,1)]*5+[(1024,2),(1024,1)]
            layers = [('stem',32,2,1,3)]
            ci = 32
            for i,(co,stride) in enumerate(config,1):
                layers += [(f'dw{i}',ci,stride,ci,3),(f'pw{i}',co,1,1,1)]
                ci = co
            for name,co,stride,groups,k in layers:
                ys = [batch,co,(xs[2]+stride-1)//stride,(xs[3]+stride-1)//stride]
                start = len(model.lines)
                model.conv(x,xs,name,ys,k,stride,groups,padding='same' if k==3 else 'valid')
                shapes[name] = (x,xs,ys,model.lines[start:].copy())
                clip = name+'_relu6'
                model.lines.append(f'{typ(ys)} {clip}=clip(x={name},alpha=fp32(0),beta=fp32(6));')
                x,xs = clip,ys
            start = len(model.lines)
            model.lines.append(f'{typ([batch,1024])} mean=reduce_mean(x={x},axes=tensor<int32,[2]>([2,3]),keep_dims=bool(false));')
            shapes['mean'] = (x,xs,[batch,1024],model.lines[start:].copy())
            start = len(model.lines)
            model.weight('head_w',[1000,1024]); model.weight('head_b',[1000])
            model.lines.append(f'{typ([batch,1000])} logits=linear(x=mean,weight=head_w,bias=head_b);')
            shapes['linear'] = ('mean',[batch,1024],[batch,1000],model.lines[start:].copy())
            line = f'{typ([batch,1000])} probabilities=softmax(x=logits,axis=int32(-1));'
            model.lines.append(line)
            shapes['softmax'] = ('logits',[batch,1000],[batch,1000],[line])
            for target in ('H13','H14'):
                for row in (r for r in records if int(r['batch']) == batch and r['target'] == target):
                    name = row['operation']
                    input_name,in_shape,out_shape,body = shapes[name]
                    assert list(json.loads(row['input_shapes']).values()) == [in_shape]
                    assert json.loads(row['output_shape']) == out_shape
                    output = 'logits' if name=='linear' else 'probabilities' if name=='softmax' else name
                    source = mil(f'{typ(in_shape)} {input_name}','\n'.join(body),output)
                    label = f'b{batch}{target}{name}'
                    package = compile_package(root,source,target,label)
                    results.append({'batch':batch,'target':target,'operation':name,'kernels':len(package.payloads),
                                    'dispatches':package.dispatch_count,'status':'pass'})
                    import shutil
                    shutil.rmtree(package.root)
                package = compile_package(root,model.source({'x':[batch,3,224,224]},'probabilities'),
                                          target,f'full{batch}{target}')
                print(f'full MobileNet batch={batch} target={target} kernels={len(package.payloads)} dispatches={package.dispatch_count}',flush=True)
                results.append({'batch':batch,'target':target,'operation':'full-model','kernels':len(package.payloads),
                                'dispatches':package.dispatch_count,'status':'pass'})
                import shutil
                shutil.rmtree(package.root)
    assert len(results) == len(records)+4
    output = ROOT/'build/spatial-acceptance.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results,indent=2)+'\n')
    print(f'spatial MobileNet acceptance: PASS ({len(records)} CSV points + 4 full models)',flush=True)


if __name__ == '__main__':
    if ACCEPTANCE_ONLY:
        acceptance()
    else:
        result = unittest.main(argv=[sys.argv[0]],exit=False).result
        output = ROOT/'build/spatial-numerics.json'
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(NUMERICS,indent=2)+'\n')
        print(f'spatial CPU comparisons: {len(NUMERICS)}; '
              f'max absolute error {max((r["maxAbsoluteError"] for r in NUMERICS),default=0):.8g}',flush=True)
        sys.exit(not result.wasSuccessful())
