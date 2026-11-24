"""Check gfx1013 mip-zero gathers through the startup pixel translator."""

import argparse
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from pixel_gather_check import SIDE, texel, write_case
from startup_sampler_heap_check import VS_WORDS, assemble, root_blob

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402


SOURCE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, 0x12
s_mov_b32 s25, 0
s_mov_b32 s26, 0x04500000
s_mov_b32 s27, 0
v_mov_b32 v0, 0.5
v_mov_b32 v1, 0x3e800000
s_waitcnt lgkmcnt(0)
image_gather4_lz v[4:7], v[0:1], s[16:23], s[24:27] dmask:{mask:#x} dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
PREFIX = bytes.fromhex('04040cf4000000fa920398be800399beff039abe0000500480039bbe'
                       'f002007eff02027e0000803e7fc08cbf')
SUFFIX = bytes.fromhex('703f8cbf0f1800f804050607000081bf')
CHANNELS = ('Red', 'Green', 'Blue', 'Alpha')


def shader(mask=1, dim=1, opcode=0x47, flags=0, high=0, prefix=PREFIX, nsa=False):
    first = 0xf0000000 | (opcode << 18) | (mask << 8) | (dim << 3) | flags
    if nsa:
        first |= 2
    extension = bytes((1, 0, 0, 0)) if nsa else b''
    return prefix + struct.pack('<II', first, 0x00c40400 | high) + extension + SUFFIX


def pipeline(code):
    return G.Pipeline(VS_WORDS, code, root_blob(flags=0))


def refused(code, reason):
    try:
        pipeline(code)
    except G.Unsupported as error:
        assert reason in str(error), error
    else:
        raise AssertionError('Gather accepted an unsupported form: ' + reason)


def controls(llvm_mc):
    for component, channel in enumerate(CHANNELS):
        mask = 1 << component
        code = shader(mask)
        if llvm_mc:
            assert code == assemble(llvm_mc, SOURCE.format(mask=mask))
        pipe = pipeline(code)
        text = pipe.hlsl_ps()
        assert f't = r0.Gather{channel}(q0, float2(F(v0), F(v1)));' in text, text
        assert all(f'v{4 + i} = U(t.{lane});' in text for i, lane in enumerate('xyzw')), text
        assert 'srv 0 tex2d table 0 0\n' in pipe.manifest(), pipe.manifest()
        assert 'smp 0 0x12 0x0 0x4500000 0x0\n' in pipe.manifest(), pipe.manifest()
    refused(shader(0), 'one-hot dmask')
    refused(shader(3), 'one-hot dmask')
    refused(shader(dim=5), '2D texture')
    refused(shader(high=1 << 30), 'modifier a16')
    refused(shader(high=1 << 31), 'modifier d16')
    refused(shader(flags=1 << 16), 'modifier tfe')
    refused(shader(nsa=True), 'two contiguous address registers')
    refused(shader(opcode=0x40), 'image_gather4')
    refused(shader(opcode=0x44), 'image_gather4_l')
    refused(shader(opcode=0x4f), 'image_gather4_c_lz')
    bad_wrap = PREFIX.replace(bytes.fromhex('920398be'), bytes.fromhex('930398be'))
    bad_min_lod = PREFIX.replace(bytes.fromhex('800399be'), bytes.fromhex('810399be'))
    missing_sampler = PREFIX.replace(bytes.fromhex('80039bbe'), b'')
    refused(shader(prefix=bad_wrap), 'requires normalized clamp')
    refused(shader(prefix=bad_min_lod), 'requires normalized clamp')
    refused(shader(prefix=missing_sampler), 'constant sampler')
    print('Four fixed-mip gather channels and unsupported encodings checked')


def expected_taps(mip, component, uv):
    extent = SIDE >> mip
    x, y = (math.floor(coord * extent - 0.5) for coord in uv)
    return tuple(texel(mip, min(max(i, 0), extent - 1), min(max(j, 0), extent - 1), component)
                 for i, j in ((x, y + 1), (x + 1, y + 1), (x + 1, y), (x, y)))


def draw(dxc, renderer, adapter):
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        hlsl = directory / 'gather.hlsl'
        dxil = directory / 'gather.dxil'
        case = directory / 'gather.case'

        def run(mips, first, component, uv, code, *, mutant=False):
            source = pipeline(code).hlsl_ps()
            # The offscreen vertex shader places TEXCOORD0 before SV_Position.
            source = source.replace('struct PSIn {', 'struct PSIn {\n    float4 uv : TEXCOORD0;', 1)
            if mutant:
                source = source.replace('U(t.x)', 'U(t.w)', 1)
            hlsl.write_text(source, encoding='ascii')
            subprocess.run([dxc, '-WX', '-T', 'ps_6_0', '-E', 'main', '-Fo', dxil, hlsl], check=True)
            # The renderer uploads fixture level zero at the first resident mip.
            write_case(case, mips, first, [expected_taps(0, component, uv)] * SIDE**2)
            proc = subprocess.run([renderer, adapter, '--check-pixel-gather', dxil, case],
                                  capture_output=True, text=True)
            output = proc.stdout + proc.stderr
            print(f'{CHANNELS[component]}: mip {first}, {mips} resource mips, uv {uv}, mutant {mutant}\n{output}',
                  end='', flush=True)
            if proc.returncode == 77:
                raise SystemExit(77)
            if mutant:
                assert proc.returncode == 1 and 'pixel_mismatch x=0 y=0' in output, output
            else:
                assert proc.returncode == 0 and 'pixels=64 channels=256 max_absolute_error=0' in output, output

        uv = (0.5, 0.25)
        for channel in range(4):
            run(1, 0, channel, uv, shader(1 << channel))
            run(3, 1, channel, uv, shader(1 << channel))
        clamped = PREFIX.replace(bytes.fromhex('f002007e'), bytes.fromhex('f102007e'))
        clamped = clamped.replace(bytes.fromhex('0000803e'), bytes.fromhex('0000a03f'))
        run(3, 0, 1, (-0.5, 1.25), shader(2, prefix=clamped))
        run(3, 1, 2, uv, shader(4), mutant=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--llvm-mc')
    mode.add_argument('--draw', nargs=3, metavar=('DXC', 'RENDERER', 'ADAPTER'))
    options = parser.parse_args()
    if options.draw:
        draw(*options.draw)
    else:
        controls(options.llvm_mc)
