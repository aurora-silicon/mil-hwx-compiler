#!/usr/bin/env python3
"""Sanitized descriptor parity for the 24 transformer-linear rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from mint_oracles import decode_task, split_h14_tasks  # noqa: E402

DUMP = str(Path(sys.argv[1] if len(sys.argv) > 1 else
                ROOT / "build/h16_transformer_linear_dump").resolve())

# name, S, Cin, Cout, flags (bias=1, gelu=2), scale, task sizes,
# kernel bytes, sanitized descriptor SHA-256 (ExeCycles excluded).
CASES = [
    ("lin_bert_3072x768",128,3072,768,0,0x3c00,[180,396],4718592,"b9cadaa7586ea5657702a4f458f16d874f4bc19da10844d75d0515d508da188d"),
    ("lin_bert_768x3072",128,768,3072,0,0x3c00,[180,396],4718592,"8bb270f41c6b77dc8f7a4ab3e37d699fe709d8c730a0994bf9ce34b16f3deee0"),
    ("lin_bert_768x768",128,768,768,0,0x3c00,[180,396],1179648,"6b8624a6795e8872606f992a3b045e396ead3e81017c5db5f32f75bd0a99f02e"),
    ("lin_wtiny_1536x384",1500,1536,384,0,0x3c00,[208,420,448],1179648,"2bf971091d967e72c4c8064dc034774cd34472c85ff9d88b5443b51d9cd7f93b"),
    ("lin_wtiny_384x1536",1500,384,1536,0,0x3c00,[180,368,392],1179648,"c24ca95c0009867dc2905c00aa9e9ca1d5c000c7e9c887e195bbedb4c49828df"),
    ("lin_wtiny_384x384",1500,384,384,0,0x3c00,[180,396],294912,"c7544a7357f1730b77d51bc7a87c71a7b0f5835c6fa04f6e59f4b05ad53a666e"),
    ("lin_wbase_2048x512",1500,2048,512,0,0x3c00,[208,420,452],2097152,"224ac8d083dcd39e21f10c5c124d52fe5568c84dd622ee329d31e97620898dd7"),
    ("lin_wbase_512x2048",1500,512,2048,0,0x3c00,[180,368,392],2097152,"82e567ad9c04e4048739c2183f6b2408f9fab6624efbd0e5d2308152f82a2d76"),
    ("lin_wbase_512x512",1500,512,512,0,0x3c00,[180,396],524288,"e308f901740ecc723b643866a9884f1c9f4a0152a196d18e159124581447a833"),
    ("linb_bert_3072x768",128,3072,768,1,0x3c00,[180,396],4720640,"051139a4ccc1af40ad6289653cf41c4c2a112d653ddb4eb646188c68c2c41166"),
    ("linb_bert_768x3072",128,768,3072,1,0x3c00,[180,396],4724736,"201c8f7332b346cd9b1c4132a6626348c415832ec21ac848f7dafda927e37e02"),
    ("linb_bert_768x768",128,768,768,1,0x3c00,[180,396],1181696,"62fc64978c241bb6989f64fc0702debe41571c042a4ece3dd277e59b973cad25"),
    ("linb_wtiny_1536x384",1500,1536,384,1,0x3c00,[208,420,448],1181696,"be9a6c0f92e167397c605cd51a9fe8c19b0ee070bc33e17be7641e320fac7f8e"),
    ("linb_wtiny_384x1536",1500,384,1536,1,0x3c00,[180,368,392],1183744,"814a05b3632ff516bd6bd5401301fdb1d570c8022d1ce7368b3435e3d81ec077"),
    ("linb_wtiny_384x384",1500,384,384,1,0x3c00,[180,396],295936,"780d512c659751a230917f4f568074c11fe0aebed72098c4d59670f32915d814"),
    ("linb_wbase_2048x512",1500,2048,512,1,0x3c00,[208,420,424,452],2100224,"3995534035b9191b7af77330c24c4a3c807bd437a9558e9f80566050c1cb69f9"),
    ("linb_wbase_512x2048",1500,512,2048,1,0x3c00,[180,368,364,392],2102272,"dd8497f9cd9cc0ff431de8c53be038ac99026c1deb2dfce6a73d02bc43942797"),
    ("linb_wbase_512x512",1500,512,512,1,0x3c00,[180,396],525312,"dcb7cd02ff3fbc06d4d298b4003c15b0432398bb674ce78ba392be92a733e46d"),
    ("lingelu_bert",128,768,3072,3,0x3c00,[180,396],4726784,"880c2c89947504f1389b8a0b92ff9fc79773748f8c6ec38ad25cc211dce37fa6"),
    ("lingelu_wtiny",1500,384,1536,3,0x3c00,[180,368,392],1187840,"fbeb425f4512ec2ab120f11de70f62c2b49e65ff3e3bc22a17162567654e7f3e"),
    ("lingelu_wbase",1500,512,2048,3,0x3c00,[180,368,364,392],2108416,"fe19c7e5df5a224fefb7a32467b9a0813c3b428aa981cb86e549cd0c197993e3"),
    ("linscale_bert",128,768,768,1,0x3000,[180,396],1181696,"80df5a523f2fb25928a724963f7d3ed11f25316df20c5a671482fb908fc17d28"),
    ("linscale_wtiny",1500,384,384,1,0x3000,[180,396],295936,"ea61eeaa0c4688293e6691699bfa3ab78777d1ec250e10f6a401e747d3802559"),
    ("linscale_wbase",1500,512,512,1,0x3000,[180,396],525312,"9b4dcac6fb958f6ec04dab05398de44767413a9f607b9d55d121df870439ffca"),
]


def canonical(task: bytes) -> dict:
    decoded = decode_task(task, "h16")
    # The captured linear packet names BAR 7 at the kernel DMA record.  A
    # packaged object must rebase that one operand to BAR 8 because its
    # validated apertures are input/output/kernel at slots 6/7/8.  Normalize
    # only that packaging correction so the remaining 24-row descriptor
    # parity digest continues to cover every decoded field.
    for record in decoded["records"]:
        if record["address"] == "0x05510":
            header = int(record["header"], 16)
            if (header & 0x3f800000) == 0x24000000:
                record["header"] = f"0x{header - 0x00800000:08x}"
    return {
        "header": decoded["header_words"][:1] + decoded["header_words"][2:],
        "blocks": decoded["blocks"],
        "records": decoded["records"],
    }


def main() -> None:
    for name, tokens, cin, cout, flags, scale, sizes, kernel_bytes, digest in CASES:
        payload_bytes = 2 * cin * cout + (2 * cout if flags & 1 else 0) + \
            (128 if flags & 2 else 0)
        payload = bytes(payload_bytes)
        args = [DUMP, "task", str(tokens), str(cin), str(cout), str(flags), hex(scale)]
        stream = subprocess.run(args, input=payload, stdout=subprocess.PIPE,
                                timeout=120, check=True).stdout
        tasks = split_h14_tasks(stream[16:])
        assert [len(task) for task in tasks] == sizes, name
        encoded = json.dumps([canonical(task) for task in tasks], sort_keys=True,
                             separators=(",", ":")).encode()
        assert hashlib.sha256(encoded).hexdigest() == digest, name
        kernel = subprocess.run([DUMP, "kernel", *args[2:]], input=payload,
                                stdout=subprocess.PIPE, timeout=120,
                                check=True).stdout
        assert len(kernel) == kernel_bytes, name
    print(f"H16G transformer linear parity PASS: {len(CASES)} descriptor/kernel rows")


if __name__ == "__main__":
    main()
