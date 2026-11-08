#!/usr/bin/env python3
"""Device-free integrity checks for the H16G ResNet-50 oracle campaign."""
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "research" / "mint_h16g_resnet50_oracles.py"
SPEC = importlib.util.spec_from_file_location("resnet_oracles", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class ResNetOracleCampaignTests(unittest.TestCase):
    def test_exact_unique_convolution_geometries(self):
        geometries = MODULE.CONV_GEOMETRIES
        self.assertEqual(len(geometries), 23)
        self.assertEqual(len(set(geometries)), 23)
        self.assertIn((3, 64, 224, 7, 2), geometries)
        self.assertIn((512, 512, 14, 3, 2), geometries)
        self.assertIn((1024, 2048, 14, 1, 2), geometries)

    def test_campaign_covers_activation_states_and_non_conv_primitives(self):
        cases = MODULE.campaign()
        names = [item["name"] for item in cases]
        self.assertEqual(len(names), len(set(names)))
        conv = [item for item in cases if item["family"] == "resnet_conv"]
        self.assertEqual(len(conv), 46)
        for geometry in MODULE.CONV_GEOMETRIES:
            self.assertEqual(
                {item["parameters"]["relu"] for item in conv
                 if tuple(item["parameters"][key] for key in
                          ("input_channels", "output_channels", "spatial",
                           "kernel", "stride")) == geometry},
                {False, True})
        self.assertEqual(
            {item["name"] for item in cases if item["family"] != "resnet_conv"},
            {"resnet_max_pool_c64_s112_k3_st2_same",
             "resnet_add_relu_c256_s56", "resnet_add_relu_c512_s28",
             "resnet_add_relu_c1024_s14", "resnet_add_relu_c2048_s7",
             "resnet_bottleneck_projection_c256_s56",
             "resnet_bottleneck_identity_c256_s56",
             "resnet_reduce_mean_c2048_s7_axes23",
             "resnet_linear_k2048_n1000_bias1"})

    def test_persisted_contract_contains_no_hwx_payload(self):
        source = PATH.read_text()
        self.assertIn("om.run_case", source)
        self.assertNotIn("write_bytes(hwx", source)
        self.assertNotIn("copyfile", source)
        for item in MODULE.campaign():
            self.assertNotIn("/Users/", item["mil"])
            self.assertNotIn("/Volumes/", item["mil"])


if __name__ == "__main__":
    unittest.main()
