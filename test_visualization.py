from pathlib import Path
import h5py, numpy as np
import matplotlib.pyplot as plt
from gluefactory.visualization.viz2d import plot_images, plot_keypoints, plot_matches
import imageio.v2 as iio


exp_dir = Path("/home/lwangax/ws/matcher/glue_factory/outputs/results/hpatches/superpoint+lightglue-official")
with h5py.File(exp_dir / "predictions.h5", "r") as f:
    print("num samples:", len(f.keys()))
    name = next(iter(f.keys()))          # pick first sample
    print("sample name:", name)
    print("datasets:", list(f[name].keys()))
    print("keys:", list(f[name]["0.ppm"].keys()))
    kp0 = f[name]["0.ppm"]["keypoints0"][...]     # (N,2)
    kp1 = f[name]["0.ppm"]["keypoints1"][...]     # (M,2)
    m0  = f[name]["0.ppm"]["matches0"][...]       # (N,)
    print("kp0 shape:", kp0.shape)
    print("kp1 shape:", kp1.shape)
    print("m0 shape:", m0.shape)
    print("kp0:", kp0)
    print("kp1:", kp1)
    print("m0:", m0)

    # You need the original images; infer or build their paths for the sample `name`
    # Example for HPatches (adjust paths if needed):
    # name is often like "sequence/idx"
seq = name
img0_path = f"/srv/ws_data/gluefactory_data/data/hpatches-sequences-release/{seq}/1.ppm"
img1_path = f"/srv/ws_data/gluefactory_data/data/hpatches-sequences-release/{seq}/2.ppm"
img0 = iio.imread(img0_path)
img1 = iio.imread(img1_path)

valid = m0 > -1
kpm0 = kp0[valid]
kpm1 = kp1[m0[valid]]

plot_images([img0, img1])
plot_keypoints([kp0, kp1], colors="royalblue", ps=2)
plot_matches(kpm0, kpm1, lw=0.5, ps=0.0, a=0.8)
plt.show(block=True)
# plt.savefig(f"{splitext(conf_file)[0]}_point_matches.svg")

