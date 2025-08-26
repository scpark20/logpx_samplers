# minimal_inception_pool3.py
import os, random
import numpy as np
import requests
import tensorflow.compat.v1 as tf

INCEPTION_V3_URL  = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
INCEPTION_V3_PATH = "classify_image_graph_def.pb"
FID_POOL_NAME     = "pool_3:0"   # returns 2048-d per image

class InceptionPool3:
    def __init__(self, session: tf.Session = None, batch_size: int = 64):
        tf.disable_eager_execution()
        if session is None:
            cfg = tf.ConfigProto(allow_soft_placement=True)
            cfg.gpu_options.allow_growth = True
            session = tf.Session(config=cfg)
        self.sess = session
        self.batch_size = batch_size
        with self.sess.graph.as_default():
            self.images = tf.placeholder(tf.float32, [None, None, None, 3])
            self.pool3 = self._build(self.images)

    def extract(self, images: np.ndarray, batch_size: int = None) -> np.ndarray:
        assert images.ndim == 4 and images.shape[-1] == 3
        bs = batch_size or self.batch_size
        outs = []
        for i in range(0, len(images), bs):
            batch = images[i:i+bs].astype(np.float32, copy=False)
            feat = self.sess.run(self.pool3, {self.images: batch})
            outs.append(feat.reshape([feat.shape[0], -1]))  # -> [B,2048]
        return np.concatenate(outs, 0)

    # ---- internals ----
    def _build(self, x):
        self._ensure_model()
        prefix = f"inc_{random.randrange(2**32)}"
        with open(INCEPTION_V3_PATH, "rb") as f:
            gd = tf.GraphDef(); gd.ParseFromString(f.read())
        (pool3,) = tf.import_graph_def(
            gd, input_map={"ExpandDims:0": x}, return_elements=[FID_POOL_NAME], name=prefix
        )
        return pool3

    def _ensure_model(self):
        if os.path.exists(INCEPTION_V3_PATH): return
        r = requests.get(INCEPTION_V3_URL, stream=True); r.raise_for_status()
        tmp = INCEPTION_V3_PATH + ".tmp"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk: f.write(chunk)
        os.replace(tmp, INCEPTION_V3_PATH)