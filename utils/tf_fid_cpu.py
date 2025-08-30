# utils/tf_fid2.py

_TF_SESS = _TF_INPUT = _TF_POOL3 = None
_TF_PB_URL  = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
_TF_PB_PATH = "classify_image_graph_def.pb"

def _ensure_tf_pb():
    import os, urllib.request
    if os.path.exists(_TF_PB_PATH):
        return
    tmp = _TF_PB_PATH + ".tmp"
    urllib.request.urlretrieve(_TF_PB_URL, tmp)
    os.replace(tmp, _TF_PB_PATH)

def _get_tf_inception():
    global _TF_SESS, _TF_INPUT, _TF_POOL3
    if _TF_SESS is not None:
        return _TF_SESS, _TF_INPUT, _TF_POOL3

    # ★ TF가 GPU를 전혀 못 보게(가장 확실)
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # TF import 전에 GPU 숨김
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # (선택) 로그 줄이기

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    _ensure_tf_pb()

    # ★ 세션도 CPU 강제
    graph = tf.Graph()
    config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)
    _TF_SESS = tf.Session(graph=graph, config=config)

    with graph.as_default():
        # ★ 연산 배치도 CPU로 고정
        with tf.device('/cpu:0'):
            _TF_INPUT = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            with open(_TF_PB_PATH, "rb") as f:
                gd = tf.GraphDef(); gd.ParseFromString(f.read())
            (_TF_POOL3,) = tf.import_graph_def(
                gd,
                input_map={"ExpandDims:0": _TF_INPUT},   # NHWC float32 in [0,255]
                return_elements=["pool_3:0"],
                name="inc",
            )
    return _TF_SESS, _TF_INPUT, _TF_POOL3

def tf_inception_encode(pil_images, batch_size=64):
    import numpy as np
    sess, image_in, pool3 = _get_tf_inception()
    arrs = [np.asarray(img).astype(np.float32) for img in pil_images]  # HWC [0,255]
    feats = []
    for i in range(0, len(arrs), batch_size):
        batch = np.stack(arrs[i:i+batch_size], 0)
        pool  = sess.run(pool3, feed_dict={image_in: batch})
        feats.append(pool.reshape([pool.shape[0], -1]))  # [N,2048]
    return np.concatenate(feats, 0)
