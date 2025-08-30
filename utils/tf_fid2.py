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

def _local_rank():
    import os
    # torchrun이 설정하는 LOCAL_RANK 사용, 없으면 0
    return int(os.environ.get("LOCAL_RANK", "0"))

def _get_tf_inception():
    global _TF_SESS, _TF_INPUT, _TF_POOL3
    if _TF_SESS is not None:
        return _TF_SESS, _TF_INPUT, _TF_POOL3

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    _ensure_tf_pb()

    # ★ 핵심: 이 세션에선 "로컬 랭크 한 개의 GPU만 보이게" 제한
    #   visible_device_list는 "프로세스가 보이는 GPU 인덱스" 기준 (CUDA_VISIBLE_DEVICES 재설정 불필요)
    lr = _local_rank()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(lr)   # ← 여기서 1개만 노출

    graph = tf.Graph()
    _TF_SESS = tf.Session(graph=graph, config=config)

    with graph.as_default():
        # visible_device_list로 1개만 보이게 했으므로 '/GPU:0'로 잡아도 안전(그 1개가 0으로 re-map됨)
        with tf.device('/GPU:0'):
            _TF_INPUT = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            with open(_TF_PB_PATH, "rb") as f:
                gd = tf.GraphDef(); gd.ParseFromString(f.read())
            (_TF_POOL3,) = tf.import_graph_def(
                gd,
                input_map={"ExpandDims:0": _TF_INPUT},  # NHWC float32 [0,255]
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
