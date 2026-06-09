import collections
import numpy as np
import datetime
import time
import logging

log = logging.getLogger(__name__)


IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])

def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    return XyzTuple(*coords_xyz)

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a)
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))


def enumerateWithEstimate(
    iter,
    desc_str,
    start_ndx=0,
    print_ndx=4,
    backoff=2,
    iter_len=None,
):
    """
    一个简洁直观的进度估计生成器，不需要复杂的边界处理。
    """
    if iter_len is None:
        try:
            iter_len = len(iter)
        except TypeError:
            iter_len = None

    if iter_len is not None:
        log.warning("{} ----/{}, starting".format(desc_str, iter_len))
    else:
        log.warning("{}, starting".format(desc_str))

    start_ts = time.time()
    for current_ndx, item in enumerate(iter):
        yield (current_ndx, item)

        # 仅在到达指定的步数时打印估计信息，并通过 backoff 乘子逐步减少打印频率
        if current_ndx == print_ndx:
            duration_sec = time.time() - start_ts
            time_per_iter = duration_sec / (current_ndx + 1 - start_ndx)

            if iter_len is not None:
                remaining_sec = (iter_len - current_ndx - 1) * time_per_iter
                done_time = datetime.datetime.now() + datetime.timedelta(seconds=remaining_sec)

                # 格式化剩余时间 (H:MM:SS)
                hours, remainder = divmod(int(remaining_sec), 3600)
                mins, secs = divmod(remainder, 60)
                remaining_str = f"{hours}:{mins:02d}:{secs:02d}"

                done_time_str = done_time.strftime('%Y-%m-%d %H:%M:%S')
                log.warning("{} {}/{}, done at {}, {}".format(
                    desc_str, current_ndx, iter_len, done_time_str, remaining_str
                ))
            else:
                log.warning("{} {}, done".format(desc_str, current_ndx))

            if backoff:
                print_ndx *= backoff

    done_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log.warning("{} ----/{}, done at {}".format(desc_str, iter_len, done_time_str))
