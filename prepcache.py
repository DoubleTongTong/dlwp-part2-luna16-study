import sys
import argparse
import logging

from dsets import LunaDataset
from util import enumerateWithEstimate

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)

class LunaPrepCacheApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        # 初始化完整数据集以填充缓存
        dataset = LunaDataset()
        log.info("Total samples to cache: {}".format(len(dataset)))

        # 利用 DataLoader 的多线程并发加载数据，自动触发 dsets.py 中的缓存机制
        from torch.utils.data import DataLoader
        batch_size = 64
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
        )

        # 使用进度生成器来直观展示缓存预热的 ETA
        for batch_ndx, batch_tup in enumerateWithEstimate(
            dataloader,
            "Caching",
            print_ndx=1,
            backoff=2,
        ):
            pass

        log.info("Caching complete!")

if __name__ == '__main__':
    LunaPrepCacheApp().main()
