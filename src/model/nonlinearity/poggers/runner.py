from .iterator import CentralIterator
from .processor import HD5Processor
from tqdm import tqdm


def runner(iterator: CentralIterator, processor: HD5Processor):
    processor.start()
    with tqdm(total=iterator.total_iterations, desc="Processing Files") as pbar:
        for i_ctx in iterator:
            pbar.set_description(f"Processing fill {i_ctx.fill}, run {i_ctx.run}")
            processor.process_iteration(i_ctx)
            pbar.update(1)
    processor.end()
        