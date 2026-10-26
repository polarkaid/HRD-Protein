"""Download, prepare, train five seeds and evaluate both benchmarks."""
from hrd.paths import initialize

if __name__ == '__main__':
    initialize()
    from hrd.pipeline import main
    main('train')
