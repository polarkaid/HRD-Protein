"""Download released weights (or use --checkpoint) and evaluate."""
from hrd.paths import initialize

if __name__ == '__main__':
    initialize()
    from hrd.pipeline import main
    main('test')
