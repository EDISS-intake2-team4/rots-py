import sys

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

def main():
    """Calculate the ROTS statistic"""
    print(f'ROTS-py version {metadata.version("rots-py")}')

if __name__ == "__main__":
    main()