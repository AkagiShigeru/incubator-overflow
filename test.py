try:
    import lzma
except ImportError:
    from backports import lzma

# def ParseZippedXML(zf):
#     date_counts = defaultdict(int)
#     with lzma.open(zf, mode="r") as handle:
#         print handle.readline()
#     return date_counts
