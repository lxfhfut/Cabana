"""Shared per-image (and per-batch) pipeline stages.

Each function here is a pure-ish unit invoked by both ``Cabana`` (single image)
and ``BatchCabana`` (folder of images). Keeping the science in one place
prevents the two pipelines from drifting apart.
"""

from .hdm import HDM


def run_hdm(args, source_path, hdm_dir, ext='.png'):
    """Run HDM quantification on a single image path or a directory of images.

    Parameters
    ----------
    args : dict
        Loaded parameters YAML — needs Quantification + Detection sections.
    source_path : str
        Either a single image path or a directory containing images.
    hdm_dir : str
        Output directory for HDM artifacts and ``ResultsHDM.csv``.
    ext : str
        Image extension(s) to process. Default ``.png``.

    Returns
    -------
    pandas.DataFrame
        Per-image HDM quantification results (the same object stored on
        ``HDM.df_hdm`` after the call).
    """
    hdm = HDM(
        max_hdm=args["Quantification"]["Maximum Display HDM"],
        sat_ratio=args["Quantification"]["Contrast Enhancement"],
        dark_line=args["Detection"]["Dark Line"],
    )
    hdm.quantify_black_space(source_path, hdm_dir, ext=ext)
    return hdm.df_hdm
