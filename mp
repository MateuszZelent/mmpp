Help on function open in module mmpp:

ooppeenn(base_path: str, **kwargs)
    Open and initialize an MMPP instance for the given directory path.
    
    This is the main entry point for using the mmpp library. It creates
    an MMPP instance that scans the provided directory for zarr files
    and builds a database for analysis.
    
    Parameters:
    -----------
    base_path : str
        Path to the directory containing zarr simulation files
    **kwargs : dict
        Additional keyword arguments passed to MMPP constructor:
        - max_workers : int, optional (default: 8)
            Maximum number of worker threads for scanning
        - database_name : str, optional (default: "mmpy_database")
            Name of the database file (without extension)
        - force : bool, optional (default: False)
            If True, force rescan even if database exists
    
    Returns:
    --------
    MMPP
        An initialized MMPP instance ready for analysis
    
    Examples:
    ---------
    >>> import mmpp as mp
    >>> db = mp.open("/path/to/simulation/data")
    >>> results = db.find(f0=2.15e+09)
    >>> results.mpl.plot("time", "my")
