if __name__ == "__main__":
    import numpy, scipy, matplotlib, pytz
    import periodictable, pyparsing, refl1d.names
    import dream

    import multiprocessing
    multiprocessing.freeze_support()

    from bumps.gui.gui_app import main
    main()
