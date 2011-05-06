if __name__ == "__main__":
    from refl1d.gui.util import publish, subscribe
    import numpy, scipy, matplotlib, pytz
    import periodictable, pyparsing, refl1d.names
    import dream

    import multiprocessing
    multiprocessing.freeze_support()

    from refl1d.gui.gui_app import main
    main()
