if __name__ == "__main__":
    import warnings
    from .cli import main as new_main

    warnings.warn(
        "starting the application from the refl1d.webview.server entry point is deprecated - use refl1d entry point"
    )
    new_main()
