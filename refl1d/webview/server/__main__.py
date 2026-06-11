if __name__ == "__main__":
    try:
        from bumps.webview.server.cli import main as new_main
    except ImportError:
        # CRUFT: bumps < 1.1
        from bumps.cli import main as new_main

    new_main()
