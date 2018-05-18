# -*- mode: python -*-

block_cipher = None


a = Analysis(['bin\\refl1d_cli.py'],
             pathex=['C:\\projects\\refl1d'],
             binaries=[],
             datas=[ ('extra/*', 'extra' ) ],
             hiddenimports=['wx', 'bumps.gui.gui_app', 'bumps.cli', 'scipy._lib.messagestream'],
             hookspath=['.'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='refl1d_cli',
          debug=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='refl1d_cli')
