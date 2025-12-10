# Refl1d v0.8.16 to v1.0.0 - Major Changes Summary

## Architecture & Code Modernization
- **Dataclass Refactor**: Complete restructuring of class definitions using Python dataclasses for serialization
- **Project Restructuring**: Major reorganization of the codebase layout and directory structure
- **Code Quality**: Added ruff linting, pre-commit hooks, and comprehensive code formatting across the entire project
- **Bumps 1.x Compatibility**: Updated to be compatible with the latest major release of the bumps fitting library

## Web Interface & Visualization
- **Webview GUI**: New web-based user interface, replacing the traditional (WXPython) desktop app, supporting
  - remote server on HPC with local client in browser
  - local server with browser client, like traditional desktop app
  - multiple simultaneous clients to same server
  - multiple simultaneous servers on same machine
- **Plotly Plotting**: Re-implemented core plots in plotly.js for web-based interactivity
- **Custom Plots**: Added ability to create and display custom plots in models
- **Interactive Profile Uncertainty**: The `align.py` script is replaced with an interactive panel in the webview
- **Model Builder**: Added a simple visual model builder interface for constructing reflectometry models

## Fitting & Analysis Improvements
- **Enhanced Magnetic Support**: 
  - Free magnetism interface enhancements
  - Fixed magnetic amplitude calculations and parameter handling
- **Oversampling Enhancements**: Improved Q-space oversampling for polarized neutron probes with configurable regions

## Data Handling & Compatibility
- **ORSO File Support**: Now able to load ORSO (Open Reflectometry Standards Organisation) data files
- **Serialization Improvements**: Better handling of dataclass and functional profile serialization
- **Duplicate Q Values**: Now supports datasets with duplicate Q values


## Installation & Distribution
- **Binary Installers**: New automated binary installer generation for easier deployment, including signed binaries for MacOS
- **Version Management**: Switch to versioningit for automatic version numbering from git tags
- **Python 3.13 Support**: Added compatibility with the latest Python version
- **Pure Python Package**: Transitioned to a pure Python package using numba for performance, removing the need for C-extensions

## CLI & Jupyter Integration
- **Webview CLI Options**: Unified command-line interface for interactive and batch-mode operations
- **Jupyter Integration**: Direct method for starting refl1d webview server from Jupyter notebooks


## All Pull Requests: Details
* Dataclass material refactor by @bmaranville in https://github.com/reflectometry/refl1d/pull/160
* Webview: separate API methods from server logic by @bmaranville in https://github.com/reflectometry/refl1d/pull/162
* [In development] Add simple builder by @mdoucet in https://github.com/reflectometry/refl1d/pull/164
* Remove ordering code and add tab and enter updates by @mdoucet in https://github.com/reflectometry/refl1d/pull/165
* Serialize stdlib dataclasses by @bmaranville in https://github.com/reflectometry/refl1d/pull/166
* Fix calc_Q for shared theta_offset by @bmaranville in https://github.com/reflectometry/refl1d/pull/175
* use separate rho_M_para and rho_M_perp profiles during contract_mag by @bmaranville in https://github.com/reflectometry/refl1d/pull/173
* Webview by @pkienzle in https://github.com/reflectometry/refl1d/pull/181
* Add ruff, pre-commit, and format existing files by @mdoucet in https://github.com/reflectometry/refl1d/pull/184
* Some cleanup by @glass-ships in https://github.com/reflectometry/refl1d/pull/183
* Get version number from tag using versioningit by @backmari in https://github.com/reflectometry/refl1d/pull/188
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/reflectometry/refl1d/pull/191
* update Zenodo badge to generic version by @bmaranville in https://github.com/reflectometry/refl1d/pull/192
* Docs build by @bmaranville in https://github.com/reflectometry/refl1d/pull/194
* remove approximation shortcut for magnetic_amplitude by @bmaranville in https://github.com/reflectometry/refl1d/pull/196
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/reflectometry/refl1d/pull/195
* Custom plots rebase by @bmaranville in https://github.com/reflectometry/refl1d/pull/199
* Monthly pre commit update by @bmaranville in https://github.com/reflectometry/refl1d/pull/200
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/reflectometry/refl1d/pull/202
* removing obsolete and non-working actions by @bmaranville in https://github.com/reflectometry/refl1d/pull/204
* Update release actions by @bmaranville in https://github.com/reflectometry/refl1d/pull/203
* switch to using defaults for versioningit by @bmaranville in https://github.com/reflectometry/refl1d/pull/209
* fix repr of LayerLimit and MagneticLayerLimit in flayer.py by @bmaranville in https://github.com/reflectometry/refl1d/pull/207
* remove getstate and setstate from Stack, and base argument from init by @bmaranville in https://github.com/reflectometry/refl1d/pull/201
* Windows shell launcher by @bmaranville in https://github.com/reflectometry/refl1d/pull/208
* add flags for enabling optional persistent path from shortcuts by @bmaranville in https://github.com/reflectometry/refl1d/pull/211
* Make LICENSE.txt consistent with documentation by @bmaranville in https://github.com/reflectometry/refl1d/pull/213
* re-enable oversampling on PolarizedQProbe by @bmaranville in https://github.com/reflectometry/refl1d/pull/214
* Profile uncertainty export with best by @bmaranville in https://github.com/reflectometry/refl1d/pull/215
* Model setstate shim by @bmaranville in https://github.com/reflectometry/refl1d/pull/216
* calculate the profile uncertainty in a thread by @bmaranville in https://github.com/reflectometry/refl1d/pull/222
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/reflectometry/refl1d/pull/217
* Formatting changes, update imports, other minor changes by @glass-ships in https://github.com/reflectometry/refl1d/pull/227
* Add frontend linting by @glass-ships in https://github.com/reflectometry/refl1d/pull/228
* allow background parameter of Probe to be negative by @bmaranville in https://github.com/reflectometry/refl1d/pull/218
* Update model.ts by @glass-ships in https://github.com/reflectometry/refl1d/pull/230
* Build against current bumps source (after bumps PR #188) by @bmaranville in https://github.com/reflectometry/refl1d/pull/233
* show y axis of reflectivity plots in scientific notation (10 to a power) by @bmaranville in https://github.com/reflectometry/refl1d/pull/234
* update bumps to 1.0.0a9 by @glass-ships in https://github.com/reflectometry/refl1d/pull/235
* Restructure and Reorganize project by @glass-ships in https://github.com/reflectometry/refl1d/pull/185
* Remove client cdn by @bmaranville in https://github.com/reflectometry/refl1d/pull/210
* fasta is part of periodictable and isn't needed in refl1d by @pkienzle in https://github.com/reflectometry/refl1d/pull/238
* Define global in vite.config.js by @glass-ships in https://github.com/reflectometry/refl1d/pull/239
* MAINT: remove unused variable by @andyfaff in https://github.com/reflectometry/refl1d/pull/240
* Add Refl1D migration between schemas by @bmaranville in https://github.com/reflectometry/refl1d/pull/241
* Add note about setting up pre-commit by @glass-ships in https://github.com/reflectometry/refl1d/pull/243
* Pytest plain asserts by @bmaranville in https://github.com/reflectometry/refl1d/pull/244
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/reflectometry/refl1d/pull/236
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/reflectometry/refl1d/pull/246
* Fix and test loading ORSO files by @jmborr in https://github.com/reflectometry/refl1d/pull/251
* Test types by @glass-ships in https://github.com/reflectometry/refl1d/pull/231
* Adjust imports in the wx gui to the new directory layout by @pkienzle in https://github.com/reflectometry/refl1d/pull/255
* Fix webview reflectivity ylabels by @bmaranville in https://github.com/reflectometry/refl1d/pull/254
* Merge FreeMagnetismInterface feature enhancement by @acaruana2009 in https://github.com/reflectometry/refl1d/pull/232
* use uv to set up test environment by @bmaranville in https://github.com/reflectometry/refl1d/pull/259
* Aguide convention fix by @bmaranville in https://github.com/reflectometry/refl1d/pull/253
* Use new Generic bumps FitProblem by @bmaranville in https://github.com/reflectometry/refl1d/pull/258
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/reflectometry/refl1d/pull/256
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/reflectometry/refl1d/pull/263
* Migrate builder fitproblem by @bmaranville in https://github.com/reflectometry/refl1d/pull/264
* Binary installers by @bmaranville in https://github.com/reflectometry/refl1d/pull/265
* Use git hash rather than branch name in the jupyter notebook header by @pkienzle in https://github.com/reflectometry/refl1d/pull/260
* Automatically hide uncertainty and custom plot tabs when they are not available by @bmaranville in https://github.com/reflectometry/refl1d/pull/267
* Webview cli fit options by @pkienzle in https://github.com/reflectometry/refl1d/pull/266
* support plotting ProbeSet data + theory in webview by @bmaranville in https://github.com/reflectometry/refl1d/pull/268
* automatically create ProbeSet object from multiple probes by @bmaranville in https://github.com/reflectometry/refl1d/pull/269
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/reflectometry/refl1d/pull/271
* Do oversampling on union of Q in PolarizedNeutronProbe by @bmaranville in https://github.com/reflectometry/refl1d/pull/274
* Bump astral-sh/setup-uv from 5 to 6 by @dependabot[bot] in https://github.com/reflectometry/refl1d/pull/275
* Fix profile uncertainty plot (for new bumps fitting.fit_state) by @bmaranville in https://github.com/reflectometry/refl1d/pull/277
* Fix QProbe serialization by @bmaranville in https://github.com/reflectometry/refl1d/pull/282
* Allow duplicate Q values in probes by @bmaranville in https://github.com/reflectometry/refl1d/pull/278
* Show refl1d version instead of bumps version from refl1d --version by @bmaranville in https://github.com/reflectometry/refl1d/pull/281
* Add python 3.13 to test matrix by @bmaranville in https://github.com/reflectometry/refl1d/pull/286
* Fix serialization of ProbeSet by @bmaranville in https://github.com/reflectometry/refl1d/pull/285
* remove vfs from refl1d since it is no longer in bumps by @pkienzle in https://github.com/reflectometry/refl1d/pull/289
* remove ZipFS loader that relied on now-removed bumps.vfs by @bmaranville in https://github.com/reflectometry/refl1d/pull/291
* Fix plot option for oversampling helper tool by @bmaranville in https://github.com/reflectometry/refl1d/pull/293
* install the fitplugin to bumps before starting cli by @bmaranville in https://github.com/reflectometry/refl1d/pull/290
* Fix breakage discovered by new pytest by @pkienzle in https://github.com/reflectometry/refl1d/pull/297
* Use trimmed MCMC for profile uncertainty plot by @pkienzle in https://github.com/reflectometry/refl1d/pull/298
* add method for starting refl1d webview server from jupyter by @bmaranville in https://github.com/reflectometry/refl1d/pull/299
* bugfix: cli flags converted to dashes but not updated in distributables by @bmaranville in https://github.com/reflectometry/refl1d/pull/300
* disable prettier-plugin-jsdoc until it is fixed by @bmaranville in https://github.com/reflectometry/refl1d/pull/302
* Fix model builder for get_model returning string by @bmaranville in https://github.com/reflectometry/refl1d/pull/304
* Install webview dependencies by default by @bmaranville in https://github.com/reflectometry/refl1d/pull/303
* Dataview panel cleanup by @bmaranville in https://github.com/reflectometry/refl1d/pull/308
* pass socket object as prop to main app by @bmaranville in https://github.com/reflectometry/refl1d/pull/284
* add serializable attributes to capture critical edge oversampling by @bmaranville in https://github.com/reflectometry/refl1d/pull/250
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/reflectometry/refl1d/pull/279
* Improve serialization of functional profiles by @pkienzle in https://github.com/reflectometry/refl1d/pull/276
* Fix running client in dev mode by @bmaranville in https://github.com/reflectometry/refl1d/pull/307
* use public api accessor probe.calc_Q instead of private _calc_Q by @bmaranville in https://github.com/reflectometry/refl1d/pull/305
* Convert data=(R,dR) to R=R, dR=dR for Probe by @pkienzle in https://github.com/reflectometry/refl1d/pull/314
* Bump actions/checkout from 4 to 5 by @dependabot[bot] in https://github.com/reflectometry/refl1d/pull/319
* Bump actions/download-artifact from 4 to 5 by @dependabot[bot] in https://github.com/reflectometry/refl1d/pull/316
* Fix profile uncertainty plots not updating by @bmaranville in https://github.com/reflectometry/refl1d/pull/320
* Check parameter name before updating in magnetism by @pkienzle in https://github.com/reflectometry/refl1d/pull/313
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/reflectometry/refl1d/pull/315
* Enable batch align and contour export by @pkienzle in https://github.com/reflectometry/refl1d/pull/317
* Fix control names in Profile Uncertainty panel by @bmaranville in https://github.com/reflectometry/refl1d/pull/321
* Allow vector parameters for functional layers by @pkienzle in https://github.com/reflectometry/refl1d/pull/311
* Add cloudpickle test to test_stack by @pkienzle in https://github.com/reflectometry/refl1d/pull/312
* Bump actions/upload-pages-artifact from 3 to 4 by @dependabot[bot] in https://github.com/reflectometry/refl1d/pull/322
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/reflectometry/refl1d/pull/323
* toggle background in data vs theory curve by @hoogerheide in https://github.com/reflectometry/refl1d/pull/309
* Add oversampled_regions to PolarizedNeutronProbe and PolarizedQProbe by @bmaranville in https://github.com/reflectometry/refl1d/pull/310
* Fix error shown at startup of client when no reflectivity data to plot by @bmaranville in https://github.com/reflectometry/refl1d/pull/325
* Refactor build_client script by @glass-ships in https://github.com/reflectometry/refl1d/pull/330

## New Contributors
* @backmari made their first contribution in https://github.com/reflectometry/refl1d/pull/188
* @pre-commit-ci[bot] made their first contribution in https://github.com/reflectometry/refl1d/pull/191
* @jmborr made their first contribution in https://github.com/reflectometry/refl1d/pull/251
* @dependabot[bot] made their first contribution in https://github.com/reflectometry/refl1d/pull/275

**Full Changelog**: https://github.com/reflectometry/refl1d/compare/v0.8.16...v1.0.0