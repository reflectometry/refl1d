# refl1d-web-gui

Web-based client for Refl1D webview GUI

## Setting up development environment

This package depends on the `bumps-webview-client` package, which can be installed in a few ways:

### Install bumps client from NPM
This will get the most recent version that has been published to npmjs.com (or update an existing version)

```sh
npm install bumps-webview-client@latest
```

### Use current git version
This gives you the most up-to-date version, but requires a little setup

_N.B. For the example below, `bumps` and `refl1d` have been cloned to the `~/dev/` folder.  Your
install location will be different..._

1. Create a conda environment
    ```sh
    conda create -n refl1d-dev python nodejs
    conda activate refl1d-dev
    ```

1. Clone the bumps repository (only need to do once)
    ```sh
    cd ~/dev
    git clone https://github.com/bumps/bumps
    cd bumps
    git checkout webview
    ```
1. Install `bumps` library in development mode
    ```sh
    pip install -e .
    pip install -r webview-requirements
    ```
1. Navigate to the webview client folder
    ```sh
    cd ~/dev/bumps/bumps/webview/client
    ```
1. Install dependencies
    ```sh
    npm install
    ```
1. Add `bumps-webview-client` to npm links
    ```sh
    npm link
    ```
1. Clone the refl1d repository
    ```sh
    cd ~/dev
    git clone https://github.com/reflectometry/refl1d
    cd refl1d
    git checkout webview
    ```
1. Install `refl1d` python library
    ```sh
    pip install -e .
    ```
1. Navigate to webview client folder
    ```sh
    cd ~/dev/refl1d/refl1d/webview/client
    ```
1. Install dependencies
    ```sh
    npm install
    ```
1. Link to local `bumps-webview-client` folder
   (this overwrites the bumps-webview-client installation from `npm install` with a hard link to the local folder)
    ```sh
    npm link bumps-webview-client
    ```
1. Build the client
    ```sh
    npm run build
    ```

### Rebuilding after changes
Now you can run the server, and it will use this locally built client.  After changes to the source code (to incorporate new client features):
```sh
cd ~/dev/bumps
git pull
cd ~/dev/refl1d
git pull
cd ~/dev/refl1d/refl1d/webview/client
npm run build
```

### Hot-reloading client preview
This mode is useful for rapid prototyping (esp. trying to fix styling in the client)

```sh
cd ~/dev/refl1d/refl1d/webview/client
npx vite
```
... this will start a local server for the rendered client at something like http://localhost:5173 (a link will appear in the console).

In a different terminal, you have to also start the python server, which will be listening on a different port, e.g.
```sh
python -m refl1d.webview.server --port 8888
```

Then, in your browser, you would navigate to the server that is rendering the client code, but also passing information on where the API server is located, e.g.

https://localhost:5173/?server=http://localhost:8888

Changes made to the client code, e.g. the .vue files, will be immediately reflected in the running client.

_(When using a pre-built client, the python API server will also serve the static client files, but for hot-reloading we want those two services separated.)_

## Recommended IDE Setup

[VSCode](https://code.visualstudio.com/) + [Volar](https://marketplace.visualstudio.com/items?itemName=Vue.volar) (and disable Vetur) + [TypeScript Vue Plugin (Volar)](https://marketplace.visualstudio.com/items?itemName=Vue.vscode-typescript-vue-plugin).

## Customize configuration

See [Vite Configuration Reference](https://vitejs.dev/config/).

## Project Setup

```sh
npm install
```

### Compile and Hot-Reload for Development

```sh
npm run dev
```

### Compile and Minify for Production

```sh
npm run build
```

# Publishing new client versions:
(...after checking to make sure there aren't extraneous files in this folder)
```sh
npm version patch
npm publish
```

and then
```sh
git commit package.json -m "webview client version bump"
git pull
git push
```
