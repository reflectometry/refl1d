// import { loadPyodideAndPackages } from './pyodide_worker.mjs';
import { expose, wrap, proxy } from 'comlink';
import { loadPyodide, version } from 'pyodide';
import type { PyodideInterface } from 'pyodide';
import type { PyProxy } from 'pyodide/ffi';
import { Signal } from './standalone_signal';
import type { Server as FitServer } from './standalone_fit_worker';

const DEBUG = true;

declare const REFL1D_WHEEL_FILE: string;
declare const BUMPS_WHEEL_FILE: string;

async function loadPyodideBase() {
    return await loadPyodide({
        indexURL: `https://cdn.jsdelivr.net/pyodide/v${version}/full/`
    });
}

async function loadBuiltins(pyodide: PyodideInterface) {
    await pyodide.loadPackage(["numpy", "scipy", "pytz", "h5py", "micropip"]);
}

async function doPipInstalls(pyodide: PyodideInterface) {
    await pyodide.runPythonAsync(`
    import micropip
    await micropip.install([
        "matplotlib",
        "plotly",
        "mpld3",
        "periodictable",
        "blinker",
        "dill",
    ])
    await micropip.install("./wheels/${BUMPS_WHEEL_FILE}")
    await micropip.install("./wheels/${REFL1D_WHEEL_FILE}", keep_going=True, deps=False)

    `);
}

async function createAPI(pyodide: PyodideInterface) {
    let api = await pyodide.runPythonAsync(`
    from typing import Any
    from bumps.webview.server import api
    import bumps.cli
    from refl1d.webview.server import api as refl1d_api
    import refl1d.fitplugin
    api.state.parallel = 0
    api.state.problem.serializer = "dataclass"
    print("api imported")

    import refl1d
    import asyncio
    import json
    import dill

    # setup backend:
    bumps.cli.install_plugin(refl1d.fitplugin)
    refl1d.use('c_ext')

    wrapped_api = {}

    def expose(method, method_name):
        def wrapper(args):
            pyargs = args.to_py() if args is not None else []
            result = method(*pyargs)
            return result

        return wrapper

    for method_name, method in api.REGISTRY.items():
        print("wrapping:", method_name)
        wrapped_api[method_name] = expose(method, method_name)

    async def worker_fit_progress_handler(serialized_event):
        event = dill.loads(serialized_event)
        await api._fit_progress_handler(event)

    async def worker_fit_complete_handler(serialized_event):
        event = dill.loads(serialized_event)
        await api._fit_complete_handler(event)

    wrapped_api["evt_fit_progress"] = expose(worker_fit_progress_handler, "evt_fit_progress")
    wrapped_api["evt_fit_complete"] = expose(worker_fit_complete_handler, "evt_fit_complete")

    from dataclasses import dataclass

    @dataclass
    class WorkerFitThread:
        fitclass: Any
        abort_event: Any
        problem: bumps.fitproblem.FitProblem
        options: dict
        parallel: int
        convergence_update: int
        uncertainty_update: int
        terminate_on_finish: bool

        _alive = True

        def join(self, timeout=None):
            self._alive = False

        def is_alive(self):
            return self._alive

        def run(self):
            print("running dummy fit thread")
            asyncio.create_task(self._run())

        def start(self):
            print("started dummy fit thread")
            self.run()
        
        async def _run(self):
            dumped = dill.dumps(self.problem)
            await api.emit("set_fit_thread_problem", dumped)
            await api.emit("start_fit_thread_fit", self.fitclass.id, self.options, self.terminate_on_finish)
            await api.emit("add_notification", {
                "title": "Fit Started",
                "content": f"Fit started with problem: {api.state.problem.fitProblem.name}",
                "timeout": 2000,
            });

    api.FitThread = WorkerFitThread

    wrapped_api
    `);
    return api;
}



const fit_worker = new Worker(new URL("./standalone_fit_worker.ts", import.meta.url), {type: 'module'});
const FitServerClass = wrap<typeof FitServer>(fit_worker);
const FitServerPromise = new FitServerClass();
type EventCallback = (message?: any) => any;

export class Server {
    handlers: { [signal: string]: EventCallback[] }
    nativefs: any;
    pyodide: PyodideInterface;
    api: PyProxy;
    initialized: Promise<void>;

    constructor() {
        this.handlers = {};
        this.nativefs = null;
        this.initialized = this.init();
    }

    async init() {
        await this.asyncEmit("server_startup_status", {status: "loading python", percent: 0});
        const pyodide = await loadPyodideBase();
        this.pyodide = pyodide;
        await this.asyncEmit("server_startup_status", {status: "initializing builtin modules", percent: 25});
        await loadBuiltins(pyodide);
        await this.asyncEmit("server_startup_status", {status: "installing pip dependencies", percent: 50});
        await doPipInstalls(pyodide);
        await this.asyncEmit("server_startup_status", {status: "loading bumps and refl1d", percent: 75});
        const api = await createAPI(pyodide);
        this.api = api;
        await this.asyncEmit("server_startup_status", {status: "api created", percent: 100});
        const fit_server = await FitServerPromise;
        const abort_fit_signal = new Signal("fit_abort_event");
        const fit_complete_signal = new Signal("fit_complete_event");
        await fit_server.set_signal(abort_fit_signal);
        await this.set_signal(abort_fit_signal);
        await fit_server.set_signal(fit_complete_signal);
        await this.set_signal(fit_complete_signal);
        const defineEmit = await pyodide.runPythonAsync(`
            def defineEmit(server):
                api.emit = server.asyncEmit;
            
            defineEmit
        `);
        await defineEmit(this);
        this.addHandler('set_fit_thread_problem', async (problem: any) => {
            const result = await fit_server.onAsyncEmit('set_problem', problem);
            console.log("set_fit_thread_problem result:", result);
        });
        this.addHandler('start_fit_thread_fit', async (...args: any[]) => {
            const result = await fit_server.onAsyncEmit('start_fit_thread', ...args);
            console.log("start_fit_thread_fit result:", result);
        });
        const fit_progress_handler = async (event: any) => {
            await this.onAsyncEmit('evt_fit_progress', event);
        }
        fit_server.addHandler('evt_fit_progress', proxy(fit_progress_handler));
        const fit_complete_handler = async (event: any) => {
            await this.onAsyncEmit('evt_fit_complete', event);
        }
        fit_server.addHandler('evt_fit_complete', proxy(fit_complete_handler));
    }

    async set_signal(signal_in: Signal) {
        const { name, buffer } = signal_in;
        const signal = new Signal(name, buffer);
        console.log("setting abort signal in worker", signal);
        const defineFitEvent = await this.pyodide.runPythonAsync(`
            def defineFitEvent(event):
                api.state.${name} = event.to_py();
            
            defineFitEvent
        `);
        await defineFitEvent(signal);
    }

    async addHandler(signal: string, handler: EventCallback) {
        const signal_handlers = this.handlers[signal] ?? [];
        signal_handlers.push(handler);
        if (DEBUG) {
            console.log(`adding handler: ${signal}`);
        }
        if (signal === 'connect') {
            await this.initialized;
            await FitServerPromise;
            await handler();
        }
        this.handlers[signal] = signal_handlers;
    }

    async removeHandler(signal: string, handler: EventCallback) {
        let signal_handlers = this.handlers[signal] ?? [];
        signal_handlers = signal_handlers.filter((h) => {
            if (h === handler) {
                console.log('matching worker handler found, removing: ', handler);
                return false;
            }
            return true;
        })
        this.handlers[signal] = signal_handlers;
    }

    async mount(dirHandle: FileSystemDirectoryHandle) {
        // const dirHandle = await self.showDirectoryPicker();
        console.log({dirHandle});   
        const nativefs = await this.pyodide.mountNativeFS("/home/pyodide/user_mount", dirHandle);
        this.nativefs = nativefs;
        await this.pyodide.runPythonAsync(`
        import os
        os.chdir("/home/pyodide/user_mount")

        `);
    }

    async syncFS() {
        let r = await this.nativefs?.syncfs?.();
    }

    async asyncEmit(signal: string, ...args: unknown[]) {
        // this is for emit() calls from the python server
        const js_args = args.map((arg) => {
            return arg?.toJs?.({dict_converter: Object.fromEntries}) ?? arg;
        });
        const handlers = this.handlers[signal] ?? [];
        for (let handler of handlers) {
            handler(...js_args);
        }
    }

    async onAsyncEmit(signal: string, ...args: any[]) {
        // this is for emit() calls from the client
        await this.initialized; // api ready after this...
        const callback = (args[args.length - 1] instanceof Function) ? args.pop() : null;
        const result = await this.api.get(signal)(args);
        const jsResult = result?.toJs?.({dict_converter: Object.fromEntries}) ?? result;
        if (callback !== null) {
            await callback(jsResult);
        }
        return jsResult;
    }

}

expose(Server);
